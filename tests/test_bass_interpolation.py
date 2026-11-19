import copy
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from stemgenrt.bass_interpolation import interpolate_bass_emas
from stemgenrt.checkpoint import ParameterEMA, state_sha256
from stemgenrt.specialist import SpecialistSeparator


def matched_emas():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(118)
        base = SpecialistSeparator(source="bass", band_width=16, global_width=16, layers=1)
    base.provenance.update(training_updates=1, current_stage_updates=1)
    average = ParameterEMA(base, decay=0.0)
    average.update(base, step=1)
    parent = average.inference_copy(base)
    continued = copy.deepcopy(parent).train().requires_grad_(True)
    continued.provenance.update(parent_training_updates=1, current_stage_updates=1, training_updates=2)
    average = ParameterEMA(continued, decay=0.5)
    with torch.no_grad():
        for tensor in continued.parameters():
            tensor.add_(0.015)
    average.update(continued, step=1)
    return parent, average.inference_copy(continued)


def test_interpolation_matches_exact_endpoints_and_rounds_once():
    parent, continued = matched_emas()
    snapshots = [state_sha256(m.state_dict()) for m in (parent, continued)]
    provenance = [copy.deepcopy(m.provenance) for m in (parent, continued)]
    for fraction, expected in zip((0., 1.), snapshots, strict=True):
        result = interpolate_bass_emas(parent, continued, trained_fraction=fraction)
        assert state_sha256(result.state_dict()) == expected
    result = interpolate_bass_emas(parent, continued, trained_fraction=0.25)
    for name, value in result.named_parameters():
        start, end = parent.get_parameter(name).double(), continued.get_parameter(name).double()
        assert torch.equal(value, (start + 0.25 * (end - start)).float())
        assert value.data_ptr() != continued.get_parameter(name).data_ptr()
        assert not value.requires_grad
    assert all(torch.equal(b, parent.get_buffer(n)) for n, b in result.named_buffers())
    assert result.provenance["checkpoint_weight_role"] == "interpolated_inference"
    assert "weight_averaging" not in result.provenance
    record = result.provenance["bass_ema_interpolation"]
    assert record["output_state_sha256"] == state_sha256(result.state_dict())
    assert [record["parent_ema_state_sha256"], record["continued_ema_state_sha256"]] == snapshots
    assert record["additional_optimizer_updates"] == 0
    audio = torch.zeros(1, 2, 256)
    with torch.inference_mode():
        values = result.render(audio).deployed
    assert values.shape == (1, 1, 2, 256) and torch.isfinite(values).all()
    assert [state_sha256(m.state_dict()) for m in (parent, continued)] == snapshots
    assert [m.provenance for m in (parent, continued)] == provenance


@pytest.mark.parametrize("fraction", [-0.1, 1.1, float("nan"), float("inf"), True])
def test_invalid_fraction(fraction):
    parent, continued = matched_emas()
    with pytest.raises(ValueError, match="trained_fraction"):
        interpolate_bass_emas(parent, continued, trained_fraction=fraction)


@pytest.mark.parametrize("change", ["base", "role", "parameters", "buffer", "train", "updates"])
def test_rejects_invalid_endpoints(change):
    parent, continued = matched_emas()
    if change == "base":
        continued.provenance["weight_averaging"]["base_state_sha256"] = "00" * 32
    elif change == "role":
        parent.provenance["checkpoint_weight_role"] = "raw_optimizer_endpoint"
    elif change == "parameters":
        next(continued.parameters()).add_(1)
    elif change == "buffer":
        next(continued.buffers()).add_(1)
    elif change == "train":
        parent.train()
    else:
        continued.provenance["parent_training_updates"] += 1
    with pytest.raises(ValueError):
        interpolate_bass_emas(parent, continued, trained_fraction=0.25)


def test_export_rejects_incomplete_blend_arguments(tmp_path):
    script = Path(__file__).resolve().parents[1] / "scripts/export_independent_model.py"
    cmd = [sys.executable, str(script), "--output", str(tmp_path/"export"), "--expected-updates", "1",
           "--bass-trained-fraction", "0.5"]
    for name in ("drums", "bass", "vocals"):
        cmd.extend(["--"+name+"-checkpoint", "unused", "--"+name+"-sha256", "0"*64])
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 2 and "interpolation requires" in result.stderr
    assert not (tmp_path/"export").exists()


def test_panel_selection_applies_guards_quality_and_tie_breaks():
    from stemgenrt.bass_interpolation_eval import select_bass_fraction
    limits = {"bass_sir": 0.1, "ordinary_absent_dbfs": 0.5, "target_absent_to_input_db": 0.5,
              "target_only_projection_db": 0.1, "target_only_sdr": 0.1}
    base = {k: 0. for k in (*limits, "bass_sdr", "other_sdr", "bass_low_sdr")}
    small = {**base, "bass_sdr": 0.2, "other_sdr": 0.11, "bass_low_sdr": 0.16}
    high = {**small, "bass_sdr": 0.3, "other_sdr": 0.25, "target_only_sdr": -0.11}
    inputs = {0.: base, 0.5: small, 0.25: small, 0.75: high}
    chosen, scores = select_bass_fraction(inputs, [0.75, 0.5, 0.25], limits)
    assert chosen == 0.25 and scores[0.25]["min_sdr_gain"] == 0.11
    assert scores[0.75]["worst_overrun"] == pytest.approx(0.1)
    high["target_only_sdr"] = -0.1
    assert select_bass_fraction(inputs, [0.25, 0.75], limits)[0] == 0.75
    high["ordinary_absent_dbfs"] = 1.0
    small["target_only_sdr"] = -0.15
    assert select_bass_fraction(inputs, [0.75, 0.5], limits)[0] == 0.5
    with pytest.raises(ValueError):
        select_bass_fraction(inputs, [1.0], limits)


def test_training_panel_reuses_peers_and_aligns_controlled_views(tmp_path):
    import numpy as np
    import soundfile as sf
    from stemgenrt.banded import BandSeparator
    from stemgenrt.bass_interpolation_eval import score_bass_interpolation_track
    from stemgenrt.evaluation import EvaluationTrack
    from stemgenrt.independent import IndependentStems, SOURCE_ORDER

    parent, continued = matched_emas()
    models = {0.0: parent, 0.25: interpolate_bass_emas(parent, continued, trained_fraction=0.25)}
    drums = SpecialistSeparator(source="drums", band_width=16, global_width=16, layers=1).eval()
    vocals = BandSeparator(sources=("vocals",), band_width=16, global_width=16, layers=1).eval()
    release = IndependentStems(drums=drums, bass=parent, vocals=vocals)
    rng = np.random.default_rng(711)
    stems = rng.normal(0., 0.05, (4, 2, 44100+377)).astype("float32")
    for name, data in zip(("mixture", *SOURCE_ORDER), (stems.sum(0), *stems), strict=True):
        sf.write(tmp_path/(name+".wav"), data.T, 44100, subtype="FLOAT")
    track = EvaluationTrack("training example", tmp_path/"mixture.wav",
                            tuple(tmp_path/(s+".wav") for s in SOURCE_ORDER), ((173, 44100+173),))
    scores, inputs = score_bass_interpolation_track(track, release, models)
    assert set(inputs) == {"target_only", "target_absent"} and all(len(v) == 64 for v in inputs.values())
    for peer in ("drums", "vocals"):
        assert scores[0.0]["ordinary"]["per_stem"][peer] == scores[0.25]["ordinary"]["per_stem"][peer]
    for system in scores.values():
        assert system["ordinary"]["excerpts"][0]["estimate_start"] == 173+128
        assert all(v["input_active_windows"] == 1 for v in system["controlled"]["scores"].values())
