import copy
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from stemgenrt.bass_replicate_average import average_bass_replicates
from stemgenrt.checkpoint import ParameterEMA, file_sha256, load_model, save_training_checkpoint, state_sha256
from stemgenrt.specialist import SpecialistSeparator


def make_endpoint(path, *, start, direction, seed=29, identity="same", scheduled_steps=1, lr=0.01):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = SpecialistSeparator(source="bass", band_width=16, global_width=16, layers=1).train()
    model.provenance["parent_training_updates"] = 4
    ema = ParameterEMA(model, decay=0.4)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, foreach=False)
    for parameter in model.parameters():
        parameter.grad = torch.full_like(parameter, direction)
    optimizer.step()
    ema.update(model, step=1)
    config = {"model_family": "specialist", "target_source": "bass", "precision": "fp32",
              "band_width": 16, "band_global_width": 16, "band_layers": 1,
              "specialist_feature_n_fft": model.feature_n_fft, "specialist_waveform_basis": model.waveform_basis,
              "batch_size": 16, "steps": scheduled_steps, "data_start": start, "lr": lr}
    return save_training_checkpoint(path, model, optimizer, ema, step=1, next_sample_index=start+16,
                                    config=config, data_identity={"train": identity}, compressed=False)


def average(a, b):
    return average_bass_replicates(a["path"], b["path"], first_sha256=a["sha256"], second_sha256=b["sha256"])


def test_full_training_checkpoints_give_one_symmetric_frozen_bass_model(tmp_path):
    a = make_endpoint(tmp_path/"early.pt", start=1008, direction=-1.)
    b = make_endpoint(tmp_path/"late.pt", start=1024, direction=1.)
    parents = [load_model(s["path"], expected_sha256=s["sha256"], role="ema") for s in (a, b)]
    sources = [copy.deepcopy(m.provenance) for m in parents]
    rng = torch.get_rng_state().clone()
    candidate = average(a, b)
    assert torch.equal(torch.get_rng_state(), rng)
    assert not candidate.training and not any(p.requires_grad for p in candidate.parameters())
    assert candidate.architecture_metadata == parents[0].architecture_metadata
    for name, value in candidate.named_parameters():
        reference = (parents[0].get_parameter(name).double() + parents[1].get_parameter(name).double()) / 2
        assert torch.equal(value, reference.float())
    assert all(torch.equal(b, parents[0].get_buffer(n)) for n, b in candidate.named_buffers())
    record = candidate.provenance["bass_replicate_average"]
    assert record["additional_optimizer_updates"] == 0 and record["output_state_sha256"] == state_sha256(candidate.state_dict())
    assert [e["data_range"] for e in record["replicates"]] == [[1008, 1024], [1024, 1040]]
    assert [e["checkpoint_sha256"] for e in record["replicates"]] == [a["sha256"], b["sha256"]]
    swapped = average(b, a)
    assert swapped.provenance == candidate.provenance and state_sha256(swapped.state_dict()) == record["output_state_sha256"]
    with torch.inference_mode():
        assert candidate.render(torch.zeros(1, 2, 256)).deployed.shape == (1, 1, 2, 256)
    assert [m.provenance for m in parents] == sources
    assert all(file_sha256(s["path"]) == s["sha256"] for s in (a, b))


@pytest.mark.parametrize("changes,message", [
    ({"start": 1008}, "ranges overlap"),
    ({"seed": 39}, "parent state"),
    ({"identity": "different"}, "dataset identities"),
    ({"lr": 0.006}, "training recipes"),
    ({"scheduled_steps": 2}, "complete FP32"),
])
def test_rejects_unmatched_or_incomplete_continuations(tmp_path, changes, message):
    a = make_endpoint(tmp_path/"early.pt", start=1008, direction=-1.)
    b = make_endpoint(tmp_path/"late.pt", **{"start": 1024, "direction": 1., **changes})
    with pytest.raises(ValueError, match=message):
        average(a, b)
    with pytest.raises(ValueError, match="SHA-256"):
        average(a, {**b, "sha256": "0" * 64})


@pytest.mark.parametrize("extra,message", [
    (["--bass-replicate-checkpoint", "second.pt"], "second checkpoint and its SHA-256"),
    (["--bass-replicate-checkpoint", "second.pt", "--bass-replicate-sha256", "2" * 64,
      "--bass-blend-parent", "parent.pt", "--bass-blend-parent-sha256", "3" * 64,
      "--bass-trained-fraction", "0.5"], "single Bass parameter-averaging method"),
])
def test_export_rejects_ambiguous_replicate_arguments_before_creating_output(tmp_path, extra, message):
    runner = Path(__file__).resolve().parents[1] / "scripts/export_independent_model.py"
    out = tmp_path/"export"
    args = [sys.executable, str(runner), "--output", str(out), "--expected-updates", "1"]
    for stem in ("drums", "bass", "vocals"):
        args.extend([f"--{stem}-checkpoint", "unused.pt", f"--{stem}-sha256", "1" * 64])
    result = subprocess.run([*args, *extra], capture_output=True, text=True)
    assert result.returncode == 2 and message in result.stderr and not out.exists()
