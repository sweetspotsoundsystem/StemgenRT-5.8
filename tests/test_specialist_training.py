"""Target-aware source views and complete optimizer, EMA and RNG recovery."""
from dataclasses import replace
import random

import numpy as np
import pytest
import torch

from stemgenrt import checkpoint, losses
from stemgenrt.checkpoint import ParameterEMA, load_model, load_training_checkpoint, save_training_checkpoint, state_sha256
from stemgenrt.model import render_scored_context
from stemgenrt.specialist import SpecialistSeparator
from stemgenrt.trainer import TrainingConfig
from test_checkpoint import tree_fingerprint


@pytest.mark.parametrize("source,index", [("bass", 1), ("drums", 0), ("vocals", 2)])
def test_source_views_remove_and_preserve_the_named_stem_without_mutation(source, index):
    targets = torch.randn(16, 4, 2, 384)
    mixture = targets.sum(1)
    before = targets.clone()
    audio, views = losses.source_views(mixture, targets, target_source=source)
    assert torch.equal(targets, before)
    assert views[0, index].count_nonzero() == 0
    assert torch.equal(audio[1], targets[15, index])
    assert torch.equal(audio, views.sum(1))
    for i in range(4):
        if i != index:
            assert views[1, i].count_nonzero() == 0
            assert torch.equal(views[0, i], targets[14, i])


@pytest.mark.parametrize("source", ["bass", "drums"])
def test_specialist_checkpoint_restores_identical_next_update_and_inference(tmp_path, source):
    random.seed(16)
    np.random.seed(16)
    torch.manual_seed(16)
    model = SpecialistSeparator(source=source, band_width=16, global_width=32, layers=2).train()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, foreach=False)
    ema = ParameterEMA(model)
    config = {"model_family": "specialist", "band_width": 16, "band_global_width": 32, "band_layers": 2,
              "target_source": source, "specialist_feature_n_fft": model.feature_n_fft,
              "specialist_waveform_basis": model.waveform_basis,
              "steps": 3, "batch_size": 1, "data_start": 0, "precision": "fp32"}
    def advance(model, optimizer, ema, step):
        audio = torch.randn(1, 2, 768) * (.02 + .001 * random.random() + .001 * np.random.random())
        truth = torch.randn(1, 1, 2, 640) * .01
        optimizer.zero_grad(set_to_none=True)
        output = render_scored_context(model, audio, warmup_samples=128, carry_state=True)
        loss = (output.deployed - truth).square().mean()
        loss.backward()
        optimizer.step()
        ema.update(model, step=step)
        return float(loss.detach())
    advance(model, optimizer, ema, 1)
    saved = save_training_checkpoint(tmp_path / "specialist.pt", model, optimizer, ema, step=1,
                                     next_sample_index=1, config=config, data_identity={"train": "fixed"})
    expected_loss = advance(model, optimizer, ema, 2)
    expected = (state_sha256(model.state_dict()), tree_fingerprint(optimizer.state_dict()),
                tree_fingerprint(ema.state_dict(model)), tree_fingerprint(checkpoint._rng_state()))
    restored = load_training_checkpoint(saved["path"], sha256=saved["sha256"], config=config,
                                        data_identity={"train": "fixed"})
    assert type(restored.model) is SpecialistSeparator and restored.model.sources == (source,)
    assert advance(restored.model, restored.optimizer, restored.ema, 2) == expected_loss
    actual = (state_sha256(restored.model.state_dict()), tree_fingerprint(restored.optimizer.state_dict()),
              tree_fingerprint(restored.ema.state_dict(restored.model)), tree_fingerprint(checkpoint._rng_state()))
    assert actual == expected
    rng = tree_fingerprint(checkpoint._rng_state())
    inference = load_model(saved["path"], expected_sha256=saved["sha256"], role="ema")
    assert inference.sources == (source,) and not inference.training
    assert inference.provenance["current_stage_updates"] == inference.provenance["training_updates"] == 1
    assert tree_fingerprint(checkpoint._rng_state()) == rng
    with pytest.raises(ValueError, match="configuration changed"):
        load_training_checkpoint(saved["path"], config={**config, "target_source": "vocals"})
    with pytest.raises(ValueError, match="specialist architecture"):
        save_training_checkpoint(tmp_path / "bad.pt", model, optimizer, ema, step=2, next_sample_index=2,
                                  config={**config, "specialist_waveform_basis": 12}, data_identity={})


@pytest.mark.parametrize("source,index", [("bass", 1), ("drums", 0)])
def test_complete_group_replay_uses_named_absence_and_preservation(source, index):
    torch.manual_seed(23)
    model = SpecialistSeparator(source=source, band_width=16, global_width=16, layers=1,
        waveform_basis=16 if source == "drums" else 0).train()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, foreach=False)
    ema = ParameterEMA(model)
    targets = .02 * torch.randn(16, 4, 2, 128 + 44160)
    targets[:3, index] = 0
    row = losses.grouped_update(model, optimizer, ema, targets.sum(1), targets, step=1,
        warmup_samples=128, ordinary_microbatch=4, auxiliary_microbatch=1,
        target_source=source, extra_ordinary_primary_sdr_weight=.2)
    assert row["ema_updates"] == 1
    assert all(v["step"] == 1 for v in optimizer.state.values())
    assert row["groups"]["ordinary"]["active_windows"] == [13]
    assert row["groups"]["ordinary"]["absent_windows"] == [3]
    assert row["groups"]["auxiliary"]["active_windows"] == [1]
    assert row["groups"]["auxiliary"]["absent_windows"] == [1]
    assert row["accumulation_policy"]["auxiliary_target_source"] == source
    assert all(v["replay_outputs_bit_exact"] for v in row["groups"].values())
    assert set(row["parameter_gradient_norms"]) == dict(model.named_parameters()).keys()


@pytest.mark.parametrize("source", ["bass", "drums"])
def test_specialist_configuration_rejects_wrong_sources_precision_and_geometry(source):
    base = TrainingConfig(model_family="specialist", target_source=source, precision="fp32")
    assert base.validate() == base
    for changes in ({"target_source": "vocals"}, {"teacher_coefficient": 1.}, {"band_width": 0},
                    {"precision": "bf16"}, {"compact_layers": 2}, {"specialist_feature_n_fft": 2048}):
        with pytest.raises(ValueError):
            replace(base, **changes).validate()
