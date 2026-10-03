"""Single-output reductions and complete banded-model training recovery."""
from dataclasses import replace
import random

import numpy as np
import pytest
import torch

from stemgenrt import checkpoint, losses
from stemgenrt.checkpoint import ParameterEMA, load_model, load_training_checkpoint, save_training_checkpoint, state_sha256
from stemgenrt.banded import BandSeparator, SOURCE_ORDER
from stemgenrt.model import render_scored_context
from stemgenrt.trainer import TrainingConfig
from test_checkpoint import tree_fingerprint


@pytest.mark.parametrize("sources", [("vocals",), SOURCE_ORDER])
def test_banded_checkpoint_restores_exact_next_update_and_inference_sources(tmp_path, sources):
    random.seed(16)
    np.random.seed(16)
    torch.manual_seed(16)
    model = BandSeparator(sources=sources, band_width=16, global_width=32, layers=2).train()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, foreach=False)
    ema = ParameterEMA(model)
    config = {"model_family": "banded", "band_width": 16, "band_global_width": 32, "band_layers": 2,
              "target_source": "vocals" if len(sources) == 1 else None,
              "steps": 3, "batch_size": 1, "data_start": 0, "precision": "fp32"}

    def advance(model, optimizer, ema, step):
        audio = torch.randn(1, 2, 512) * (.02 + .001 * random.random() + .001 * np.random.random())
        truth = torch.randn(1, len(sources), 2, 384) * .01
        optimizer.zero_grad(set_to_none=True)
        out = render_scored_context(model, audio, warmup_samples=128, carry_state=True)
        objective = (out.deployed - truth).square().mean()
        objective.backward()
        optimizer.step()
        ema.update(model, step=step)
        return float(objective.detach())

    advance(model, optimizer, ema, 1)
    path = tmp_path / "banded.pt"
    saved = save_training_checkpoint(path, model, optimizer, ema, step=1, next_sample_index=1,
                                     config=config, data_identity={"train": "fixed"})
    expected_loss = advance(model, optimizer, ema, 2)
    expected = (state_sha256(model.state_dict()), tree_fingerprint(optimizer.state_dict()),
                tree_fingerprint(ema.state_dict(model)), tree_fingerprint(checkpoint._rng_state()))
    restored = load_training_checkpoint(path, sha256=saved["sha256"], config=config,
                                        data_identity={"train": "fixed"})
    assert type(restored.model) is BandSeparator and restored.model.sources == sources
    assert advance(restored.model, restored.optimizer, restored.ema, 2) == expected_loss
    actual = (state_sha256(restored.model.state_dict()), tree_fingerprint(restored.optimizer.state_dict()),
              tree_fingerprint(restored.ema.state_dict(restored.model)), tree_fingerprint(checkpoint._rng_state()))
    assert actual == expected
    before = tree_fingerprint(checkpoint._rng_state())
    inference = load_model(path, expected_sha256=saved["sha256"], role="ema")
    assert inference.sources == sources and not inference.training
    assert inference.provenance["current_stage_updates"] == inference.provenance["training_updates"] == 1
    assert tree_fingerprint(checkpoint._rng_state()) == before
    with pytest.raises(ValueError, match="configuration changed"):
        load_training_checkpoint(path, config={**config, "band_width": 64})
    with pytest.raises(ValueError, match="banded architecture"):
        save_training_checkpoint(tmp_path / "bad.pt", model, optimizer, ema, step=2, next_sample_index=2,
                                  config={**config, "target_source": "bass"}, data_identity={})


def test_single_output_complete_group_update_replays_gradients_and_commits_once():
    torch.manual_seed(23)
    model = BandSeparator(band_width=16, global_width=16, layers=1).train()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, foreach=False)
    ema = ParameterEMA(model)
    targets = .02 * torch.randn(16, 4, 2, 128 + 44160)
    targets[:3, 2] = 0
    row = losses.grouped_update(model, optimizer, ema, targets.sum(1), targets,
        step=1, warmup_samples=128, ordinary_microbatch=4, auxiliary_microbatch=1,
        target_source="vocals", extra_ordinary_primary_sdr_weight=.2)
    assert row["ema_updates"] == 1
    assert all(v["step"] == 1 for v in optimizer.state.values())
    assert len(row["groups"]["ordinary"]["active_windows"]) == 1
    assert row["groups"]["auxiliary"]["active_windows"] == [1]
    assert row["groups"]["auxiliary"]["absent_windows"] == [1]
    assert all(v["replay_outputs_bit_exact"] for v in row["groups"].values())


def test_banded_training_config_rejects_silent_recipe_changes():
    base = TrainingConfig(model_family="banded", target_source="vocals", precision="fp32")
    assert base.validate() == base
    for changes in ({"target_source": "bass"}, {"teacher_coefficient": 1.},
                    {"band_width": 0}, {"model_family": "unknown"}, {"precision": "bf16"}, {"compact_layers": 2}):
        with pytest.raises(ValueError):
            replace(base, **changes).validate()
