"""Single-output reductions and complete compact-model training recovery."""
from dataclasses import replace
import random

import numpy as np
import pytest
import torch

from stemgenrt import checkpoint, losses
from stemgenrt.checkpoint import ParameterEMA, load_model, load_training_checkpoint, save_training_checkpoint, state_sha256
from stemgenrt.compact import CompactSeparator, SOURCE_ORDER
from stemgenrt.model import render_scored_context
from stemgenrt.trainer import TrainingConfig
from test_checkpoint import tree_fingerprint


@pytest.mark.parametrize("auxiliary", [False, True])
def test_single_output_loss_and_gradients_match_selected_four_head_coordinates(auxiliary):
    torch.manual_seed(37)
    targets = .1 * torch.randn(16, 4, 2, 44160)
    mixture = targets.sum(1)
    if auxiliary:
        mixture, targets = losses.source_views(mixture, targets)
    else:
        targets[:4, 2] = 0
        mixture = targets.sum(1)
    raw = (.1 * torch.randn_like(targets)).requires_grad_()
    deployed = (.1 * torch.randn_like(targets)).requires_grad_()
    loss = losses.auxiliary_objective if auxiliary else losses.objective
    joint = loss(raw, deployed, targets, mixture, target_source="vocals").total
    expected = torch.autograd.grad(joint, (raw, deployed))
    single_raw = raw[:, 2:3].detach().clone().requires_grad_()
    single_deployed = deployed[:, 2:3].detach().clone().requires_grad_()
    single = loss(single_raw, single_deployed, targets[:, 2:3], mixture).total
    actual = torch.autograd.grad(single, (single_raw, single_deployed))
    torch.testing.assert_close(single, joint, rtol=2e-6, atol=2e-6)
    for found, reference in zip(actual, expected):
        torch.testing.assert_close(found, reference[:, 2:3], rtol=2e-6, atol=2e-6)


def test_model_target_mapping_preserves_full_input_views_and_rejects_wrong_target():
    candidate = CompactSeparator(hidden_size=16, layers=1)
    targets = torch.randn(16, 4, 2, 256)
    mixtures, truth = losses.source_views(targets.sum(1), targets)
    selected, local_source = losses.model_targets(candidate, truth, "vocals")
    assert selected.shape == (2, 1, 2, 256) and local_source is None
    assert selected[0].count_nonzero() == 0
    assert mixtures[0].count_nonzero() > 0
    torch.testing.assert_close(selected[1, 0], mixtures[1], rtol=0, atol=0)
    with pytest.raises(ValueError, match="selected training target"):
        losses.model_targets(candidate, truth, None)


@pytest.mark.parametrize("sources", [("vocals",), SOURCE_ORDER])
def test_compact_checkpoint_restores_exact_next_update_and_inference_sources(tmp_path, sources):
    random.seed(16)
    np.random.seed(16)
    torch.manual_seed(16)
    model = CompactSeparator(sources=sources, hidden_size=32, layers=2).train()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, foreach=False)
    ema = ParameterEMA(model)
    config = {"model_family": "compact", "compact_hidden_size": 32, "compact_layers": 2,
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
    path = tmp_path / "compact.pt"
    saved = save_training_checkpoint(path, model, optimizer, ema, step=1, next_sample_index=1,
                                     config=config, data_identity={"train": "fixed"})
    expected_loss = advance(model, optimizer, ema, 2)
    expected = (state_sha256(model.state_dict()), tree_fingerprint(optimizer.state_dict()),
                tree_fingerprint(ema.state_dict(model)), tree_fingerprint(checkpoint._rng_state()))
    restored = load_training_checkpoint(path, sha256=saved["sha256"], config=config,
                                        data_identity={"train": "fixed"})
    assert type(restored.model) is CompactSeparator and restored.model.sources == sources
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
        load_training_checkpoint(path, config={**config, "compact_hidden_size": 64})
    with pytest.raises(ValueError, match="compact architecture"):
        save_training_checkpoint(tmp_path / "bad.pt", model, optimizer, ema, step=2, next_sample_index=2,
                                  config={**config, "target_source": "bass"}, data_identity={})


def test_single_output_complete_group_update_replays_gradients_and_commits_once():
    torch.manual_seed(23)
    model = CompactSeparator(hidden_size=16, layers=1).train()
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


def test_compact_training_config_rejects_silent_recipe_changes():
    base = TrainingConfig(model_family="compact", target_source="vocals")
    assert base.validate() == base
    for changes in ({"target_source": "bass"}, {"teacher_coefficient": 1.},
                    {"compact_hidden_size": 0}, {"model_family": "unknown"}):
        with pytest.raises(ValueError):
            replace(base, **changes).validate()
