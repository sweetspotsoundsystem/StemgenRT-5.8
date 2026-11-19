from dataclasses import replace
import copy
import math

import pytest
import torch

from stemgenrt import losses
from stemgenrt._losses import release_margin as margin
from stemgenrt.checkpoint import ParameterEMA, _validate_training_provenance, state_sha256
from stemgenrt.model import render_scored_context
from stemgenrt.specialist import SpecialistSeparator
from stemgenrt.trainer import TrainingConfig


WINDOW = margin.METRICS.window_samples


def test_quiet_margin_has_one_sided_gradients_and_whole_condition_denominator():
    mix = torch.full((1, 2, 3 * WINDOW + 20), .4)
    truth = torch.zeros_like(mix)
    truth[..., WINDOW:2*WINDOW] = .1
    release = torch.full_like(mix, .01)
    trial = release.clone()
    trial[..., :WINDOW] *= 2
    trial[..., WINDOW:2*WINDOW] *= 4  # Bass is active.
    trial[..., 3*WINDOW:] *= 8  # Incomplete trailing window.
    trial.requires_grad_()
    value, conditions = margin.margin_loss(trial, release, mix, truth, group="ordinary")
    expected = (20 * math.log10(2) - .5) / 2
    row = conditions["ordinary_quiet"]
    assert row["eligible_windows"] == 2 and row["violating_windows"] == 1
    assert row["mean_excess_db"] == pytest.approx(expected)
    assert value.item() == pytest.approx(expected / 3)
    value.backward()
    assert (trial.grad[..., :WINDOW] > 0).all() and torch.count_nonzero(trial.grad[..., WINDOW:]) == 0
    with pytest.raises(ValueError, match="fixed release"):
        margin.margin_loss(trial, release.requires_grad_(), mix, truth, group="ordinary")


def test_removed_and_solo_use_physical_levels_floor_ceiling_and_separate_denominators():
    mix = torch.full((2, 2, 3*WINDOW), .25)
    mix[0, :, 2*WINDOW:] = 0.
    truth = mix.clone()
    truth[0].zero_()
    release = torch.zeros_like(mix)
    release[0, :, :WINDOW] = .005
    release[1] = truth[1] - .01
    release[1, :, 2*WINDOW:] = truth[1, :, 2*WINDOW:]  # Perfect reference and student, bounded SDR.
    trial = release.clone()
    trial[0, :, :WINDOW] *= 2
    trial[0, :, 2*WINDOW:] = .1  # The removed input is silent here.
    trial[1, :, :WINDOW] = truth[1, :, :WINDOW] - .02
    trial.requires_grad_()
    value, rows = margin.margin_loss(trial, release, mix, truth, group="auxiliary")
    assert rows["removed"]["eligible_windows"] == 2 and rows["removed"]["violating_windows"] == 1
    assert rows["solo"]["eligible_windows"] == 3 and rows["solo"]["violating_windows"] == 1
    removed = (20 * math.log10(2) - .5) / 2
    solo = (20 * math.log10(2) - .1) / 3
    assert value.item() == pytest.approx((removed + solo) / 3, abs=3e-5)
    value.backward()
    assert torch.isfinite(trial.grad).all() and torch.count_nonzero(trial.grad[..., WINDOW:]) == 0
    assert (trial.grad[0, :, :WINDOW] > 0).all() and (trial.grad[1, :, :WINDOW] < 0).all()
    # No window exceeds a margin when the candidate matches the release.
    equal = release.clone().requires_grad_()
    value, rows = margin.margin_loss(equal, release, mix, truth, group="auxiliary")
    value.backward()
    assert value.item() == 0 and torch.count_nonzero(equal.grad) == 0
    assert all(row["violating_windows"] == 0 for row in rows.values())


def test_margin_configuration_and_reference_provenance(monkeypatch):
    config = TrainingConfig(model_family="specialist", target_source="bass", precision="fp32",
                            release_margin_weight=.2, bass_release_sha256="a"*64)
    assert config.validate() == config
    for fields in ({"release_margin_weight": float("nan")}, {"release_margin_weight": -1},
                   {"bass_release_sha256": None}, {"bass_release_sha256": "wrong"},
                   {"release_margin_weight": 0}, {"target_source": "drums"}):
        with pytest.raises(ValueError):
            replace(config, **fields).validate()
    ref = SpecialistSeparator(source="bass", band_width=16, global_width=16, layers=1)
    def load(path, *, expected_sha256, role, device):
        assert path == "release" and expected_sha256 == "a"*64 and role == "ema" and device == "cpu"
        torch.rand(2)
        return ref
    monkeypatch.setattr(margin, "load_model", load)
    rng = torch.get_rng_state().clone()
    objective = margin.BassReleaseMargin(.2, "a"*64, checkpoint="release", device="cpu")
    assert torch.equal(rng, torch.get_rng_state()) and all(not p.requires_grad for p in ref.parameters())
    inputs = .03 * torch.randn(2, 2, 128 + 44160)
    with torch.no_grad():
        expected = torch.cat([render_scored_context(ref, x, warmup_samples=128, carry_state=True).deployed[:, 0]
                              for x in inputs.split(1)])
    truth = inputs[..., 128:].clone()
    truth[0].zero_()
    output = expected + .01
    v1, rows1 = margin.margin_loss(output, expected, inputs[..., 128:], truth, group="auxiliary")
    v2, rows2 = objective(output, inputs, truth, group="auxiliary", warmup_samples=128, microbatch=1)
    assert torch.equal(v1, v2) and rows1 == rows2
    assert not any(p.grad is not None for p in ref.parameters())
    compact = {"model_family":"specialist", "target_source":"bass", "release_margin_weight":.2, "bass_release_sha256":"a"*64}
    provenance = {"current_stage_bass_release_margin":objective.specification}
    _validate_training_provenance(compact, provenance)
    for wrong in ({**compact, "release_margin_weight":.25}, {**compact, "bass_release_sha256":"b"*64}, {}):
        with pytest.raises(ValueError, match="margin"):
            _validate_training_provenance(wrong, provenance)


def test_grouped_replay_adds_margin_once_outside_the_source_view_weight():
    torch.manual_seed(884)
    model = SpecialistSeparator(source="bass", band_width=16, global_width=16, layers=1)
    altered = copy.deepcopy(model)
    targets = .035 * torch.randn(16, 4, 2, 128 + 44160)
    targets[:2, 1].zero_()
    mixture = targets.sum(1)
    class KnownRelease:
        weight = .23
        def __call__(self, student, mixture, truth, *, group, warmup_samples, microbatch):
            # A perfect Bass reference makes the active penalty observable for each group.
            return margin.margin_loss(student, truth, mixture[..., warmup_samples:], truth, group=group)
    def update(m, objective):
        return losses.grouped_update(m, torch.optim.Adam(m.parameters(), lr=.001, foreach=False), ParameterEMA(m),
                                     mixture, targets, step=1, warmup_samples=128, ordinary_microbatch=4,
                                     target_source="bass", auxiliary_weight=.1, release_margin=objective)
    base = update(model, None)
    trial = update(altered, KnownRelease())
    for name in ("ordinary", "auxiliary"):
        expected = base["groups"][name]["weighted_loss"]
        h = trial["groups"][name]["release_margin_unweighted"]
        assert h > 0 and trial["groups"][name]["weighted_loss"] == pytest.approx(expected + .23*h, abs=2e-6)
    assert trial["accumulation_policy"]["version"] == margin.VERSION
    assert trial["groups"]["auxiliary"]["view_contribution_multipliers"] == [1., .25]
    assert trial["ema_updates"] == base["ema_updates"] == 1
    assert state_sha256(model.state_dict()) != state_sha256(altered.state_dict())


def test_solo_multiplier_changes_only_solo_gradient_and_keeps_original_denominator():
    mixture = torch.full((2, 2, WINDOW), .25)
    truth = mixture.clone()
    truth[0].zero_()
    release = truth - .01
    trial = (truth - .025).requires_grad_()
    base, rows = margin.margin_loss(trial, release, mixture, truth, group="auxiliary")
    heavier, updated = margin.margin_loss(trial, release, mixture, truth, group="auxiliary", solo_multiplier=4.)
    base_gradient, = torch.autograd.grad(base, trial)
    heavy_gradient, = torch.autograd.grad(heavier, trial)
    assert rows == updated and all(r["violating_windows"] for r in rows.values())
    assert heavier.item() - base.item() == pytest.approx(rows["solo"]["mean_excess_db"], abs=2e-6)
    assert torch.equal(base_gradient[0], heavy_gradient[0])
    assert torch.equal(4 * base_gradient[1], heavy_gradient[1])
    assert torch.count_nonzero(base_gradient[0]) and torch.count_nonzero(base_gradient[1])
    quiet, old_rows = margin.margin_loss(trial, release, mixture, truth, group="ordinary")
    new_quiet, new_rows = margin.margin_loss(trial, release, mixture, truth, group="ordinary", solo_multiplier=4.)
    assert torch.equal(quiet, new_quiet) and old_rows == new_rows
    assert torch.equal(torch.autograd.grad(quiet, trial)[0], torch.autograd.grad(new_quiet, trial)[0])


def test_weighted_solo_recipe_binds_recovery_and_preserves_legacy_metadata():
    config = {"model_family":"specialist", "target_source":"bass", "release_margin_weight":.2, "bass_release_sha256":"a"*64}
    old = margin.specification(.2, "a"*64)
    assert old == margin.specification(.2, "a"*64, solo_multiplier=1.) and "solo_multiplier" not in old
    new = margin.specification(.2, "a"*64, solo_multiplier=4.)
    assert new["version"] == margin.SOLO_VERSION and new["solo_multiplier"] == 4.
    provenance = {"current_stage_bass_release_margin":new}
    _validate_training_provenance({**config, "release_margin_solo_multiplier":4.}, provenance)
    with pytest.raises(ValueError, match="margin"):
        _validate_training_provenance(config, provenance)
    with pytest.raises(ValueError, match="margin"):
        _validate_training_provenance({**config,"release_margin_solo_multiplier":4.}, {"current_stage_bass_release_margin":old})
    policy = losses.policy(release_margin_weight=.2, release_margin_solo_multiplier=4.)
    assert policy["version"] == margin.SOLO_VERSION and policy["release_margin_solo_multiplier"] == 4.
    assert "release_margin_solo_multiplier" not in losses.policy(release_margin_weight=.2)
    template = TrainingConfig(**config, precision="fp32")
    for value in (-1., float("nan"), float("inf"), True):
        with pytest.raises(ValueError, match="multiplier"):
            replace(template, release_margin_solo_multiplier=value).validate()
    with pytest.raises(ValueError, match="multiplier"):
        replace(template, release_margin_solo_multiplier=4., release_margin_weight=0., bass_release_sha256=None).validate()


def test_lower_rate_recipe_scales_every_update_of_the_matched_continuation():
    import json
    from pathlib import Path
    from stemgenrt.trainer import learning_rate

    configs = Path(__file__).resolve().parents[1] / "configs"
    base = TrainingConfig(**json.loads((configs/"bass-release-margin-finetune.json").read_text())).validate()
    lower = TrainingConfig(**json.loads((configs/"bass-margin-low-lr-finetune.json").read_text())).validate()
    assert lower == replace(base, lr=0.0001, min_lr=0.00001)
    assert lower.steps == 8000 and lower.warmup == 100
    for step in range(lower.steps):
        assert math.isclose(learning_rate(step, lower), learning_rate(step, base)/3, rel_tol=1e-14)
    assert learning_rate(lower.warmup-1, lower) == lower.lr
    assert learning_rate(lower.steps-1, lower) == lower.min_lr
