"""Independent scalar/derivative checks for the supported training objective."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from stemgenrt import losses


@pytest.fixture(autouse=True, scope="module")
def one_cpu_thread():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def source_fixture(*, silent=False):
    generator = torch.Generator().manual_seed(202611016)
    targets = .02 * torch.randn(2, 4, 2, 88320, generator=generator)
    targets[0, 2] = 0
    targets[1, [0, 1, 3]] = 0
    targets[0, 1, :, :44100] = 0
    targets[1, 2, :, 44100:88200] *= .01
    if silent:
        targets.zero_()
    raw = (targets + .003 * torch.randn(targets.shape, generator=generator)).requires_grad_()
    deployed = (targets + .004 * torch.randn(targets.shape, generator=generator)).requires_grad_()
    return raw, deployed, targets, targets.sum(1)


def independent_auxiliary_scalar(raw, deployed, targets, mixture, weights):
    """Isolate each view using the unweighted whole-group objective.

    Fix the other view's estimates at their references and subtract its known
    constant. The constant includes both the -60 dB active-window floor and
    the leakage penalty for nonzero references below the activity threshold.
    This retains joint denominators without calling contribution().
    """
    reference = targets[..., :88200].reshape(2, 4, 2, 2, 44100)
    active = reference.square().sum((2, 4)) / 88200 > 1e-5
    counts = active.sum((0, 2))
    per_view = active.sum(2)
    constants = [-60 * .2 * (row.float() / counts.clamp_min(1)).sum()
                 / (counts > 0).sum().clamp_min(1) for row in per_view]
    absent_counts = (~active).sum((0, 2))
    physical = mixture[..., :88200].reshape(2, 2, 2, 44100)
    mixture_power = physical.square().mean((1, 3))
    leakage = reference.square().mean((2, 4))
    absence_values = 10 * torch.log10(1 + leakage / mixture_power[:, None].clamp_min(1e-5))
    for view in (0, 1):
        per_stem = torch.where(~active[view], absence_values[view], 0).sum(1) / absent_counts.clamp_min(1)
        constants[view] += .2 * .5 * per_stem.sum() / (absent_counts > 0).sum().clamp_min(1)
    values = []
    for view in (0, 1):
        estimates = [torch.cat([audio[i:i + 1] if i == view else targets[i:i + 1]
                                for i in (0, 1)]) for audio in (raw, deployed)]
        values.append(losses.objective(*estimates, targets, mixture).total - constants[1 - view])
    return sum(weight * value for weight, value in zip(weights, values, strict=True))


@pytest.mark.parametrize("silent", [False, True], ids=["mixed_activity", "all_silent"])
@pytest.mark.parametrize("weights", [(1., 1.), (1., .25), (1., 0.)])
def test_auxiliary_scalar_and_gradients_preserve_joint_denominators(silent, weights):
    raw, deployed, targets, mixture = source_fixture(silent=silent)
    ordinary = .1 * losses.objective(raw, deployed, targets, mixture).total
    ordinary_gradients = torch.autograd.grad(ordinary, (raw, deployed))
    actual = .1 * losses.auxiliary_objective(raw, deployed, targets, mixture, weights=weights).total
    expected = .1 * independent_auxiliary_scalar(raw, deployed, targets, mixture, weights)
    assert abs(float((actual - expected).detach())) < 3e-6
    actual_gradients = torch.autograd.grad(actual, (raw, deployed))
    expected_gradients = torch.autograd.grad(expected, (raw, deployed))
    multiplier = raw.new_tensor(weights)[:, None, None, None]
    for actual_gradient, expected_gradient, ordinary_gradient in zip(
            actual_gradients, expected_gradients, ordinary_gradients, strict=True):
        assert torch.isfinite(actual_gradient).all()
        torch.testing.assert_close(actual_gradient, expected_gradient, atol=1e-7, rtol=1e-4)
        torch.testing.assert_close(actual_gradient, ordinary_gradient * multiplier, atol=1e-7, rtol=1e-4)
        relative = (actual_gradient - expected_gradient).double().norm() / expected_gradient.double().norm()
        assert relative < 5e-5
    if weights == (1., 1.):
        assert abs(float((actual - ordinary).detach())) < 3e-6


def test_direct_sdr_matches_independent_numpy_window_metric():
    raw, deployed, targets, mixture = source_fixture()
    term = losses.direct_sdr(raw, deployed, targets, mixture)
    truth = targets.numpy()[..., :88200].reshape(2, 4, 2, 2, 44100)
    estimates = deployed.detach().numpy()[..., :88200].reshape(2, 4, 2, 2, 44100)
    signal = (truth.astype(np.float64) ** 2).sum(axis=(2, 4))
    error = ((estimates.astype(np.float64) - truth) ** 2).sum(axis=(2, 4))
    active = signal / 88200 > 1e-5
    values = np.clip(10 * np.log10((error + 1e-12) / (signal + 1e-12)), -60, 60)
    expected = np.where(active, values, 0).sum(axis=(0, 2)) / np.maximum(active.sum(axis=(0, 2)), 1)
    np.testing.assert_allclose(term.per_stem_negative_sdr_db.detach().numpy(), expected, atol=2e-5, rtol=0)
    gradient, = torch.autograd.grad(term.negative_sdr_db, deployed, retain_graph=True)
    assert torch.count_nonzero(gradient[..., 88200:]) == 0
    assert torch.count_nonzero(gradient[0, 2]) == 0
    full_gradient, = torch.autograd.grad(term.total, deployed)
    assert torch.count_nonzero(full_gradient[0, 2]) > 0
    assert float((full_gradient * (deployed.detach() - targets)).sum()) > 0


def test_complete_reduction_matches_whole_batch_scalar_and_gradients():
    raw, deployed, targets, mixture = source_fixture()
    reduction = losses.prepare_reduction(targets)
    split = sum(losses.contribution(raw[i:i + 1], deployed[i:i + 1], targets[i:i + 1],
                                   mixture[i:i + 1], reduction).total for i in (0, 1))
    whole = losses.objective(raw, deployed, targets, mixture).total
    torch.testing.assert_close(split, whole, atol=3e-6, rtol=0)
    for actual, expected in zip(torch.autograd.grad(split, (raw, deployed)),
                                torch.autograd.grad(whole, (raw, deployed)), strict=True):
        torch.testing.assert_close(actual, expected, atol=1e-7, rtol=1e-4)


def test_source_views_remove_sources_through_warmup_and_preserve_ordinary_batch():
    generator = torch.Generator().manual_seed(42)
    targets = torch.randn(16, 4, 2, 768, generator=generator)
    mixture = targets.sum(1) + .001
    original = targets.clone(), mixture.clone()
    auxiliary_mix, auxiliary_targets = losses.source_views(mixture, targets)
    assert torch.equal(targets, original[0]) and torch.equal(mixture, original[1])
    assert torch.equal(auxiliary_mix, auxiliary_targets.sum(1))
    assert torch.equal(auxiliary_targets[0, [0, 1, 3]], targets[14, [0, 1, 3]])
    assert torch.equal(auxiliary_targets[1, 2], targets[15, 2])
    assert torch.count_nonzero(auxiliary_targets[0, 2]) == 0
    assert torch.count_nonzero(auxiliary_targets[1, [0, 1, 3]]) == 0


def test_silence_and_auxiliary_input_validation():
    raw, deployed, targets, mixture = source_fixture(silent=True)
    assert losses.objective(targets, targets, targets, mixture).total.item() == 0
    assert losses.auxiliary_objective(targets, targets, targets, mixture).total.item() == 0
    for weights in ((1., -.25), (1., float("nan")), (.25, 1.), (1.,), [1., .25]):
        with pytest.raises(ValueError, match="ordered source-view pair"):
            losses.auxiliary_objective(raw, deployed, targets, mixture, weights=weights)
    raw, deployed, targets, mixture = source_fixture()
    with pytest.raises(ValueError, match="ordering"):
        losses.auxiliary_objective(raw.flip(0), deployed.flip(0), targets.flip(0), mixture.flip(0))


@pytest.mark.parametrize("extra", [0., .2], ids=["baseline", "primary_sdr_ablation"])
@pytest.mark.parametrize("teacher_coefficient", [0., 1.])
@pytest.mark.parametrize("target_source", [None, "vocals"])
def test_output_vjp_replay_matches_full_graph_parameter_gradients(monkeypatch, extra, teacher_coefficient, target_source):
    """Compare replay to an independently retained tiny neural graph.

    This isolates the accumulation protocol without a costly full-size model;
    model streaming/context parity has separate tests.
    """
    import stemgenrt.model

    # Isolate this fixture from test order. Accumulating a million FP32 linear
    # products in one batch versus B4/B1 partitions can itself exceed the VJP
    # tolerance through reassociation. Use FP64 only for the tiny neural map
    # and its parameter-gradient reduction; the actual loss coordinates and
    # every loss operation still use the production FP32 arithmetic.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(43)
        model = torch.nn.Linear(2, 8, dtype=torch.float64)
    model.architecture_metadata = {"source_order": list(losses.SOURCE_NAMES)}
    with torch.no_grad():
        model.weight.mul_(.1)
        model.bias.mul_(.1)
    last_coordinates = None

    def render(model, audio, *, warmup_samples, carry_state):
        nonlocal last_coordinates
        assert carry_state
        # This fixture intentionally has no path from warmup to the gradient.
        scored = audio[..., warmup_samples:]
        raw = model(scored.transpose(1, 2).double()).float().transpose(1, 2).reshape(len(audio), 4, 2, -1)
        deployed = raw + .03 * scored[:, None]
        last_coordinates = raw.detach(), deployed.detach()
        return SimpleNamespace(raw=raw, deployed=deployed, initial_state_detached=True, flush_hops=1)

    monkeypatch.setattr(stemgenrt.model, "render_scored_context", render)
    generator = torch.Generator().manual_seed(44)
    targets = .02 * torch.randn(16, 4, 2, 128 + 44160, generator=generator)
    targets[:3, 2] = 0
    mixture = targets.sum(1)
    teacher_targets = .03 * torch.randn(targets[..., 128:].shape, generator=generator)
    auxiliary_mix, auxiliary_targets = losses.source_views(mixture, targets)
    inputs = (("ordinary", mixture, targets, losses.objective, 1.),
              ("auxiliary", auxiliary_mix, auxiliary_targets, losses.auxiliary_objective, .1))
    reference_coordinates = {}
    reference_loss = 0.
    for group, audio, truth, objective, coefficient in inputs:
        result = render(model, audio, warmup_samples=128, carry_state=True)
        reference_coordinates[group] = last_coordinates
        reference_loss = reference_loss + coefficient * objective(
            result.raw, result.deployed, truth[..., 128:], audio[..., 128:], target_source=target_source,
            **({"extra_ordinary_primary_sdr_weight": extra} if group == "ordinary" else {})).total
        if teacher_coefficient and group == "ordinary":
            from stemgenrt._losses.teacher import contribution
            reference_loss = reference_loss + teacher_coefficient * contribution(
                result.deployed, teacher_targets, truth[..., 128:], audio[..., 128:],
                losses.prepare_reduction(truth[..., 128:]), target_source=target_source).total
    expected = torch.autograd.grad(reference_loss, tuple(model.parameters()))

    def compare_capture_coordinates(phase, group, offset):
        if phase == "canonical_capture":
            for actual, reference in zip(last_coordinates, reference_coordinates[group], strict=True):
                assert torch.equal(actual, reference[offset:offset + len(actual)])

    rows = losses.accumulate_groups(model, mixture, targets, warmup_samples=128,
                                   ordinary_microbatch=4, auxiliary_microbatch=1,
                                   verify_input_gradients=True, progress=compare_capture_coordinates,
                                   extra_ordinary_primary_sdr_weight=extra,
                                   teacher_coefficient=teacher_coefficient, teacher_targets=teacher_targets,
                                   target_source=target_source)
    for parameter, reference in zip(model.parameters(), expected, strict=True):
        torch.testing.assert_close(parameter.grad, reference, atol=1e-6, rtol=1e-5)
    assert abs(sum(row["weighted_loss"] for row in rows.values()) - float(reference_loss.detach())) < 3e-6
    assert all(row["replay_outputs_bit_exact"] for row in rows.values())
    assert all(row["whole_group_objective_evaluations"] == 1 for row in rows.values())
    assert rows["auxiliary"]["view_contribution_multipliers"] == [1., .25]
    assert 'teacher_coefficient' not in rows['auxiliary']
    assert ('teacher_coefficient' in rows['ordinary']) == bool(teacher_coefficient)


@pytest.mark.parametrize("silent", [False, True])
def test_primary_sdr_increment_preserves_absence_anchor_and_partial_tail(silent):
    raw, deployed, targets, mixture = source_fixture(silent=silent)
    baseline = losses.objective(raw, deployed, targets, mixture)
    candidate = losses.objective(raw, deployed, targets, mixture, extra_ordinary_primary_sdr_weight=.2)
    # Independently construct the active full-window metric, without using
    # direct_sdr() or the candidate's exposed component value.
    truth = targets[..., :88200].reshape(2, 4, 2, 2, 44100)
    estimate = deployed[..., :88200].reshape(2, 4, 2, 2, 44100)
    signal = truth.square().sum((2, 4))
    active = signal / 88200 > 1e-5
    count = active.sum((0, 2))
    db = (10 * torch.log10(((estimate - truth).square().sum((2, 4)) + 1e-12)
                          / (signal + 1e-12))).clamp(-60, 60)
    primary = (torch.where(active, db, 0).sum((0, 2)) / count.clamp_min(1)).sum()
    primary = primary / (count > 0).sum().clamp_min(1)
    expected = baseline.total + .2 * primary
    torch.testing.assert_close(candidate.total, expected, atol=1e-6, rtol=0)
    for actual, reference in zip(torch.autograd.grad(candidate.total, (raw, deployed)),
                                torch.autograd.grad(expected, (raw, deployed), retain_graph=True), strict=True):
        torch.testing.assert_close(actual, reference, atol=1e-7, rtol=1e-4)
    increment_gradient, = torch.autograd.grad(primary, deployed)
    assert torch.count_nonzero(increment_gradient[..., 88200:]) == 0
    assert torch.count_nonzero(increment_gradient[0, 2]) == 0
    for field in ("direct_sdr_loss", "absence_db", "direct_raw_anchor", "reconstruction_loss"):
        assert torch.equal(getattr(candidate, field), getattr(baseline, field))
    assert losses.policy()["direct_sdr_weight"] == .2
    assert losses.policy(extra_ordinary_primary_sdr_weight=.2)["ordinary_primary_sdr_weight"] == .4


@pytest.mark.parametrize("stop_group", ["ordinary", "auxiliary"])
def test_grouped_update_has_one_commit_boundary_and_can_retry_interruption(monkeypatch, stop_group):
    import stemgenrt.checkpoint as checkpoint

    class TinyModel(torch.nn.Module):
        architecture_metadata = {"test": "40 small FP32 parameters"}
        parameter_tensor_count = 40

        def __init__(self):
            super().__init__()
            self.values = torch.nn.ParameterList([
                torch.nn.Parameter(torch.tensor([.01 * (index + 1)])) for index in range(40)])

    monkeypatch.setattr(checkpoint, "StemgenRT58", TinyModel)
    model = TinyModel().train()
    optimizer = torch.optim.Adam(model.parameters(), lr=6e-5, foreach=False)
    ema = checkpoint.ParameterEMA(model)
    original = checkpoint.state_sha256(model.state_dict())
    original_ema = checkpoint.state_sha256(ema.parameters)
    events = []
    zero_calls = []

    def accumulate(model, *args, after_group=None, check_continue=None, **kwargs):
        rows = {}
        for group in ("ordinary", "auxiliary"):
            if check_continue:
                check_continue()
            for parameter in model.parameters():
                if parameter.grad is None:
                    parameter.grad = torch.ones_like(parameter)
                else:
                    parameter.grad.add_(.1)
            rows[group] = {"weighted_loss": 1.}
            if after_group:
                after_group(group, rows[group])
        return rows

    clip, step, average = torch.nn.utils.clip_grad_norm_, optimizer.step, ema.update
    clear = optimizer.zero_grad

    def observed_clear(*args, **kwargs):
        zero_calls.append(1)
        return clear(*args, **kwargs)

    def observed_clip(*args, **kwargs):
        events.append("clip")
        return clip(*args, **kwargs)

    def observed_step(*args, **kwargs):
        events.append("Adam")
        return step(*args, **kwargs)

    def observed_average(*args, **kwargs):
        events.append("EMA")
        return average(*args, **kwargs)

    monkeypatch.setattr(losses, "accumulate_groups", accumulate)
    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", observed_clip)
    monkeypatch.setattr(optimizer, "step", observed_step)
    monkeypatch.setattr(optimizer, "zero_grad", observed_clear)
    monkeypatch.setattr(ema, "update", observed_average)
    inputs = torch.zeros(16, 2, 256), torch.zeros(16, 4, 2, 256)

    class RequestedStop(Exception):
        pass

    stopped = False

    def check_continue():
        if stopped:
            raise RequestedStop()

    def stop_after_group(group, row):
        nonlocal stopped
        assert checkpoint.state_sha256(model.state_dict()) == original
        assert not optimizer.state and ema.updates == 0
        stopped = group == stop_group

    with pytest.raises(ValueError, match="contiguous EMA endpoint"):
        losses.grouped_update(model, optimizer, ema, *inputs, step=2, warmup_samples=128)
    assert all(parameter.grad is None for parameter in model.parameters())
    assert zero_calls == []
    with pytest.raises(RequestedStop):
        losses.grouped_update(model, optimizer, ema, *inputs, step=1, warmup_samples=128,
                              after_group=stop_after_group, check_continue=check_continue)
    assert events == [] and not optimizer.state and ema.updates == 0
    assert len(zero_calls) == 1
    assert checkpoint.state_sha256(model.state_dict()) == original
    assert checkpoint.state_sha256(ema.parameters) == original_ema
    result = losses.grouped_update(model, optimizer, ema, *inputs, step=1, warmup_samples=128)
    assert events == ["clip", "Adam", "EMA"]
    assert len(zero_calls) == 2
    assert result["gradient_norm_before_clip"] == pytest.approx(np.sqrt(40 * 1.1 ** 2))
    assert result["weighted_loss"] == 2. and ema.updates == 1
    assert result["raw_model_state_sha256"] != original
    assert result["ema_parameters_sha256"] != original_ema
    assert len(optimizer.state) == 40
    assert all(state["step"].item() == 1 for state in optimizer.state.values())
