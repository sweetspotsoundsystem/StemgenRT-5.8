"""Source selection must cover every objective component without rescaling."""
from dataclasses import replace

import pytest
import torch

from stemgenrt import losses, trainer
from stemgenrt._losses.teacher import contribution as teacher_term


@pytest.mark.parametrize("source", losses.SOURCE_NAMES)
@pytest.mark.parametrize("silent", [False, True])
def test_selected_loss_matches_four_identical_sources_and_has_no_other_output_gradients(source, silent):
    torch.set_num_threads(1)
    rng = torch.Generator().manual_seed(172)
    truth = .02 * torch.randn(2, 4, 2, 44160, generator=rng)
    index = losses.SOURCE_NAMES.index(source)
    truth[0, index] = 0
    if silent:
        truth[:, index] = 0
    mixture = truth.sum(1)
    raw = (truth + .003 * torch.randn(truth.shape, generator=rng)).requires_grad_()
    deployed = (truth + .005 * torch.randn(truth.shape, generator=rng)).requires_grad_()
    teacher = truth + .001 * torch.randn(truth.shape, generator=rng)

    def repeated(tensor):
        return tensor[:, index:index + 1].repeat(1, 4, 1, 1)

    # An independent oracle: the unchanged joint reduction on four identical
    # copies has exactly one source's value. Keep the physical mixture fixed.
    actual = losses.objective(raw, deployed, truth, mixture,
        extra_ordinary_primary_sdr_weight=.2, target_source=source).total
    expected = losses.objective(repeated(raw), repeated(deployed), repeated(truth), mixture,
        extra_ordinary_primary_sdr_weight=.2).total
    selected_teacher = teacher_term(deployed, teacher, truth, mixture,
        losses.prepare_reduction(truth), target_source=source).total
    joint_teacher = teacher_term(repeated(deployed), repeated(teacher), repeated(truth), mixture,
        losses.prepare_reduction(repeated(truth))).total
    torch.testing.assert_close(selected_teacher, joint_teacher, atol=2e-6, rtol=1e-6)
    actual = actual + selected_teacher
    expected = expected + joint_teacher
    torch.testing.assert_close(actual, expected, atol=3e-6, rtol=1e-6)
    for actual_grad, expected_grad in zip(torch.autograd.grad(actual, (raw, deployed)),
                                        torch.autograd.grad(expected, (raw, deployed)), strict=True):
        torch.testing.assert_close(actual_grad, expected_grad, atol=1e-7, rtol=1e-5)
        assert torch.count_nonzero(actual_grad[:, [i for i in range(4) if i != index]]) == 0
        # Even with absent vocals, raw anchors and leakage must remain active.
        assert torch.count_nonzero(actual_grad[0, index]) > 0


def test_selected_auxiliary_views_keep_joint_denominators_and_weights():
    rng = torch.Generator().manual_seed(173)
    truth = .02 * torch.randn(2, 4, 2, 44160, generator=rng)
    truth[0, 2] = 0
    truth[1, [0, 1, 3]] = 0
    mixture = truth.sum(1)
    raw = (truth + .003 * torch.randn(truth.shape, generator=rng)).requires_grad_()
    deployed = (truth + .005 * torch.randn(truth.shape, generator=rng)).requires_grad_()
    ordinary = losses.objective(raw, deployed, truth, mixture, target_source="vocals").total
    auxiliary = losses.auxiliary_objective(raw, deployed, truth, mixture, target_source="vocals").total
    ordinary_grads = torch.autograd.grad(ordinary, (raw, deployed))
    auxiliary_grads = torch.autograd.grad(auxiliary, (raw, deployed))
    weights = raw.new_tensor([1., .25])[:, None, None, None]
    for actual, reference in zip(auxiliary_grads, ordinary_grads, strict=True):
        torch.testing.assert_close(actual, reference * weights, atol=1e-7, rtol=1e-5)
        assert torch.count_nonzero(actual[:, [0, 1, 3]]) == 0
        assert torch.count_nonzero(actual[0, 2]) > 0
        assert torch.count_nonzero(actual[1, 2]) > 0


def test_source_selection_is_explicit_in_policy_and_configuration():
    assert "target_source" not in losses.policy()
    assert losses.policy(target_source=None) == losses.policy()
    assert losses.policy(target_source="vocals")["target_source"] == "vocals"
    assert replace(trainer.TrainingConfig(), target_source="vocals").validate().target_source == "vocals"
    for invalid in ("lead", 2, False, ["vocals"]):
        with pytest.raises(ValueError, match="target_source"):
            losses.policy(target_source=invalid)
