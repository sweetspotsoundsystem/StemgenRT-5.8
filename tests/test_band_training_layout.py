"""Grouped training projections match independent maps without framewise weights."""
import copy

import pytest
import torch
from torch import nn

from stemgenrt._model.bands import GroupedAffine
from stemgenrt.banded import BandSeparator, SOURCE_ORDER


@pytest.mark.parametrize("batch,frames,groups,inputs,outputs", [(2, 7, 3, 24, 16), (1, 1, 1, 16, 388), (3, 5, 20, 16, 16)])
def test_training_projection_and_all_vjps_match_independent_linears(batch, frames, groups, inputs, outputs):
    torch.manual_seed(520)
    linears = nn.ModuleList(nn.Linear(inputs, outputs, dtype=torch.float64) for _ in range(groups))
    grouped = GroupedAffine(linears).train()
    # Transpose storage to exercise the noncontiguous slices produced by the core.
    values = torch.randn(batch, groups, frames, inputs, dtype=torch.float64).transpose(1, 2).requires_grad_()
    reference_input = values.detach().clone().requires_grad_()
    expected = torch.stack([module(reference_input[:, :, index]) for index, module in enumerate(linears)], dim=2)
    actual = grouped(values)
    cotangent = torch.randn_like(actual)
    expected.backward(cotangent)
    actual.backward(cotangent)
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(values.grad, reference_input.grad, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(grouped.weight.grad, torch.stack([m.weight.grad for m in linears]), rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(grouped.bias.grad, torch.stack([m.bias.grad for m in linears]), rtol=1e-12, atol=1e-12)


def test_training_autograd_saves_no_batch_frame_copy_of_weights():
    model = GroupedAffine([nn.Linear(16, 32) for _ in range(3)]).train()
    values = torch.randn(4, 17, 3, 16, requires_grad=True)
    saved = []
    def pack(tensor):
        saved.append(tuple(tensor.shape))
        return tensor
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
        model(values).square().mean().backward()
    # One [band,input,output] matrix suffices, independent of batch/frame count.
    assert (3, 16, 32) in saved
    assert not any(shape[-2:] == (16, 32) and shape != (3, 16, 32) for shape in saved)


@pytest.mark.parametrize("sources", [("vocals",), SOURCE_ORDER])
def test_training_layout_preserves_streaming_state_and_evaluation_audio(sources):
    torch.manual_seed(99)
    training = BandSeparator(sources=sources, band_width=16, global_width=32, layers=2).train()
    evaluation = copy.deepcopy(training).eval()
    with torch.no_grad():
        for model in (training, evaluation):
            # Shared nontrivial mask weights expose more than the initial .25 mix.
            for head in model.backbone.mask_heads:
                head.weight.mul_(10.)
        audio = torch.randn(2, 2, 128 * 13) * .1
        actual, expected = training.render(audio), evaluation.render(audio)
        torch.testing.assert_close(actual.deployed, expected.deployed, rtol=1e-5, atol=1e-7)
        for left, right in zip(actual.state, expected.state):
            torch.testing.assert_close(left, right, rtol=1e-5, atol=1e-6)
        changed = audio.clone()
        changed[..., 640:] *= -10.
        torch.testing.assert_close(training.render(changed).deployed[..., :640], actual.deployed[..., :640], rtol=0, atol=0)
    assert training.compute_budget() == evaluation.compute_budget()
