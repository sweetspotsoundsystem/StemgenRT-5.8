"""Causality, physical alignment and budget of the experimental band separator."""
import pytest
import torch
from torch import nn

from stemgenrt.banded import BandSeparator, SOURCE_ORDER
from stemgenrt._model.bands import GroupedAffine
from stemgenrt.model import render_scored_context


@pytest.mark.parametrize("sources", [("vocals",), SOURCE_ORDER])
def test_grouped_literal_future_causality_and_state(sources):
    torch.manual_seed(731)
    model = BandSeparator(sources=sources, band_width=16, global_width=32, layers=2).eval()
    audio = torch.randn(2, 2, 128 * 9) * .1
    with torch.no_grad():
        grouped = model.render(audio)
        state, chunks = None, []
        for chunk in audio.split(128, -1):
            output, state = model.forward_chunk(chunk, state)
            chunks.append(output)
        torch.testing.assert_close(torch.cat(chunks, -1), grouped.deployed, rtol=1e-5, atol=1e-7)
        for literal, batch in zip(state, grouped.state):
            torch.testing.assert_close(literal, batch, rtol=1e-5, atol=1e-6)
        changed = audio.clone()
        changed[..., 512:] = torch.randn_like(changed[..., 512:]) * 30
        torch.testing.assert_close(model.render(changed).deployed[..., :512],
                                   grouped.deployed[..., :512], rtol=0, atol=0)
    assert grouped.deployed.shape == (2, len(sources), 2, audio.shape[-1])
    assert len(state) == 4 and all(v.dtype == torch.float32 for v in state)
    with pytest.raises(ValueError, match="state type"):
        model.render(audio, tuple(state))


def test_identity_mask_reconstructs_every_hop_phase_and_flush():
    model = BandSeparator(band_width=16, global_width=16, layers=1).eval()
    with torch.no_grad():
        for head, (_, _, bins) in zip(model.backbone.mask_heads, model.backbone.groups):
            head.weight.zero_()
            head.bias.zero_()
            head.bias.reshape(-1, 1, 2, bins, 2)[..., 0] = 1
        # One impulse at each of the 128 possible positions within a callback.
        audio = torch.zeros(128, 2, 512)
        audio[torch.arange(128), :, 128 + torch.arange(128)] = 1
        out = model.render(audio)
        flushed, _ = model.flush(out.state)
        aligned = torch.cat((out.deployed, flushed), -1)[..., 128:128 + 512]
        torch.testing.assert_close(aligned[:, 0], audio, atol=3e-7, rtol=1e-6)


def test_matched_seed_backbone_vocal_slice_and_rng_are_identical():
    torch.manual_seed(812)
    vocal = BandSeparator()
    rng = torch.get_rng_state()
    torch.manual_seed(812)
    joint = BandSeparator(sources=SOURCE_ORDER)
    assert torch.equal(torch.get_rng_state(), rng)
    for key, value in vocal.state_dict().items():
        if not key.startswith("backbone.mask_heads"):
            torch.testing.assert_close(joint.state_dict()[key], value, rtol=0, atol=0)
    for v, j in zip(vocal.backbone.mask_heads, joint.backbone.mask_heads):
        torch.testing.assert_close(j.weight.reshape(j.weight.shape[0], 4, -1, 96)[:, 2], v.weight, rtol=0, atol=0)
        torch.testing.assert_close(j.bias.reshape(j.bias.shape[0], 4, -1)[:, 2], v.bias, rtol=0, atol=0)
    with torch.no_grad():
        audio = torch.randn(1, 2, 512) * .1
        torch.testing.assert_close(joint.render(audio).deployed[:, 2:3], vocal.render(audio).deployed,
                                   rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize("silent", [False, True])
def test_scored_context_detaches_warmup_and_all_parameters_have_finite_gradients(silent):
    model = BandSeparator(band_width=16, global_width=16, layers=1).train()
    audio = ((torch.zeros if silent else torch.randn)(1, 2, 641) * .1).requires_grad_()
    result = render_scored_context(model, audio, warmup_samples=128, carry_state=True)
    assert torch.equal(result.physical_mixture, audio[..., 128:])
    result.deployed.square().mean().backward()
    assert audio.grad[..., :128].count_nonzero() == 0
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    if silent:
        assert result.deployed.count_nonzero() == 0
    else:
        assert audio.grad[..., 128:].count_nonzero() > 0


@pytest.mark.parametrize("sources,parameters,macs", [(("vocals",), 2720484, 4805568),
                                                       (SOURCE_ORDER, 3317616, 5396544)])
def test_budget_counts_executed_matrix_multiplications(sources, parameters, macs):
    model = BandSeparator(sources=sources).eval()
    counted = []
    def count(module, args, result):
        x = args[0]
        if isinstance(module, nn.GRU):
            counted.append(x.shape[0] * x.shape[1] * 3 * module.hidden_size * (module.input_size + module.hidden_size))
        else:
            counted.append(x.numel() * module.out_features)
    hooks = [m.register_forward_hook(count) for m in model.modules()
             if isinstance(m, (nn.GRU, nn.Linear, GroupedAffine))]
    with torch.no_grad():
        model.forward_chunk(torch.randn(1, 2, 128))
    for hook in hooks:
        hook.remove()
    budget = model.compute_budget()
    assert sum(counted) == budget["dense_macs_per_hop"] == macs
    assert budget["parameters"] == parameters
    assert budget["persistent_state_elements"] == 6016 + 256 * len(sources)
