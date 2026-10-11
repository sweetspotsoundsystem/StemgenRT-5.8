"""Physical alignment, causality and genuine single-output architecture checks."""
import pytest
import torch

from stemgenrt.compact import CompactSeparator, SOURCE_ORDER


def model(sources=("vocals",)):
    return CompactSeparator(sources=sources, hidden_size=32, layers=2)


def identity_mask(module):
    with torch.no_grad():
        module.mask_head.weight.zero_()
        bias = module.mask_head.bias.reshape(len(module.sources), 2, 513, 2)
        bias.zero_()
        bias[..., 0] = 1


@pytest.mark.parametrize("sources", [("vocals",), SOURCE_ORDER])
def test_identity_synthesis_and_literal_hop_alignment(sources):
    net = model(sources).eval()
    identity_mask(net)
    audio = torch.randn(2, 2, 128 * 12)
    with torch.no_grad():
        result = net.render(audio)
        tail, _ = net.flush(result.state)
        aligned = torch.cat((result.deployed, tail), -1)[..., 128:128 + audio.shape[-1]]
    assert aligned.shape == (2, len(sources), 2, audio.shape[-1])
    torch.testing.assert_close(aligned, audio[:, None].expand_as(aligned), rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(result.deployed[..., :128], torch.zeros_like(result.deployed[..., :128]),
                               rtol=0, atol=2e-6)


def test_all_impulse_phases_have_exactly_one_hop_of_graph_alignment():
    net = model().eval()
    identity_mask(net)
    # One independent stream per input phase also exercises the overlap boundary.
    audio = torch.zeros(128, 2, 3 * 128)
    audio[torch.arange(128), :, 128 + torch.arange(128)] = 1
    with torch.no_grad():
        result = net.render(audio).deployed[:, 0]
    expected = torch.zeros_like(audio)
    expected[torch.arange(128), :, 256 + torch.arange(128)] = 1
    torch.testing.assert_close(result, expected, rtol=2e-5, atol=1e-6)


@pytest.mark.parametrize("sources", [("vocals",), SOURCE_ORDER])
def test_grouped_render_matches_literal_hops_and_all_states(sources):
    net = model(sources).eval()
    audio = torch.randn(2, 2, 13 * 128)
    with torch.no_grad():
        grouped = net.render(audio)
        state, outputs = None, []
        for chunk in audio.split(128, dim=-1):
            result = net.render(chunk, state)
            outputs.append(result.deployed)
            state = result.state
    torch.testing.assert_close(grouped.deployed, torch.cat(outputs, -1), rtol=2e-5, atol=2e-6)
    for batch_state, hop_state in zip(grouped.state, state):
        torch.testing.assert_close(batch_state, hop_state, rtol=2e-5, atol=2e-6)


def test_future_hops_cannot_change_past_outputs_or_persistent_state():
    net = model().eval()
    audio = torch.randn(2, 2, 12 * 128)
    altered = audio.clone()
    altered[..., 7 * 128:] = torch.randn_like(altered[..., 7 * 128:]) * 10
    with torch.no_grad():
        first = net.render(audio).deployed
        second = net.render(altered).deployed
        prefix = net.render(audio[..., :7 * 128])
        continuation = net.render(audio[..., 7 * 128:], prefix.state)
        full = net.render(audio)
    torch.testing.assert_close(first[..., :7 * 128], second[..., :7 * 128], rtol=0, atol=0)
    for resumed, whole in zip(continuation.state, full.state):
        torch.testing.assert_close(resumed, whole, rtol=2e-5, atol=2e-6)


def test_matched_initialization_and_true_single_output_head():
    torch.manual_seed(42)
    vocal = model()
    torch.manual_seed(42)
    joint = model(SOURCE_ORDER)
    assert vocal.mask_head.out_features == 2052
    assert joint.mask_head.out_features == 4 * 2052
    assert vocal.initial_state(1).spectral_numerator_tail.shape == (1, 1, 2, 128)
    for name, value in vocal.state_dict().items():
        other = joint.state_dict()[name]
        if name.startswith("mask_head"):
            other = other.reshape(4, *value.shape)[2]
        torch.testing.assert_close(value, other, rtol=0, atol=0)
    audio = torch.randn(1, 2, 1024)
    with torch.no_grad():
        torch.testing.assert_close(vocal(audio)[0][:, 0], joint(audio)[0][:, 2], rtol=2e-5, atol=2e-6)


def test_detached_warmup_and_trainable_single_output():
    net = model().train()
    audio = torch.randn(2, 2, 128 * 6 + 31, requires_grad=True)
    out = net.render_scored_context(audio, warmup_samples=128 * 2, carry_state=True)
    assert out.raw.shape == (2, 1, 2, 128 * 4 + 31)
    assert out.flush_hops == 1 and out.initial_state_detached
    out.deployed.square().mean().backward()
    assert audio.grad[..., :256].count_nonzero() == 0
    assert audio.grad[..., 256:].count_nonzero() > 0
    for name, parameter in net.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
        assert parameter.grad.count_nonzero() > 0, name


def test_silence_and_finite_gradients():
    net = model()
    audio = torch.zeros(1, 2, 256, requires_grad=True)
    result = net.render(audio)
    assert result.deployed.count_nonzero() == 0
    result.deployed.sum().backward()
    assert torch.isfinite(audio.grad).all()
    assert all(torch.isfinite(v).all() for v in result.state)


def test_state_source_count_mismatch_is_rejected():
    vocal, joint = model(), model(SOURCE_ORDER)
    with pytest.raises(ValueError, match="state shape"):
        vocal.render(torch.randn(1, 2, 128), joint.initial_state(1))


def test_dense_mac_count_matches_weight_matrices():
    net = model()
    expected = sum(p.numel() for name, p in net.named_parameters()
                   if name.endswith("weight") and p.ndim == 2 or ".weight_" in name)
    assert net.compute_budget()["dense_macs_per_hop"] == expected
