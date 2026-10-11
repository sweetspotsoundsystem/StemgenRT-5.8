"""Vocal analysis extension: initialization, causality and additional gradients."""
import pytest
import torch

from stemgenrt.banded import BandSeparator, extend_vocal_analysis
from stemgenrt.checkpoint import state_sha256


def test_extension_preserves_existing_weights_outputs_states_flags_and_rng():
    torch.manual_seed(79)
    source = BandSeparator(band_width=16, global_width=32, layers=2).eval().requires_grad_(False)
    with torch.no_grad():
        for head in source.backbone.mask_heads:
            head.weight.normal_(std=.05)
    digest, rng = state_sha256(source.state_dict()), torch.get_rng_state()
    model = extend_vocal_analysis(source)
    assert state_sha256(source.state_dict()) == digest and torch.equal(torch.get_rng_state(), rng)
    assert not model.training and all(not p.requires_grad for p in model.parameters())
    for key, value in source.state_dict().items():
        torch.testing.assert_close(model.state_dict()[key], value, rtol=0, atol=0)
    assert all(torch.count_nonzero(module.weight) == 0 for module in model.backbone.long_encoders)
    old_state = new_state = None
    with torch.inference_mode():
        for count in (128, 128 * 37, 128 * 4, 128):
            audio = torch.randn(2, 2, count) * .1
            old = source.render(audio, old_state)
            new = model.render(audio, new_state)
            torch.testing.assert_close(new.deployed, old.deployed, rtol=0, atol=0)
            torch.testing.assert_close(new.state.audio_history[..., -896:], old.state.audio_history, rtol=0, atol=0)
            for got, want in zip(new.state[1:], old.state[1:], strict=True):
                torch.testing.assert_close(got, want, rtol=0, atol=0)
            old_state, new_state = old.state, new.state
    assert model.compute_budget()['dense_macs_per_hop'] - source.compute_budget()['dense_macs_per_hop'] == 2 * 2049 * 16
    assert model.compute_budget()['parameters'] - source.compute_budget()['parameters'] == 2 * 2049 * 16
    with pytest.raises(ValueError, match='short-analysis'):
        extend_vocal_analysis(model)


def test_trained_long_features_preserve_causality_chunking_and_receive_gradients():
    torch.manual_seed(73)
    model = BandSeparator(band_width=16, global_width=32, layers=1, feature_n_fft=4096).eval()
    with torch.no_grad():
        for head in model.backbone.mask_heads:
            head.weight.normal_(std=.02)
        for module in model.backbone.long_encoders:
            module.weight.normal_(std=.02)
    audio = torch.randn(1, 2, 128 * 41) * .1
    alternate = audio.clone()
    alternate[..., 128 * 23:] = torch.randn_like(alternate[..., 128 * 23:])
    with torch.inference_mode():
        full = model.render(audio)
        changed = model.render(alternate)
        torch.testing.assert_close(full.deployed[..., :128 * 23], changed.deployed[..., :128 * 23], rtol=0, atol=0)
        state, outputs = None, []
        for chunk in audio.split(128, -1):
            step = model.render(chunk, state)
            outputs.append(step.deployed)
            state = step.state
        torch.testing.assert_close(torch.cat(outputs, -1), full.deployed, rtol=1e-5, atol=2e-7)
        for got, want in zip(state, full.state, strict=True):
            torch.testing.assert_close(got, want, rtol=1e-5, atol=2e-6)
    model.train()
    out = model.render(audio)
    out.deployed.square().mean().backward()
    assert all(module.weight.grad is not None and torch.isfinite(module.weight.grad).all()
               and torch.count_nonzero(module.weight.grad) for module in model.backbone.long_encoders)
