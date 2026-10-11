"""Streaming geometry, future independence and executed budgets of stem specialists."""
import pytest
import torch
from torch import nn

from stemgenrt._model.bands import GroupedAffine
from stemgenrt.model import render_scored_context
from stemgenrt.specialist import SpecialistSeparator


@pytest.mark.parametrize("source,feature_n_fft,waveform_basis", [
    ("bass", 4096, 0), ("bass", 1024, 0), ("drums", 1024, 32), ("drums", 1024, 0)])
def test_specialist_grouping_future_independence_and_state(source, feature_n_fft, waveform_basis):
    torch.manual_seed(184)
    model = SpecialistSeparator(source=source, band_width=16, global_width=32, layers=2,
        feature_n_fft=feature_n_fft, waveform_basis=waveform_basis).eval()
    audio = torch.randn(2, 2, 128 * 40) * .05
    with torch.no_grad():
        grouped = model.render(audio)
        state, chunks = None, []
        for chunk in audio.split(128, -1):
            value, state = model.forward_chunk(chunk, state)
            chunks.append(value)
        torch.testing.assert_close(torch.cat(chunks, -1), grouped.deployed, rtol=2e-5, atol=2e-7)
        for literal, batched in zip(state, grouped.state):
            torch.testing.assert_close(literal, batched, rtol=2e-5, atol=2e-6)
        changed = audio.clone()
        changed[..., 1024:] = torch.randn_like(changed[..., 1024:]) * 30
        torch.testing.assert_close(model.render(changed).deployed[..., :1024],
                                   grouped.deployed[..., :1024], rtol=0, atol=0)
    assert grouped.deployed.shape == (2, 1, 2, audio.shape[-1])
    assert state.audio_history.shape[-1] == feature_n_fft - 128
    assert len(state) == (5 if waveform_basis else 4)
    assert model.architecture_metadata["source_order"] == [source]
    with pytest.raises(ValueError, match="state type"):
        model.render(audio, tuple(state))


@pytest.mark.parametrize("source", ["bass", "drums"])
def test_identity_carrier_all_callback_phases_and_flush(source):
    model = SpecialistSeparator(source=source, band_width=16, global_width=16, layers=1).eval()
    with torch.no_grad():
        for head, (_, _, bins) in zip(model.backbone.mask_heads, model.backbone.groups):
            head.weight.zero_()
            head.bias.zero_()
            head.bias.reshape(-1, 1, 2, bins, 2)[..., 0] = 1
        if model.waveform_basis:
            model.waveform_decode.weight.zero_()
        audio = torch.zeros(128, 2, 512)
        audio[torch.arange(128), :, 128 + torch.arange(128)] = 1
        out = model.render(audio)
        flushed, _ = model.flush(out.state)
        aligned = torch.cat((out.deployed, flushed), -1)[..., 128:128 + 512]
        torch.testing.assert_close(aligned[:, 0], audio, atol=3e-7, rtol=1e-6)


@pytest.mark.parametrize("source", ["bass", "drums"])
@pytest.mark.parametrize("silent", [False, True])
def test_complete_gradient_path_and_detached_warmup(source, silent):
    model = SpecialistSeparator(source=source, band_width=16, global_width=16, layers=1).train()
    audio = ((torch.zeros if silent else torch.randn)(1, 2, 1025) * .05).requires_grad_()
    result = render_scored_context(model, audio, warmup_samples=128, carry_state=True)
    assert torch.equal(result.physical_mixture, audio[..., 128:])
    result.deployed.square().mean().backward()
    assert audio.grad[..., :128].count_nonzero() == 0
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    if silent:
        assert result.deployed.count_nonzero() == 0
    else:
        assert audio.grad[..., 128:].count_nonzero() > 0
        additions = model.backbone.long_encoders if source == "bass" else model.waveform_encode
        assert all(p.grad.norm() > 0 for p in additions.parameters())


@pytest.mark.parametrize("source,parameters,macs", [
    ("bass", 2114052, 3330240), ("drums", 3032036, 5116864)])
def test_budget_counts_the_executed_operations(source, parameters, macs):
    model = SpecialistSeparator(source=source).eval()
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
    assert budget["parameters"] == parameters
    assert sum(counted) == budget["dense_macs_per_hop"] == macs
    assert budget["persistent_state_elements"] == sum(v.numel() for v in model.initial_state(1))


def test_specialist_geometry_rejects_other_sources_and_mixed_variants():
    for kwargs in ({"source": "vocals"}, {"source": "other"}, {"source": "bass", "waveform_basis": 16},
                   {"source": "drums", "feature_n_fft": 4096}, {"source": "bass", "feature_n_fft": 2048}):
        with pytest.raises(ValueError):
            SpecialistSeparator(**kwargs)
