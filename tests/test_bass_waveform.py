"""Zero correction initialization and useful gradient flow after its first update."""
import pytest
import torch

from stemgenrt.checkpoint import state_sha256
from stemgenrt.specialist import SpecialistSeparator, extend_bass_waveform, BASS_WAVEFORM_VERSION


def test_extension_preserves_weights_flags_rng_audio_and_existing_states():
    torch.manual_seed(149)
    source = SpecialistSeparator(source="bass", band_width=16, global_width=32, layers=1).eval()
    next(source.parameters()).requires_grad_(False)
    digest, rng = state_sha256(source.state_dict()), torch.get_rng_state()
    candidate = extend_bass_waveform(source, waveform_basis=16)
    assert state_sha256(source.state_dict()) == digest and torch.equal(torch.get_rng_state(), rng)
    assert not candidate.training and candidate.architecture_metadata['version'] == BASS_WAVEFORM_VERSION
    assert candidate.waveform_decode.weight.count_nonzero() == 0
    for name, value in source.named_parameters():
        assert candidate.get_parameter(name).requires_grad == value.requires_grad
    for name, value in source.state_dict().items():
        torch.testing.assert_close(candidate.state_dict()[name], value, rtol=0, atol=0)
    old_state = new_state = None
    with torch.inference_mode():
        for count in (128, 128*37, 128*11, 128):
            audio = torch.randn(2, 2, count)*.1
            old, new = source.render(audio, old_state), candidate.render(audio, new_state)
            torch.testing.assert_close(new.deployed, old.deployed, rtol=0, atol=0)
            for got, want in zip(new.state[:4], old.state, strict=True):
                torch.testing.assert_close(got, want, rtol=0, atol=0)
            assert not new.state.waveform_numerator_tail.count_nonzero()
            old_state, new_state = old.state, new.state
    assert candidate.compute_budget()['dense_macs_per_hop'] - source.compute_budget()['dense_macs_per_hop'] == 16*(1024+32)
    assert candidate.compute_budget()['parameters'] - source.compute_budget()['parameters'] == 16*(1024+32+1)
    with pytest.raises(ValueError, match='without a waveform'):
        extend_bass_waveform(candidate)


def test_zero_decoder_receives_gradient_then_encoder_and_gates_learn():
    torch.manual_seed(161)
    source = SpecialistSeparator(source='bass', band_width=16, global_width=32, layers=1)
    model = extend_bass_waveform(source, waveform_basis=16).train()
    audio, target = torch.randn(1,2,1024)*.1, torch.randn(1,1,2,1024)*.03
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, foreach=False)
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        (model.render(audio).deployed-target).square().mean().backward()
        assert torch.isfinite(model.waveform_decode.weight.grad).all()
        assert model.waveform_decode.weight.grad.count_nonzero()
        for module in (model.waveform_encode, model.waveform_gates):
            for parameter in module.parameters():
                assert torch.isfinite(parameter.grad).all()
                assert bool(parameter.grad.count_nonzero()) == bool(step)
        optimizer.step()
