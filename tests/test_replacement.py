import numpy as np
import pytest
import torch
import soundfile as sf

from stemgenrt.compact import CompactSeparator, SOURCE_ORDER
from stemgenrt.evaluation import NativeRenderer, EvaluationTrack, stream_track
from stemgenrt.replacement import VocalReplacement


def models(sources):
    return (CompactSeparator(sources=SOURCE_ORDER, hidden_size=16, layers=1).eval(),
            CompactSeparator(sources=sources, hidden_size=16, layers=1).eval())


@pytest.mark.parametrize("sources", [("vocals",), SOURCE_ORDER])
def test_replacement_uses_fixed_db_candidate_v_and_residual_other(sources):
    parent, candidate = models(sources)
    system = VocalReplacement(parent, candidate)
    audio = torch.randn(2, 2, 7 * 128) * .1
    with torch.inference_mode():
        result = system.render(audio)
        base, proposal = parent.render(audio), candidate.render(audio)
    torch.testing.assert_close(result.deployed[:, :2], base.deployed[:, :2], rtol=0, atol=0)
    torch.testing.assert_close(result.deployed[:, 2], proposal.deployed[:, sources.index("vocals")], rtol=0, atol=0)
    torch.testing.assert_close(result.deployed.sum(1), result.delayed_mixture, rtol=0, atol=1e-7)
    assert system.architecture_metadata["deployment_candidate"] is False


@pytest.mark.parametrize("sources", [("vocals",), SOURCE_ORDER])
@pytest.mark.parametrize("group", [1, 3, 64])
def test_streamed_replacement_excerpts_match_aligned_native_audio_through_eof(tmp_path, sources, group):
    system = VocalReplacement(*models(sources))
    audio = np.random.default_rng(17).normal(0, .02, (2, 1797)).astype(np.float32)
    path = tmp_path / "mixture.wav"
    sf.write(path, audio.T, 44100, subtype="FLOAT")
    intervals = ((17, 280), (998, 1523), (1778, 1797))
    track = EvaluationTrack("synthetic alignment", path, (path,) * 4, intervals)
    outputs, metadata = stream_track(NativeRenderer(system), track, frames=audio.shape[-1], unroll_hops=group)
    padding = (-audio.shape[-1]) % 128
    with torch.inference_mode():
        padded = torch.from_numpy(np.pad(audio, ((0, 0), (0, padding + 128))))
        full = system.render(padded[None])
    expected = full.deployed[0].numpy()[..., 128:128 + audio.shape[-1]]
    for estimate, (left, right) in zip(outputs, intervals):
        np.testing.assert_allclose(estimate, expected[..., left:right], rtol=1e-5, atol=1e-7)
        np.testing.assert_allclose(estimate.sum(0), audio[:, left:right], rtol=0, atol=1e-7)
    assert metadata["graph_alignment_samples"] == 128


def test_replacement_preserves_released_model_drums_and_bass():
    from stemgenrt.model import StemgenRT58
    parent = StemgenRT58().eval()
    candidate = CompactSeparator(hidden_size=16, layers=1).eval()
    audio = torch.randn(1, 2, 256) * .01
    with torch.inference_mode():
        base = parent.render(audio)
        result = VocalReplacement(parent, candidate).render(audio)
    torch.testing.assert_close(base.deployed[:, :2], result.deployed[:, :2], rtol=0, atol=0)


def test_replacement_rejects_training_mode():
    parent, candidate = models(("vocals",))
    candidate.train()
    with pytest.raises(ValueError, match="eval mode"):
        VocalReplacement(parent, candidate)
