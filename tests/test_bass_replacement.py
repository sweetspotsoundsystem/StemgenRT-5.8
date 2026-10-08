"""Continuous alignment and literal bass-output checks for partial evaluation."""
import numpy as np
import pytest
import soundfile as sf
import torch

from stemgenrt import checkpoint as ck
from stemgenrt.bass_replacement import BassEvaluationHybrid
from stemgenrt.compact import CompactSeparator, SOURCE_ORDER
from stemgenrt.specialist import SpecialistSeparator
from stemgenrt.replacement import VocalReplacement
from stemgenrt.evaluation import EvaluationTrack, NativeRenderer, stream_track
from stemgenrt.source_views import stream_source_views


def hybrid():
    torch.manual_seed(81)
    parent = CompactSeparator(sources=SOURCE_ORDER, hidden_size=16, layers=1).eval()
    vocals = CompactSeparator(sources=("vocals",), hidden_size=16, layers=1).eval()
    bass = SpecialistSeparator(source="bass", feature_n_fft=4096, waveform_basis=0,
                               band_width=16, global_width=16, layers=1).eval()
    anchored = VocalReplacement(parent, vocals)
    return BassEvaluationHybrid(anchored, bass), anchored, bass


def test_hybrid_preserves_literal_bass_and_anchored_streams_and_states():
    system, anchored, bass = hybrid()
    original = ck.state_sha256(system.state_dict())
    state, anchor_state, bass_state = system.initial_state(1), anchored.initial_state(1), bass.initial_state(1)
    with torch.inference_mode():
        for hops in (1, 3, 7, 2):
            audio = torch.randn(1, 2, 128 * hops) * .02
            result = system.render(audio, state)
            anchor = anchored.render(audio, anchor_state)
            target = bass.render(audio, bass_state)
            assert torch.equal(result.deployed[:, 0], anchor.deployed[:, 0])
            assert torch.equal(result.deployed[:, 2], anchor.deployed[:, 2])
            assert torch.equal(result.deployed[:, 1], target.deployed[:, 0])
            assert torch.equal(result.delayed_mixture, target.delayed_mixture)
            torch.testing.assert_close(result.deployed.sum(1), result.delayed_mixture, rtol=0, atol=1e-7)
            for found, expected in zip(result.state.candidate, target.state, strict=True):
                assert torch.equal(found, expected)
            state, anchor_state, bass_state = result.state, anchor.state, target.state
    assert ck.state_sha256(system.state_dict()) == original
    assert system.architecture_metadata["complete_independent_system"] is False


@pytest.mark.parametrize("group", [1, 3, 64])
def test_hybrid_origin_gaps_partial_eof_and_controlled_views(tmp_path, group):
    system, _, bass = hybrid()
    sources = np.random.default_rng(37).normal(0, .02, (4, 2, 1797)).astype(np.float32)
    paths = []
    for i, audio in enumerate(sources):
        path = tmp_path / (str(i) + ".wav")
        sf.write(path, audio.T, 44100, subtype="FLOAT")
        paths.append(path)
    mixture = sources.sum(0, dtype=np.float32)
    mix = tmp_path / "mixture.wav"
    sf.write(mix, mixture.T, 44100, subtype="FLOAT")
    track = EvaluationTrack("synthetic", mix, tuple(paths), ((17, 280), (998, 1523), (1778, 1797)))
    actual, _ = stream_track(NativeRenderer(system), track, frames=1797, unroll_hops=group)
    padded = torch.from_numpy(np.pad(mixture, ((0, 0), (0, (-1797) % 128 + 128))))[None]
    state, target = bass.initial_state(1), []
    with torch.inference_mode():
        for start in range(0, padded.shape[-1], group * 128):
            result = bass.render(padded[..., start:start + group * 128], state)
            target.append(result.deployed[0, 0].numpy())
            state = result.state
    target = np.concatenate(target, axis=-1)
    for left, (start, stop) in zip(actual, track.intervals, strict=True):
        np.testing.assert_array_equal(left[1], target[..., start + 128:stop + 128])
    refs, outputs, mixes, meta = stream_source_views(system, track, target_source="bass", unroll_hops=group)
    assert meta["coverage_complete"] and meta["physical_alignment_verified"]
    assert meta["zero_input_samples"] > 0
    for view, indices in (("target_only", [1]), ("target_absent", [0, 2, 3])):
        for ref, out, audio in zip(refs, outputs[view], mixes[view], strict=True):
            np.testing.assert_array_equal(audio, ref[indices].sum(0, dtype=np.float32))
            np.testing.assert_allclose(out.sum(0), audio, rtol=0, atol=1e-7)


@pytest.mark.parametrize("failure", ["source", "geometry", "training"])
def test_hybrid_rejects_incompatible_child(failure, monkeypatch):
    _, anchored, bass = hybrid()
    if failure == "source": bass = CompactSeparator(sources=("vocals",), hidden_size=16, layers=1).eval()
    if failure == "geometry":
        original = SpecialistSeparator.architecture_metadata
        monkeypatch.setattr(SpecialistSeparator, "architecture_metadata", property(
            lambda self: {**original.fget(self), "sample_rate": 48000}))
    if failure == "training": bass.train()
    with pytest.raises(ValueError):
        BassEvaluationHybrid(anchored, bass)

