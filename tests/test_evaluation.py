"""Physical alignment and frozen scoring tests without trained weights."""
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from stemgenrt.evaluation import (EvaluationTrack, MetricConfig, SOURCE_ORDER,
                                  NativeRenderer, evaluate_manifest, load_evaluation_manifest,
                                  stream_track)
from stemgenrt._evaluation.metrics import windowed_sdr


class DelayedRenderer:
    """One-hop-delayed split with optional lookahead to detect EOF mistakes."""

    def __init__(self, lookahead=0):
        self.lookahead = lookahead
        self.reset()

    def reset(self):
        self.previous = np.zeros((2, 128), dtype=np.float32)

    def render(self, audio):
        delayed = np.concatenate((self.previous, audio[:, :-128]), axis=-1)
        self.previous = audio[:, -128:].copy()
        return np.stack([0.25 * (delayed + self.lookahead * audio)] * 4)


def test_native_renderer_uses_current_model_contract_and_carries_state(monkeypatch):
    import torch
    from stemgenrt.model import StemgenRT58
    torch.set_num_threads(1)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(71)
        model = StemgenRT58().eval().requires_grad_(False)
    audio = np.random.default_rng(72).normal(0, .02, (2, 640)).astype('float32')
    with torch.inference_mode():
        expected = model.render(torch.from_numpy(audio[None])).deployed[0].numpy()
    renderer = NativeRenderer(model)
    actual = np.concatenate([renderer.render(audio[:, :128]), renderer.render(audio[:, 128:384]),
                             renderer.render(audio[:, 384:])], axis=-1)
    np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=0)
    renderer.reset()
    np.testing.assert_array_equal(renderer.render(audio), expected)
    wrong = model.architecture_metadata
    wrong['source_order'] = list(reversed(SOURCE_ORDER))
    monkeypatch.setattr(StemgenRT58, 'architecture_metadata', property(lambda self: wrong))
    with pytest.raises(ValueError, match='source order'):
        NativeRenderer(model)


def write_audio(path, audio):
    sf.write(path, audio.T, 44100, subtype="FLOAT")


@pytest.mark.parametrize("frames", [1, 127, 128, 129, 257, 1024])
def test_exact_alignment_and_final_partial_hop(tmp_path, frames):
    audio = np.random.default_rng(123).standard_normal((2, frames)).astype("float32")
    path = tmp_path / "mixture.wav"
    write_audio(path, audio)
    track = EvaluationTrack("test", path, (path,) * 4, ((0, frames),))
    outputs, meta = stream_track(DelayedRenderer(), track, frames=frames, unroll_hops=3)
    np.testing.assert_array_equal(outputs[0], np.stack([audio * 0.25] * 4))
    assert meta["real_input_samples"] == frames
    assert meta["flush_hops"] == 1
    assert 128 <= meta["zero_input_samples"] < 256
    assert meta["capture_intervals"] == [[128, frames + 128]]


def test_interior_capture_uses_real_future_and_keeps_state_through_gaps(tmp_path):
    audio = np.random.default_rng(7).standard_normal((2, 1500)).astype("float32")
    path = tmp_path / "mixture.wav"
    write_audio(path, audio)
    intervals = ((173, 281), (803, 1119))
    track = EvaluationTrack("test", path, (path,) * 4, intervals)
    # This model uses the current hop to change the previous hop's output.
    first, meta = stream_track(DelayedRenderer(0.2), track, frames=1500, unroll_hops=1)
    grouped, _ = stream_track(DelayedRenderer(0.2), track, frames=1500, unroll_hops=5)
    for output, other, (start, end) in zip(first, grouped, intervals):
        expected = 0.25 * (audio[:, start:end] + 0.2 * audio[:, start + 128:end + 128])
        np.testing.assert_array_equal(output, np.stack([expected] * 4))
        np.testing.assert_array_equal(output, other)
    assert meta["zero_input_samples"] == meta["flush_hops"] == 0
    assert meta["real_input_samples"] == 1280


def test_frozen_sdr_is_scale_dependent_and_skips_inactive_windows():
    config = MetricConfig(window_samples=128, hop_samples=128)
    ref = np.ones((2, 256))
    ref[:, 128:] = 0
    score = windowed_sdr(ref, ref * 0.5, config)
    assert score["db"] == pytest.approx(6.020599913, abs=1e-8)
    assert score["active_windows"] == 1
    assert score["total_windows"] == 2
    assert windowed_sdr(np.zeros((2, 128)), np.ones((2, 128)), config)["db"] is None


def manifest_file(tmp_path, *, excerpts=None):
    length = 4097
    time = np.arange(length) / 44100
    mixture = np.stack([np.sin(2 * np.pi * 55 * time)] * 2).astype("float32")
    write_audio(tmp_path / "mixture.wav", mixture)
    sources = {}
    for name in SOURCE_ORDER:
        sources[name] = name + ".wav"
        write_audio(tmp_path / sources[name], mixture * 0.25)
    manifest = {"schema_version": 1, "sample_rate": 44100,
                "source_order": list(SOURCE_ORDER),
                "metrics": {"window_samples": 1024, "hop_samples": 1024},
                "tracks": [{"name": "test", "mixture": "mixture.wav", "sources": sources}]}
    if excerpts is not None:
        manifest["excerpts"] = excerpts
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    return path, manifest


def test_manifest_defaults_and_explicit_source_order(tmp_path):
    path, manifest = manifest_file(tmp_path)
    tracks, _ = load_evaluation_manifest(path)
    assert tracks[0].intervals == ((30 * 44100, 45 * 44100), (75 * 44100, 90 * 44100))
    assert tracks[0].sources[0] == tmp_path / "drums.wav"
    manifest["source_order"] = list(reversed(SOURCE_ORDER))
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="source order"):
        load_evaluation_manifest(path)


def test_scoring_and_reset_between_tracks_are_group_independent(tmp_path):
    path, manifest = manifest_file(tmp_path, excerpts=[{"start_samples": 17, "duration_samples": 4080}])
    manifest["tracks"].append({**manifest["tracks"][0], "name": "second"})
    path.write_text(json.dumps(manifest))
    first = evaluate_manifest(DelayedRenderer(), path, unroll_hops=1)
    second = evaluate_manifest(DelayedRenderer(), path, unroll_hops=8)
    assert first["aggregate"] == second["aggregate"]
    assert first["aggregate"]["full_sdr_db"] == 60
    assert first["aggregate"]["mixture_consistency_db"] == 60
    assert first["tracks"][0]["per_stem"] == first["tracks"][1]["per_stem"]
    assert first["tracks"][0]["per_stem"]["drums"]["active_windows"] == 3
    assert first["streams"][0]["zero_input_samples"] == 255
    json.dumps(first, allow_nan=False)


def test_silent_sources_are_missing_metrics_not_nan_or_artificial_success(tmp_path):
    path, manifest = manifest_file(tmp_path, excerpts=[{"start_samples": 0, "duration_samples": 4097}])
    for filename in ["mixture.wav", *(source + ".wav" for source in SOURCE_ORDER)]:
        write_audio(tmp_path / filename, np.zeros((2, 4097), dtype="float32"))
    result = evaluate_manifest(DelayedRenderer(), path)
    assert result["aggregate"]["full_sdr_db"] is None
    assert result["aggregate"]["val_score"] is None
    assert result["aggregate"]["primary_metrics_complete"] is False
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("excerpts", [
    [{"start_samples": 0, "duration_samples": 0}],
    [{"start_samples": -1, "duration_samples": 1}],
    [{"start_samples": 0.5, "duration_samples": 1}],
    [{"start_samples": 0, "duration_samples": 100}, {"start_samples": 50, "duration_samples": 100}],
])
def test_invalid_excerpt_geometry_is_rejected(tmp_path, excerpts):
    path, _ = manifest_file(tmp_path, excerpts=excerpts)
    with pytest.raises(ValueError):
        load_evaluation_manifest(path)


def test_track_length_mismatch_rejected_before_inference(tmp_path):
    path, _ = manifest_file(tmp_path, excerpts=[{"start_samples": 0, "duration_samples": 100}])
    write_audio(tmp_path / "bass.wav", np.zeros((2, 100), dtype="float32"))
    with pytest.raises(ValueError, match="lengths must match"):
        evaluate_manifest(None, path)
