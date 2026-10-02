"""Portable, continuously streamed evaluation on explicit physical excerpts.

The default excerpt protocol is 30–45s and 75–90s, scored with the original
one-second scale-dependent SDR windows. It reproduces the development panel
protocol when supplied the same 14 tracks; an arbitrary manifest is not that
panel and is never labeled as an independent benchmark automatically.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from ._evaluation.metrics import MetricConfig, SOURCE_ORDER
from ._evaluation.scoring import _aggregate_tracks, _score_track

HOP = 128
SAMPLE_RATE = 44100
DEFAULT_EXCERPTS = ({"start_seconds": 30, "duration_seconds": 15},
                    {"start_seconds": 75, "duration_seconds": 15})


@dataclass(frozen=True)
class EvaluationTrack:
    name: str
    mixture: Path
    sources: tuple[Path, ...]
    intervals: tuple[tuple[int, int], ...]


def _coordinate(value: Any, *, seconds: bool) -> int:
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value < 0):
        raise ValueError("Excerpt coordinates must be finite, nonnegative numbers")
    if seconds:
        return round(value * SAMPLE_RATE)
    if type(value) is not int:
        raise ValueError("Sample coordinates must be integers")
    return value


def load_evaluation_manifest(path: str | Path) -> tuple[list[EvaluationTrack], dict]:
    """Resolve explicit paths relative to the manifest, without dataset discovery.

    Each track supplies ``name``, ``mixture``, and a ``sources`` object keyed by
    drums/bass/vocals/other. ``excerpts`` may be global or per track; each row
    uses start_samples/duration_samples or start_seconds/duration_seconds.
    Omitting excerpts selects 30–45s and 75–90s. Audio must be stereo 44.1kHz;
    files are never resampled, normalized, or silently truncated.
    """
    path = Path(path).resolve()
    document = json.loads(path.read_text())
    if (document.get("schema_version") != 1
            or document.get("sample_rate") != SAMPLE_RATE
            or tuple(document.get("source_order", ())) != SOURCE_ORDER):
        raise ValueError("Expected manifest v1, 44100 Hz, source order drums/bass/vocals/other")
    rows = document.get("tracks")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Manifest must contain at least one explicit track")
    tracks, names = [], set()
    for row in rows:
        name = row.get("name")
        if not isinstance(name, str) or not name.strip() or name in names:
            raise ValueError("Track names must be nonempty and unique")
        names.add(name)
        sources = row.get("sources")
        if not isinstance(sources, dict) or set(sources) != set(SOURCE_ORDER):
            raise ValueError("Every track requires exactly the four named source paths")

        def resolve(value):
            if not isinstance(value, str) or not value:
                raise ValueError("Audio paths must be nonempty strings")
            return (path.parent / value).resolve()

        excerpts = row.get("excerpts", document.get("excerpts", DEFAULT_EXCERPTS))
        if not isinstance(excerpts, (tuple, list)) or not excerpts:
            raise ValueError("Every track requires at least one excerpt")
        intervals = []
        for excerpt in excerpts:
            fields = set(excerpt)
            if fields == {"start_samples", "duration_samples"}:
                start = _coordinate(excerpt["start_samples"], seconds=False)
                duration = _coordinate(excerpt["duration_samples"], seconds=False)
            elif fields == {"start_seconds", "duration_seconds"}:
                start = _coordinate(excerpt["start_seconds"], seconds=True)
                duration = _coordinate(excerpt["duration_seconds"], seconds=True)
            else:
                raise ValueError("Each excerpt requires start and duration in samples or seconds")
            if duration <= 0:
                raise ValueError("Excerpt duration must be positive")
            intervals.append((start, start + duration))
        intervals.sort()
        if any(a[1] > b[0] for a, b in zip(intervals, intervals[1:])):
            raise ValueError("Physical excerpts must not overlap")
        tracks.append(EvaluationTrack(name, resolve(row.get("mixture")),
                                     tuple(resolve(sources[source]) for source in SOURCE_ORDER),
                                     tuple(intervals)))
    return tracks, document


class NativeRenderer:
    """Adapt the current PyTorch model to raw graph output coordinates.

    ``role`` selection (raw optimizer weights versus EMA weights) happens at
    checkpoint loading. Both roles are scored using deployed residual Other.
    """

    def __init__(self, model):
        import torch

        if (model.hop_samples != HOP or model.graph_alignment_samples != HOP
                or tuple(model.architecture_metadata.get("source_order", ())) != SOURCE_ORDER):
            raise ValueError("Unsupported native model geometry or source order")
        if any(module.training for module in model.modules()):
            raise ValueError("Place the complete model in eval mode before evaluation")
        self.model = model
        self.device = next(model.parameters()).device
        if any(value.dtype != torch.float32 for value in (*model.parameters(), *model.buffers())):
            raise ValueError("Native evaluation requires FP32 weights and buffers")
        self.reset()

    def reset(self):
        self.state = self.model.initial_state(1, device=self.device)
        self.previous = np.zeros((2, HOP), dtype=np.float32)

    def render(self, audio):
        import torch

        with torch.inference_mode():
            result = self.model.render(torch.from_numpy(audio[None]).to(self.device), self.state)
        self.state = result.state
        delayed = result.delayed_mixture[0].detach().cpu().numpy()
        expected = np.concatenate((self.previous, audio[:, :-HOP]), axis=-1)
        if not np.array_equal(delayed, expected):
            raise RuntimeError("Native graph no longer emits mixture delayed by 128 samples")
        self.previous = audio[:, -HOP:].copy()
        return result.deployed[0].detach().cpu().numpy()


class OnnxRenderer:
    """Adapt StreamingSeparator while preserving the graph's initial pre-roll."""

    def __init__(self, separator):
        self.separator = separator

    def reset(self):
        self.separator.reset()

    def render(self, audio):
        outputs = []
        for start in range(0, audio.shape[-1], HOP):
            output = self.separator.process_chunk(audio[:, start:start + HOP])
            outputs.append(np.zeros((4, 2, HOP), dtype=np.float32)
                           if output is None else output)
        return np.concatenate(outputs, axis=-1)


def _audio_info(track: EvaluationTrack) -> int:
    import soundfile as sf

    infos = [sf.info(path) for path in (track.mixture, *track.sources)]
    if any(info.samplerate != SAMPLE_RATE or info.channels != 2 for info in infos):
        raise ValueError(f"{track.name}: all audio must be stereo 44100 Hz")
    frames = infos[0].frames
    if frames <= 0 or any(info.frames != frames for info in infos):
        raise ValueError(f"{track.name}: mixture and source lengths must match")
    if any(end > frames for _, end in track.intervals):
        raise ValueError(f"{track.name}: excerpt extends past EOF")
    return frames


def stream_track(renderer, track: EvaluationTrack, *, frames: int, unroll_hops: int = 64):
    """Capture excerpts after continuous inference from sample zero.

    Interior excerpts receive real future input through the final callback.
    Only true EOF is zero-padded and flushed. The +128 graph delay is applied
    exactly once, and every requested physical sample must be captured.
    """
    import soundfile as sf

    if type(unroll_hops) is not int or unroll_hops <= 0:
        raise ValueError("unroll_hops must be a positive integer")
    captures = [(start + HOP, end + HOP) for start, end in track.intervals]
    receive_end = ((max(end for _, end in captures) + HOP - 1) // HOP) * HOP
    outputs = [np.empty((4, 2, end - start), dtype=np.float32)
               for start, end in track.intervals]
    coverage = [0] * len(outputs)
    group = unroll_hops * HOP
    renderer.reset()
    try:
        with sf.SoundFile(track.mixture) as stream:
            for start in range(0, receive_end, group):
                stop = min(start + group, receive_end)
                count = stop - start
                real_count = min(count, max(0, frames - start))
                audio = stream.read(real_count, dtype="float32", always_2d=True).T
                if audio.shape != (2, real_count) or not np.isfinite(audio).all():
                    raise ValueError("Mixture ended early or contains non-finite audio")
                if real_count < count:
                    audio = np.pad(audio, ((0, 0), (0, count - real_count)))
                output = renderer.render(np.ascontiguousarray(audio))
                if output.shape != (4, 2, count) or not np.isfinite(output).all():
                    raise RuntimeError("Invalid streaming output shape or non-finite audio")
                for index, (left, right) in enumerate(captures):
                    lo, hi = max(start, left), min(stop, right)
                    if hi > lo:
                        outputs[index][..., lo - left:hi - left] = output[..., lo - start:hi - start]
                        coverage[index] += hi - lo
    finally:
        renderer.reset()
    if coverage != [end - start for start, end in track.intervals]:
        raise RuntimeError("Incomplete physical excerpt coverage")
    return outputs, {
        "graph_alignment_samples": HOP,
        "reference_intervals": [list(row) for row in track.intervals],
        "capture_intervals": [list(row) for row in captures],
        "received_samples": receive_end,
        "real_input_samples": min(frames, receive_end),
        "zero_input_samples": max(0, receive_end - frames),
        "flush_hops": int(receive_end > ((frames + HOP - 1) // HOP) * HOP),
        "state": "reset per track; continuous from sample zero through unscored prefixes and gaps",
        "coverage_complete": True,
    }


def _read_excerpt(path, start, end):
    import soundfile as sf

    with sf.SoundFile(path) as stream:
        stream.seek(start)
        audio = stream.read(end - start, dtype="float32", always_2d=True).T
    if audio.shape != (2, end - start) or not np.isfinite(audio).all():
        raise ValueError(f"Invalid reference excerpt: {path}")
    return audio


def evaluate_manifest(renderer, manifest_path: str | Path, *, unroll_hops: int = 64,
                      progress=None) -> dict:
    """Score each track independently, preserving historical metric aggregation."""
    path = Path(manifest_path).resolve()
    manifest_digest = sha256(path.read_bytes()).hexdigest()
    tracks, manifest = load_evaluation_manifest(path)
    config = MetricConfig.from_mapping(manifest.get("metrics"))
    if config.sample_rate != SAMPLE_RATE or "low_20_250" not in config.bands_hz:
        raise ValueError("Metrics require 44100 Hz and the low_20_250 band")
    # Validate the complete panel before spending time rendering any track.
    lengths = [_audio_info(track) for track in tracks]
    scores, streams = [], []
    for track, frames in zip(tracks, lengths):
        estimates, stream = stream_track(renderer, track, frames=frames, unroll_hops=unroll_hops)
        mixtures = [_read_excerpt(track.mixture, a, b) for a, b in track.intervals]
        references = [np.stack([_read_excerpt(source, a, b) for source in track.sources])
                      for a, b in track.intervals]
        intervals = [{"reference_start": a, "reference_end": b,
                      "estimate_start": a + HOP, "estimate_end": b + HOP}
                     for a, b in track.intervals]
        score = _score_track(track.name, intervals, mixtures, references, estimates, config)
        scores.append(score)
        streams.append({"name": track.name, **stream})
        if progress is not None:
            progress({"track": track.name, "full_sdr_db": score["full_sdr_db"]})
    if sha256(path.read_bytes()).hexdigest() != manifest_digest:
        raise RuntimeError("Evaluation manifest changed during scoring")
    return {
        "schema_version": 1,
        "manifest_sha256": manifest_digest,
        "source_order": list(SOURCE_ORDER),
        "sample_rate": SAMPLE_RATE,
        "graph_alignment_samples": HOP,
        "normalization": "none",
        "output_policy": "deployed stems; residual Other",
        "metrics": config.to_dict(),
        "metric_source_sha256": sha256(Path(__file__).with_name("_evaluation").joinpath("metrics.py").read_bytes()).hexdigest(),
        "aggregation": "active-window-weighted within each track; equal track means per source; equal source means",
        "tracks": scores,
        "streams": streams,
        "aggregate": _aggregate_tracks(scores),
        "limitations": ["Manifest membership determines development versus independent evaluation.",
                        "Grouped evaluation does not qualify real-time callback timing or host latency."],
    }
