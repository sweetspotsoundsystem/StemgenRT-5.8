"""Continuous target-only and target-absent evaluation without normalization.

The selected subset is fixed from the track origin, including unscored prefixes
and gaps. Only real EOF is padded. These controls complement ordinary scores.
"""
from contextlib import ExitStack
import hashlib

import numpy as np

from .evaluation import HOP, NativeRenderer, _audio_info, _read_excerpt
from ._evaluation.metrics import SOURCE_ORDER, db_ratio, frame_ranges, mean_or_none, rms_dbfs
from ._evaluation.scoring import _score_track


def source_subsets(target_source):
    if target_source not in SOURCE_ORDER:
        raise ValueError("Choose a named source for the controlled inputs")
    index = SOURCE_ORDER.index(target_source)
    return {"target_only": (index,), "target_absent": tuple(i for i in range(4) if i != index)}


def combine_sources(sources, included):
    if sources.shape[:2] != (4, 2) or sources.dtype != np.float32:
        raise ValueError("Require four FP32 stereo source recordings")
    return sources[list(included)].sum(axis=0, dtype=np.float32)


def stream_source_views(model, track, *, target_source, unroll_hops=64, require_closure=True):
    """Stream both views through a four-output system with independent states."""
    import soundfile as sf

    views = source_subsets(target_source)
    if type(unroll_hops) is not int or unroll_hops <= 0 or type(require_closure) is not bool:
        raise ValueError("Invalid controlled streaming configuration")
    frames = _audio_info(track)
    captures = [(start + HOP, end + HOP) for start, end in track.intervals]
    receive_end = ((max(end for _, end in captures) + HOP - 1) // HOP) * HOP
    outputs = {view: [np.empty((4, 2, end - start), dtype=np.float32)
                      for start, end in track.intervals] for view in views}
    mixtures = {view: [np.empty((2, end - start), dtype=np.float32)
                       for start, end in track.intervals] for view in views}
    coverage = {view: [0] * len(captures) for view in views}
    renderers = {view: NativeRenderer(model) for view in views}
    digests = {view: hashlib.sha256() for view in views}
    calls = 0
    try:
        with ExitStack() as stack:
            readers = [stack.enter_context(sf.SoundFile(path)) for path in track.sources]
            for start in range(0, receive_end, unroll_hops * HOP):
                stop = min(start + unroll_hops * HOP, receive_end)
                count = stop - start
                real_count = min(count, max(0, frames - start))
                sources = np.stack([reader.read(real_count, dtype="float32", always_2d=True).T
                                    for reader in readers])
                if sources.shape != (4, 2, real_count) or not np.isfinite(sources).all():
                    raise ValueError("Controlled source read ended early or contains nonfinite audio")
                if real_count < count:
                    sources = np.pad(sources, ((0, 0), (0, 0), (0, count - real_count)))
                for view, included in views.items():
                    mixture = combine_sources(sources, included)
                    digests[view].update(np.ascontiguousarray(mixture.T).tobytes())
                    renderer = renderers[view]
                    # NativeRenderer verifies this delay against the model.
                    delayed = np.concatenate((renderer.previous, mixture[:, :-HOP]), axis=-1)
                    estimates = renderer.render(mixture)
                    if estimates.shape != (4, 2, count) or estimates.dtype != np.float32 or not np.isfinite(estimates).all():
                        raise RuntimeError("Controlled output geometry, precision or finiteness changed")
                    for index, (left, right) in enumerate(captures):
                        lo, hi = max(start, left), min(stop, right)
                        if hi > lo:
                            outputs[view][index][..., lo-left:hi-left] = estimates[..., lo-start:hi-start]
                            mixtures[view][index][..., lo-left:hi-left] = delayed[..., lo-start:hi-start]
                            coverage[view][index] += hi - lo
                calls += 1
    finally:
        for renderer in renderers.values():
            renderer.reset()
    references = [np.stack([_read_excerpt(path, start, stop) for path in track.sources])
                  for start, stop in track.intervals]
    closure = {}
    for view, included in views.items():
        if coverage[view] != [stop-start for start, stop in track.intervals]:
            raise RuntimeError("Incomplete controlled excerpt coverage")
        closure[view] = 0.
        for refs, estimates, mixture in zip(references, outputs[view], mixtures[view], strict=True):
            if not np.array_equal(mixture, combine_sources(refs, included)):
                raise RuntimeError("Controlled physical input alignment differs")
            closure[view] = max(closure[view], float(np.max(np.abs(estimates.sum(0, dtype=np.float32) - mixture))))
        if require_closure and closure[view] > 1e-6:
            raise RuntimeError("Controlled mixture reconstruction failed")
    return references, outputs, mixtures, {
        "target_source": target_source, "views": {name: [SOURCE_ORDER[i] for i in indices] for name, indices in views.items()},
        "input_stream_sha256": {view: digest.hexdigest() for view, digest in digests.items()},
        "received_samples": receive_end, "real_input_samples": min(frames, receive_end),
        "zero_input_samples": max(0, receive_end - frames), "actual_forward_calls_per_view": calls,
        "reference_intervals": [list(row) for row in track.intervals],
        "capture_intervals": [list(row) for row in captures], "graph_alignment_samples": HOP,
        "source_subset_fixed_from_track_origin": True, "physical_alignment_verified": True,
        "coverage_complete": True, "reconstruction_max_abs": closure, "require_closure": require_closure,
        "normalization": "none"}


def score_source_views(name, intervals, references, outputs, mixtures, config, *, target_source):
    """Keep silent outputs in leakage statistics whenever the input is active."""
    result = {}
    for view, included in source_subsets(target_source).items():
        desired, windows = [], []
        for excerpt_index, (refs, estimates, mixture) in enumerate(zip(
                references, outputs[view], mixtures[view], strict=True)):
            target = np.zeros_like(refs)
            target[list(included)] = refs[list(included)]
            desired.append(target)
            for start, stop in frame_ranges(mixture.shape[-1], config.window_samples, config.hop_samples):
                segment = mixture[:, start:stop].astype(np.float64)
                input_dbfs = rms_dbfs(segment, config.epsilon)
                input_energy = float(np.square(segment).sum())
                cells = {}
                for index, stem in enumerate(SOURCE_ORDER):
                    value = estimates[index, :, start:stop].astype(np.float64)
                    truth = target[index, :, start:stop].astype(np.float64)
                    truth_active = rms_dbfs(truth, config.epsilon) > config.activity_dbfs
                    cells[stem] = {
                        "output_rms_dbfs": rms_dbfs(value, config.epsilon),
                        "output_to_input_db": db_ratio(float(np.square(value).sum()), input_energy,
                            epsilon=config.epsilon, floor=config.db_floor, ceiling=config.db_ceiling),
                        "desired_active": truth_active,
                        "signed_desired_projection_gain": float((value * truth).sum() / (float(np.square(truth).sum()) + config.epsilon))
                            if truth_active else None, "off_target": index not in included}
                windows.append({"excerpt_index": excerpt_index,
                    "physical_start": int(intervals[excerpt_index]["reference_start"]) + start,
                    "physical_end": int(intervals[excerpt_index]["reference_start"]) + stop,
                    "input_rms_dbfs": input_dbfs, "input_active": input_dbfs > config.activity_dbfs,
                    "per_stem": cells})
        score = _score_track(name, intervals, mixtures[view], desired, outputs[view], config)
        levels = {}
        for index, stem in enumerate(SOURCE_ORDER):
            active = [row["per_stem"][stem] for row in windows if row["input_active"]]
            levels[stem] = {"off_target": index not in included, "input_active_windows": len(active),
                **{field: mean_or_none(cell[field] for cell in active) for field in
                   ("output_rms_dbfs", "output_to_input_db", "signed_desired_projection_gain")}}
        result[view] = {"desired_stems": [SOURCE_ORDER[i] for i in included],
            "standard_scores_on_remixed_references": score,
            "input_active_windows": sum(row["input_active"] for row in windows),
            "total_windows": len(windows), "native_output_levels": levels, "windows": windows}
    return result
