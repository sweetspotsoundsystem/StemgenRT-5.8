"""Calibrate parent/continuation interpolation on explicitly supplied training songs."""
import math
import statistics

import numpy as np
import torch
from torch import nn

from .evaluation import MetricConfig, NativeRenderer, _audio_info, _read_excerpt, stream_track
from ._evaluation.scoring import _score_track
from .model import ModelOutput
from .source_views import score_source_views, stream_source_views


class _BassProbe(nn.Module):
    """Adapt one Bass output for the four-stem evaluator's controlled-input measurements."""
    hop_samples = graph_alignment_samples = 128
    flush_hops = 1

    def __init__(self, bass):
        super().__init__()
        if bass.sources != ("bass",) or any(module.training for module in bass.modules()):
            raise ValueError("Supply a frozen Bass model in inference mode")
        self.bass = bass
        self.eval()

    def initial_state(self, batch_size, *, device=None):
        return self.bass.initial_state(batch_size, device=device)

    @property
    def architecture_metadata(self):
        return {"source_order": ["drums", "bass", "vocals", "other"], "hop_samples": 128,
                "graph_alignment_samples": 128, "sample_rate": 44100, "flush_hops": 1}

    def render(self, audio, state=None):
        result = self.bass.render(audio, state)
        bass = result.deployed[:, 0]
        zero = torch.zeros_like(bass)
        values = torch.stack((zero, bass, zero, result.delayed_mixture - bass), dim=1)
        return ModelOutput(values, values, values, torch.zeros_like(values), result.delayed_mixture, result.state, values)


def score_bass_interpolation_track(track, release, models):
    """Score fraction zero from the release and each nonzero fraction with the same Drums/Vocals.

    Supply numeric fractions mapped to their authenticated Bass models, including zero.
    Returns ordinary and controlled per-song scores, and checksums of the controlled inputs.
    """
    if 0.0 not in models:
        raise ValueError("Include the released Bass anchor at fraction zero")
    from .checkpoint import state_sha256
    if state_sha256(models[0.0].state_dict()) != state_sha256(release.bass.state_dict()):
        raise ValueError("The Bass anchor differs from the released mixture renderer")
    intervals = track.intervals
    config = MetricConfig()
    spans = [{"reference_start":a,"reference_end":b,"estimate_start":a+128,"estimate_end":b+128} for a,b in intervals]
    mixes = [_read_excerpt(track.mixture,a,b) for a,b in intervals]
    refs = [np.stack([_read_excerpt(path,a,b) for path in track.sources]) for a,b in intervals]
    frames = _audio_info(track)
    baseline, stream = stream_track(NativeRenderer(release), track, frames=frames)
    if not stream["coverage_complete"] or not all(np.array_equal(x[3], m-((x[0]+x[1])+x[2])) for x,m in zip(baseline,mixes,strict=True)):
        raise ValueError("Invalid physical residual in the release anchor")
    systems = {}
    hashes = None
    for fraction, bass in models.items():
        probe = _BassProbe(bass)
        if fraction == 0.0:
            rendered = baseline
        else:
            estimates, observed = stream_track(NativeRenderer(probe), track, frames=frames)
            if not observed["coverage_complete"]:
                raise RuntimeError("Incomplete Bass coverage")
            rendered = [base.copy() for base in baseline]
            for m,full,out in zip(mixes,rendered,estimates,strict=True):
                full[1] = out[1]
                full[3] = m - ((full[0]+full[1])+full[2])
                if np.max(np.abs(full.sum(0)-m)) > 1e-6:
                    raise RuntimeError("Incomplete mixture reconstruction")
            del estimates
        ordinary = _score_track(track.name,spans,mixes,refs,rendered,config)
        ref2, outputs, cmix, controlled = stream_source_views(probe,track,target_source="bass")
        if not (controlled["physical_alignment_verified"] and controlled["coverage_complete"]
                and all(np.array_equal(a,b) for a,b in zip(refs,ref2,strict=True))):
            raise RuntimeError("Controlled audio coverage or alignment changed")
        if hashes is None:
            hashes = controlled["input_stream_sha256"]
        if hashes != controlled["input_stream_sha256"]:
            raise RuntimeError("Controlled inputs differ between fractions")
        scores = score_source_views(track.name, spans, refs, outputs, cmix, config, target_source="bass")
        views = {view: {"bass_scores": row["standard_scores_on_remixed_references"]["per_stem"]["bass"],
                        "bass_output_levels": row["native_output_levels"]["bass"],
                        "input_active_windows": row["input_active_windows"], "total_windows": row["total_windows"]}
                 for view, row in scores.items()}
        systems[fraction] = {"ordinary": ordinary, "controlled": {"scores": views}}
        del rendered, ref2, outputs, cmix
    return systems, hashes


def summarize_bass_tracks(rows, controls):
    """Aggregate active-window scores with equal weight per eligible song."""
    def average(seq):
        vals = [v for v in seq if v is not None]
        if not vals or not all(math.isfinite(v) for v in vals):
            raise ValueError("Every panel metric needs finite observations")
        return statistics.mean(vals)
    def metric(stem, key):
        return average(row["per_stem"][stem][key] for row in rows)
    def view(name, group, key):
        return average(row["scores"][name][group][key] for row in controls)
    return dict(bass_sdr=metric("bass","full_sdr_db"), other_sdr=metric("other","full_sdr_db"),
                bass_sir=metric("bass","sir_db"), ordinary_absent_dbfs=metric("bass","absent_fp_dbfs"),
                bass_low_sdr=average(row["per_stem"]["bass"]["band_sdr_db"]["low_20_250"] for row in rows),
                target_absent_to_input_db=view("target_absent","bass_output_levels","output_to_input_db"),
                target_only_projection_db=20*math.log10(view("target_only","bass_output_levels","signed_desired_projection_gain")),
                target_only_sdr=view("target_only","bass_scores","full_sdr_db"))


def select_bass_fraction(metrics_by_fraction, candidates, protection_limits):
    """Rank interior fractions by feasibility, worst normalized excess, quality, then fraction."""
    signs = {"bass_sir": -1, "ordinary_absent_dbfs": 1, "target_absent_to_input_db": 1,
             "target_only_projection_db": -1, "target_only_sdr": -1}
    if (not candidates or 0.0 not in metrics_by_fraction
            or any(f not in metrics_by_fraction or not 0.0 < f < 1.0 for f in candidates)):
        raise ValueError("Include the released anchor and at least one scored interior fraction")
    if signs.keys() != protection_limits.keys() or not all(math.isfinite(t) and t > 0 for t in protection_limits.values()):
        raise ValueError("Supply positive finite dB tolerances for all five protections")
    qualities = ("bass_sdr", "other_sdr", "bass_low_sdr")
    required = set(signs).union(qualities)
    base = metrics_by_fraction[0.0]
    scores = {}
    for fraction, metrics in metrics_by_fraction.items():
        if metrics.keys() != required or not all(math.isfinite(v) for v in metrics.values()):
            raise ValueError("Supply all eight finite panel metrics")
        delta = {key: value - base[key] for key, value in metrics.items()}
        over = {k: max(0.0, (signs[k] * delta[k] - limit) / limit) for k, limit in protection_limits.items()}
        worst = max(over.values())
        scores[fraction] = {"metrics": metrics, "delta_vs_release": delta, "passes_protections": worst == 0.0,
                            "normalized_overruns": over, "worst_overrun": worst,
                            "min_sdr_gain": min(delta[k] for k in qualities)}

    def rank(fraction):
        row = scores[fraction]
        return row["worst_overrun"], -row["min_sdr_gain"], fraction

    return min(candidates, key=rank), scores
