"""Choose difficult Bass-only source views using the frozen release at original levels."""
import torch

from ..model import render_scored_context
from .release_margin import METRICS, _energy, _level, _ratio

VERSION = "alternating-release-hard-solo-v1"
DEFAULT_INDEX = 15
THRESHOLD_DB = 50.
MIN_ACTIVE_WINDOWS = 2


def specification():
    return {"version": VERSION, "default_index": DEFAULT_INDEX, "candidate_count": 16,
            "screen_on": "even one-based optimizer updates", "minimum_active_windows": MIN_ACTIVE_WINDOWS,
            "maximum_mean_sdr_db_exclusive": THRESHOLD_DB, "reference_role": "released Bass EMA",
            "tie_break": "prefer default, then lowest index"}


def choose_solo(reference, estimate):
    """Rank the aligned, post-warmup Bass crops using full, active one-second windows."""
    if (reference.ndim != 3 or reference.shape[:2] != (16, 2) or estimate.shape != reference.shape
            or reference.shape[-1] < METRICS.window_samples or reference.dtype != torch.float32
            or estimate.dtype != reference.dtype or estimate.device != reference.device
            or reference.requires_grad or estimate.requires_grad):
        raise ValueError("Require sixteen fixed, aligned FP32 stereo Bass crops and release predictions")
    if not torch.isfinite(reference).all() or not torch.isfinite(estimate).all():
        raise ValueError("Bass solo selection requires finite reference audio and predictions")
    signal = _energy(reference)
    sdr = _ratio(signal, _energy(estimate.double() - reference.double()))
    active = _level(signal) > METRICS.activity_dbfs
    counts = active.sum(dim=1)
    means = (sdr * active).sum(dim=1) / counts.clamp_min(1)
    active_windows = counts.tolist()
    scores = [s if n >= MIN_ACTIVE_WINDOWS else None for s, n in zip(means.tolist(), active_windows, strict=True)]
    eligible = [i for i, score in enumerate(scores) if score is not None and score < THRESHOLD_DB]
    selected = min(eligible, key=lambda i: (scores[i], i != DEFAULT_INDEX, i)) if eligible else DEFAULT_INDEX
    return {"selected_index": selected, "screened": True, "active_windows": active_windows,
            "release_solo_sdr_db": scores, "qualifying_crops": len(eligible)}


@torch.no_grad()
def screen_batch(released, bass, *, warmup_samples, microbatch):
    """Score all sixteen isolated crops on the same released model with fresh warmup states."""
    if (released.sources != ("bass",) or any(m.training for m in released.modules())
            or any(p.requires_grad for p in released.parameters()) or type(microbatch) is not int
            or not 1 <= microbatch <= 16 or bass.ndim != 3 or bass.shape[:2] != (16, 2)
            or bass.dtype != torch.float32 or bass.requires_grad):
        raise ValueError("Supply the frozen release, a full Bass batch, and a valid microbatch")
    device = next(released.parameters()).device
    estimates = []
    for part in bass.split(microbatch):
        result = render_scored_context(released, part.to(device), warmup_samples=warmup_samples, carry_state=True)
        estimates.append(result.deployed[:, 0])
    truth = bass[..., warmup_samples:].to(device)
    return choose_solo(truth, torch.cat(estimates))


def select_for_update(released, bass, *, step, warmup_samples, microbatch):
    if type(step) is not int or step < 1:
        raise ValueError("Expected a positive one-based optimizer update")
    if step % 2:
        return {"selected_index": DEFAULT_INDEX, "screened": False}
    return screen_batch(released, bass, warmup_samples=warmup_samples, microbatch=microbatch)
