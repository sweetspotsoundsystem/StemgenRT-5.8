"""Penalize Bass regressions relative to a frozen release on quiet and solo inputs."""
import math
import re

import torch

from .._evaluation.metrics import MetricConfig
from ..checkpoint import load_model
from ..model import render_scored_context


METRICS = MetricConfig()
MARGINS_DB = {"ordinary_quiet": 0.5, "removed": 0.5, "solo": 0.1}
VERSION = "bass-release-margin-v1"
SOLO_VERSION = "bass-release-margin-solo-v1"


def validate_solo_multiplier(value):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError("Bass solo margin multiplier must be finite and nonnegative")
    return float(value)


def validate_recipe(weight, digest, *, solo_multiplier=1.):
    validate_solo_multiplier(solo_multiplier)
    if not weight and solo_multiplier != 1.:
        raise ValueError("A solo multiplier requires an enabled Bass release margin")
    if type(weight) not in (int, float) or not math.isfinite(weight) or weight < 0:
        raise ValueError("Bass release margin weight must be finite and nonnegative")
    if weight == 0 and digest is None:
        return
    if not weight or not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError("A positive Bass release margin weight requires a reference checkpoint SHA-256")


def specification(weight, digest, *, solo_multiplier=1.):
    validate_recipe(weight, digest, solo_multiplier=solo_multiplier)
    result = {"version": VERSION, "weight": weight, "reference_sha256": digest, "role": "ema",
            "margins_db": dict(MARGINS_DB), "window_samples": METRICS.window_samples,
            "activity_dbfs": METRICS.activity_dbfs, "epsilon": METRICS.epsilon,
            "ratio_db_limits": [METRICS.db_floor, METRICS.db_ceiling],
            "reduction": "Mean excess over all eligible windows within each condition, with equal thirds"}
    if solo_multiplier != 1.:
        result.update(version=SOLO_VERSION, solo_multiplier=float(solo_multiplier),
                      reduction="(ordinary quiet excess + removed excess + solo multiplier * solo excess) / 3")
    return result


def _energy(audio):
    length = audio.shape[-1] // METRICS.window_samples * METRICS.window_samples
    windows = audio[..., :length].unfold(-1, METRICS.window_samples, METRICS.window_samples)
    return windows.double().square().sum(dim=(1, 3))


def _level(energy):
    return 10 * (energy / (2 * METRICS.window_samples)).clamp_min(METRICS.epsilon).log10()


def _ratio(numerator, denominator):
    ratio = (numerator + METRICS.epsilon) / (denominator + METRICS.epsilon)
    return (10 * ratio.log10()).clamp(METRICS.db_floor, METRICS.db_ceiling)


def _excess(degradation, eligible, name):
    excess = torch.where(eligible, (degradation - MARGINS_DB[name]).relu(), 0.)
    count = eligible.sum()
    average = excess.sum() / count.clamp_min(1)
    return average, {"eligible_windows": int(count), "violating_windows": int((excess > 0).sum()),
                     "mean_excess_db": float(average.detach())}


def margin_loss(student, released, mixture, truth, *, group, solo_multiplier=1.):
    """Score aligned stereo suffixes; reductions include eligible windows within tolerance."""
    validate_solo_multiplier(solo_multiplier)
    if (student.ndim != 3 or student.shape[1] != 2 or student.shape[-1] < METRICS.window_samples
            or student.dtype != torch.float32 or any(
                t.shape != student.shape or t.device != student.device or t.dtype != student.dtype or t.requires_grad
                for t in (released, mixture, truth))):
        raise ValueError("Require aligned FP32 stereo Bass and fixed release, mixture and reference")
    if group == "ordinary":
        quiet = _level(_energy(truth)) <= METRICS.activity_dbfs
        degradation = _level(_energy(student)) - _level(_energy(released))
        value, row = _excess(degradation, quiet, "ordinary_quiet")
        rows = {"ordinary_quiet": row}
    elif group == "auxiliary" and len(student) == 2:
        # Full-context source_views places Bass-removed first and Bass-only second.
        if torch.count_nonzero(truth[0]).item() or not torch.equal(truth[1], mixture[1]):
            raise ValueError("The auxiliary views must remove Bass first and isolate it second")
        input_energy = _energy(mixture[:1])
        released_ratio = _ratio(_energy(released[:1]), input_energy)
        student_ratio = _ratio(_energy(student[:1]), input_energy)
        removed, removed_row = _excess(student_ratio - released_ratio,
                                       _level(input_energy) > METRICS.activity_dbfs, "removed")
        solo_reference = truth[1:].double()
        signal = _energy(solo_reference)
        released_sdr = _ratio(signal, _energy(released[1:] - solo_reference))
        student_sdr = _ratio(signal, _energy(student[1:] - solo_reference))
        solo, solo_row = _excess(released_sdr - student_sdr, _level(signal) > METRICS.activity_dbfs, "solo")
        if solo_multiplier != 1.:
            solo = solo * solo_multiplier
        value, rows = removed + solo, {"removed": removed_row, "solo": solo_row}
    else:
        raise ValueError("Bass release margins require ordinary inputs or the two ordered source views")
    return (value / len(MARGINS_DB)).float(), rows


class BassReleaseMargin:
    def __init__(self, weight, digest, *, checkpoint, device, solo_multiplier=1.):
        self.specification = specification(weight, digest, solo_multiplier=solo_multiplier)
        self.solo_multiplier = float(solo_multiplier)
        if not weight or checkpoint is None:
            raise ValueError("Supply the frozen Bass release checkpoint for margin training")
        # Loading a separately frozen reference must preserve the student's initialization RNG.
        with torch.random.fork_rng(devices=[]):
            model = load_model(checkpoint, expected_sha256=digest, role="ema", device="cpu")
        if tuple(model.architecture_metadata["source_order"]) != ("bass",):
            raise ValueError("The release margin reference must contain exactly one Bass output")
        self.model = model.eval().requires_grad_(False).to(device)
        self.weight = float(weight)

    def __call__(self, student, mixture, truth, *, group, warmup_samples, microbatch):
        with torch.no_grad():
            released = torch.cat([render_scored_context(self.model, part, warmup_samples=warmup_samples,
                                                        carry_state=True).deployed[:, 0]
                                  for part in mixture.split(microbatch)])
        return margin_loss(student, released, mixture[..., warmup_samples:], truth, group=group,
                           solo_multiplier=self.solo_multiplier)
