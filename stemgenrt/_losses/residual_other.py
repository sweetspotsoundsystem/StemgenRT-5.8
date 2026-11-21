"""Ordinary-mixture supervision for Bass through the deployed residual Other."""
from __future__ import annotations

import math
import re

import torch

from ..checkpoint import load_model
from ..model import render_scored_context
from .release_margin import METRICS, _energy, _level, _ratio


MIXTURE_POWER_FLOOR = 1e-6
WINDOW_SDR_VERSION = "bass-residual-other-window-sdr-v1"


def validate_weights(weight, sdr_weight):
    for value in (weight, sdr_weight):
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError("Residual Other weights must be finite and nonnegative")


def validate_recipe(weight, hashes, *, sdr_weight=0.):
    validate_weights(weight, sdr_weight)
    if not (weight or sdr_weight) and hashes is None:
        return
    if (not (weight or sdr_weight) or not isinstance(hashes, dict) or set(hashes) != {"drums", "vocals"}
            or any(not isinstance(v, str) or not re.fullmatch(r"[0-9a-f]{64}", v) for v in hashes.values())):
        raise ValueError("Residual Other supervision requires a positive weight and Drums/Vocals SHA-256 digests")


def specification(weight, hashes, *, sdr_weight=0.):
    validate_recipe(weight, hashes, sdr_weight=sdr_weight)
    result = {"version": "bass-residual-other-v1", "weight": weight, "peer_checkpoint_sha256": hashes,
              "role": "ema", "mixture_power_floor": MIXTURE_POWER_FLOOR}
    if sdr_weight:
        result.update(version=WINDOW_SDR_VERSION, sdr_weight=sdr_weight, window_samples=METRICS.window_samples,
                      hop_samples=METRICS.hop_samples, activity_dbfs=METRICS.activity_dbfs, epsilon=METRICS.epsilon,
                      ratio_db_limits=[METRICS.db_floor, METRICS.db_ceiling],
                      sdr_reduction="Negative mean of eligible crop means over complete active stereo windows")
    return result


def _physical_other(bass, drums, vocals, mixture, reference):
    if (bass.ndim != 3 or bass.shape[1] != 2 or any(
            t.shape != bass.shape or t.device != bass.device or t.requires_grad
            for t in (drums, vocals, mixture, reference))):
        raise ValueError("Require aligned stereo bass and detached peers, mix and Other reference")
    return mixture - ((drums + bass) + vocals)


def _normalized_error(other, mixture, reference):
    error_power = (other - reference).square().mean(dim=(1, 2))
    mixture_power = mixture.square().mean(dim=(1, 2)).clamp_min(MIXTURE_POWER_FLOOR)
    return (error_power / mixture_power).mean()


def normalized_other_error(bass, drums, vocals, mixture, reference):
    """Mean of per-crop stereo errors, normalized by that scored mix's power."""
    return _normalized_error(_physical_other(bass, drums, vocals, mixture, reference), mixture, reference)


def windowed_other_sdr_loss(other, reference):
    """Negative bounded SDR in dB, equally weighting crops with active reference windows."""
    if (other.ndim != 3 or other.shape[1] != 2 or other.shape[-1] < METRICS.window_samples
            or reference.shape != other.shape or reference.device != other.device or reference.requires_grad
            or reference.dtype != torch.float32 or other.dtype != torch.float32):
        raise ValueError("Require aligned stereo FP32 Other and a fixed reference with at least one full window")
    reference = reference.double()
    signal = _energy(reference)
    active = _level(signal) > METRICS.activity_dbfs
    sdr = _ratio(signal, _energy(other.double() - reference))
    counts = active.sum(dim=1)
    crop_scores = (sdr * active).sum(dim=1) / counts.clamp_min(1)
    eligible_crops = (counts > 0).sum()
    loss = -crop_scores.sum() / eligible_crops.clamp_min(1)
    return loss.float(), {"eligible_crops": int(eligible_crops), "active_windows": int(counts.sum())}


class ResidualOtherLoss:
    def __init__(self, weight, hashes, *, drums_checkpoint, vocals_checkpoint, device, sdr_weight=0.):
        self.specification = specification(weight, hashes, sdr_weight=sdr_weight)
        if not (weight or sdr_weight) or drums_checkpoint is None or vocals_checkpoint is None:
            raise ValueError("Supply both frozen peer checkpoints for residual Other supervision")
        self.weight = float(weight)
        self.sdr_weight = float(sdr_weight)
        self.peers = []
        for name, path in (("drums", drums_checkpoint), ("vocals", vocals_checkpoint)):
            model = load_model(path, expected_sha256=hashes[name], role="ema", device=device)
            if tuple(model.architecture_metadata["source_order"]) != (name,):
                raise ValueError("Wrong source in frozen " + name + " checkpoint")
            self.peers.append(model.eval().requires_grad_(False))

    def _prediction(self, bass, mixture, reference, *, warmup_samples, microbatch):
        # Render the same augmented history and flush for both frozen peers.
        with torch.no_grad():
            drums, vocals = [
                torch.cat([render_scored_context(peer, part, warmup_samples=warmup_samples,
                                                 carry_state=True).deployed[:, 0]
                           for part in mixture.split(microbatch)])
                for peer in self.peers
            ]
        return _physical_other(bass, drums, vocals, mixture[..., warmup_samples:], reference)

    def __call__(self, bass, mixture, reference, *, warmup_samples, microbatch):
        other = self._prediction(bass, mixture, reference, warmup_samples=warmup_samples, microbatch=microbatch)
        return _normalized_error(other, mixture[..., warmup_samples:], reference)

    def windowed_terms(self, bass, mixture, reference, *, warmup_samples, microbatch):
        other = self._prediction(bass, mixture, reference, warmup_samples=warmup_samples, microbatch=microbatch)
        sdr, counts = windowed_other_sdr_loss(other, reference)
        return _normalized_error(other, mixture[..., warmup_samples:], reference), sdr, counts
