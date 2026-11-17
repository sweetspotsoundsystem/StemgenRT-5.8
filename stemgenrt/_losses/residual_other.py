"""Ordinary-mixture supervision for Bass through the deployed residual Other."""
from __future__ import annotations

import math
import re

import torch

from ..checkpoint import load_model
from ..model import render_scored_context


MIXTURE_POWER_FLOOR = 1e-6


def validate_recipe(weight, hashes):
    if type(weight) not in (int, float) or not math.isfinite(weight) or weight < 0:
        raise ValueError("Residual Other weight must be finite and nonnegative")
    if weight == 0 and hashes is None:
        return
    if (not weight or not isinstance(hashes, dict) or set(hashes) != {"drums", "vocals"}
            or any(not isinstance(v, str) or not re.fullmatch(r"[0-9a-f]{64}", v) for v in hashes.values())):
        raise ValueError("Residual Other supervision requires a positive weight and Drums/Vocals SHA-256 digests")


def specification(weight, hashes):
    validate_recipe(weight, hashes)
    return {"version": "bass-residual-other-v1", "weight": weight, "peer_checkpoint_sha256": hashes,
            "role": "ema", "mixture_power_floor": MIXTURE_POWER_FLOOR}


def normalized_other_error(bass, drums, vocals, mixture, reference):
    """Mean of per-crop stereo errors, normalized by that scored mix's power."""
    if (bass.ndim != 3 or bass.shape[1] != 2 or any(
            t.shape != bass.shape or t.device != bass.device or t.requires_grad
            for t in (drums, vocals, mixture, reference))):
        raise ValueError("Require aligned stereo bass and detached peers, mix and Other reference")
    other = mixture - ((drums + bass) + vocals)
    error_power = (other - reference).square().mean(dim=(1, 2))
    mixture_power = mixture.square().mean(dim=(1, 2)).clamp_min(MIXTURE_POWER_FLOOR)
    return (error_power / mixture_power).mean()


class ResidualOtherLoss:
    def __init__(self, weight, hashes, *, drums_checkpoint, vocals_checkpoint, device):
        self.specification = specification(weight, hashes)
        if not weight or drums_checkpoint is None or vocals_checkpoint is None:
            raise ValueError("Supply both frozen peer checkpoints for residual Other supervision")
        self.weight = float(weight)
        self.peers = []
        for name, path in (("drums", drums_checkpoint), ("vocals", vocals_checkpoint)):
            model = load_model(path, expected_sha256=hashes[name], role="ema", device=device)
            if tuple(model.architecture_metadata["source_order"]) != (name,):
                raise ValueError("Wrong source in frozen " + name + " checkpoint")
            self.peers.append(model.eval().requires_grad_(False))

    def __call__(self, bass, mixture, reference, *, warmup_samples, microbatch):
        # Render the exact augmented ordinary mixture, including its history.
        # The aligned renderer advances the delayed synthesis through one flush.
        with torch.no_grad():
            drums, vocals = [
                torch.cat([render_scored_context(peer, part, warmup_samples=warmup_samples,
                                                 carry_state=True).deployed[:, 0]
                           for part in mixture.split(microbatch)])
                for peer in self.peers
            ]
        return normalized_other_error(bass, drums, vocals, mixture[..., warmup_samples:], reference)
