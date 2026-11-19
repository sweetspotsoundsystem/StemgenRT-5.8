"""Interpolate a Bass continuation EMA with the exact EMA from which it was initialized."""
import copy
import math

import torch

from .checkpoint import state_sha256
from .specialist import SpecialistSeparator


def interpolate_bass_emas(parent, continued, *, trained_fraction):
    """Return an inference-only Bass model. Both endpoints must be authenticated CPU EMAs."""
    if type(trained_fraction) not in (int, float) or not math.isfinite(trained_fraction) or not 0 <= trained_fraction <= 1:
        raise ValueError("trained_fraction must lie between zero and one")
    for model in (parent, continued):
        if type(model) is not SpecialistSeparator or model.sources != ("bass",):
            raise ValueError("Interpolate two single-output Bass specialists")
        if (any(module.training for module in model.modules())
                or any(p.device.type != "cpu" or p.dtype != torch.float32 or p.requires_grad for p in model.parameters())
                or model.provenance.get("checkpoint_weight_role") != "averaged_inference"):
            raise ValueError("Load both Bass EMAs on CPU in frozen inference mode")
        parameters_sha = state_sha256(dict(model.named_parameters()))
        if model.provenance.get("weight_averaging", {}).get("ema_parameters_sha256") != parameters_sha:
            raise ValueError("Bass parameters differ from their authenticated EMA fingerprint")
    parent_sha = state_sha256(parent.state_dict())
    continued_state = state_sha256(continued.state_dict())
    parent_updates = parent.provenance.get("training_updates")
    if (type(parent_updates) is not int or parent_updates <= 0
            or parent.architecture_metadata != continued.architecture_metadata
            or continued.provenance["weight_averaging"].get("base_state_sha256") != parent_sha
            or continued.provenance.get("parent_training_updates") != parent_updates):
        raise ValueError("The continuation must start from this exact Bass EMA and geometry")
    buffers = dict(parent.named_buffers())
    trained_buffers = dict(continued.named_buffers())
    if buffers.keys() != trained_buffers.keys() or any(not torch.equal(x, trained_buffers[n]) for n, x in buffers.items()):
        raise ValueError("The two Bass endpoints must have identical fixed buffers")
    result = copy.deepcopy(continued)
    starts = dict(parent.named_parameters())
    with torch.no_grad():
        for name, parameter in result.named_parameters():
            parameter.copy_(torch.lerp(starts[name].double(), parameter.double(), float(trained_fraction)))
    result.provenance.pop("weight_averaging")
    result.provenance.update(checkpoint_weight_role="interpolated_inference", quality_measured=False,
        bass_ema_interpolation={"version":"parent-continuation-bass-ema-v1","trained_fraction":float(trained_fraction),
            "parent_ema_state_sha256":parent_sha,"continued_ema_state_sha256":continued_state,
            "output_state_sha256":state_sha256(result.state_dict()),"additional_optimizer_updates":0})
    return result
