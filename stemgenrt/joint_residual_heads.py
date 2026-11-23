"""Train three streaming specialists together using the error of their physical residual."""
import copy

import torch
from torch import nn

from . import bass_mask_pilot as paired, checkpoint
from .banded import BandSeparator
from .specialist import SpecialistSeparator

STEMS = ("drums", "bass", "vocals")
RECIPE = paired.PairedHeadRecipe("joint-residual-mask-heads-v1", ("bass_only", "all_stems"), ("reference_stems",))


def head_parameters(models):
    if (type(models) is not nn.ModuleDict or tuple(models) != STEMS
            or any(type(models[s]) not in (BandSeparator, SpecialistSeparator) or models[s].sources != (s,) for s in STEMS)):
        raise ValueError("Require the three ordered independent stem specialists")
    params = {k: v for k, v in models.named_parameters() if k.partition(".")[2].startswith("backbone.mask_heads.")}
    if {k.partition(".")[0] for k in params} != set(STEMS):
        raise ValueError("Missing spectral output heads")
    return params


def predictions(drums, bass, vocals, mixture, arm):
    if arm == "bass_only":
        residual = paired.physical_other(mixture, drums.detach(), bass[0], vocals.detach())
    elif arm == "all_stems":
        residual = paired.physical_other(mixture, drums, bass[0], vocals)
    else:
        raise ValueError("Unknown residual-gradient routing")
    return torch.stack((drums, bass[0], vocals, residual, bass[1]))


def normalized_loss(estimated, truth, scales):
    return (paired.errors(estimated, truth) / scales).mean()


def assemble_joint_endpoints(parents, path, digest, arm, parent_digests):
    if arm not in RECIPE.arms or checkpoint.file_sha256(path) != digest:
        raise ValueError("Incorrect joint checkpoint identity or arm")
    saved = paired.validated_payload(torch.load(path, map_location="cpu", weights_only=True), recipe=RECIPE)
    if (saved["base_state_sha256"] != checkpoint.state_sha256(parents.state_dict()) or saved["parents"] != parent_digests
            or saved["step"] != saved["steps"]):
        raise ValueError("Require a complete matched checkpoint and all three original parent weights")
    result = copy.deepcopy(parents).eval().requires_grad_(False)
    paired.apply_heads(result, saved["arms"][arm]["parameters"], parameters=head_parameters)
    for s in STEMS:
        count = parents[s].provenance["training_updates"]
        result[s].provenance = {**checkpoint._raw_provenance(result[s].provenance), "parent_training_updates": count,
            "training_updates": count + saved["step"], "joint_residual_heads": {
                "arm": arm, "source": s, "updates": saved["step"], "parents": parent_digests,
                "plan_sha256": saved["plan_sha256"], "paired_checkpoint_sha256": digest}}
    return result
