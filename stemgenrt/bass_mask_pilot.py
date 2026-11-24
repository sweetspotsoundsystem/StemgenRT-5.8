"""Paired supervised and teacher-assisted updates of a causal Bass specialist's output heads."""
import copy
from typing import NamedTuple

import numpy as np
import torch

from . import checkpoint
from .specialist import SpecialistSeparator

SCHEMA = "paired-bass-mask-teacher-v1"
ARMS = ("supervised", "student")
LR = 0.0003


class PairedHeadRecipe(NamedTuple):
    schema: str
    arms: tuple[str, str]
    targets: tuple[str, ...]


BASS_RECIPE = PairedHeadRecipe(SCHEMA, ARMS, ("direct", "power"))


def mask_parameters(model):
    if type(model) is not SpecialistSeparator or model.sources != ("bass",):
        raise ValueError("Require a Bass specialist")
    return {name: param for name, param in model.named_parameters() if name.startswith("backbone.mask_heads.")}


def new_students(parent, *, recipe=BASS_RECIPE, parameters=mask_parameters):
    models, optimizers = {}, {}
    for arm in recipe.arms:
        model = copy.deepcopy(parent).eval().requires_grad_(False)
        params = parameters(model)
        for param in params.values():
            param.requires_grad_(True)
        models[arm] = model
        optimizers[arm] = torch.optim.Adam(params.values(), lr=LR, foreach=False)
    return models, optimizers


def physical_other(mixture, drums, bass, vocals):
    return mixture - ((drums + bass) + vocals)


def teacher_target(native_bass, native_other, mix, drums, vocals, role):
    """Project with the two native teacher energies over the scored stereo clip."""
    if role == "direct":
        return native_bass
    if role != "power":
        raise ValueError("Unknown teaching target")
    bass_energy, other_energy = (x.double().square().sum() for x in (native_bass, native_other))
    ratio = (bass_energy / (bass_energy + other_energy).clamp_min(1e-20)).clamp(.05, .95)
    error = physical_other(mix, drums, native_bass, vocals) - native_other
    return native_bass + ratio.float() * error


def errors(estimates, targets):
    return (estimates-targets).square().mean(dim=(-1, -2))


def training_scales(release, truth, mixtures):
    floor = 1e-4 * mixtures.square().mean(dim=(-1, -2))
    return torch.maximum(errors(release, truth), floor).clamp_min(1e-12).detach()


def mask_loss(estimates, truth, scales, teacher_pair=None):
    ordinary = errors(estimates[:2], truth[:2])
    if teacher_pair is not None:
        ordinary = .5 * (ordinary + errors(estimates[:2], teacher_pair.detach()))
    normalized = torch.cat((ordinary, errors(estimates[2:], truth[2:]))) / scales
    return .4 * normalized[:2].sum() + .1 * normalized[2:].sum()


def validated_payload(payload, *, plan_sha256=None, runtime=False, recipe=BASS_RECIPE):
    if (payload.get("schema") != recipe.schema or tuple(payload["arms"]) != recipe.arms
            or payload["target"] not in recipe.targets
            or not 0 < payload["step"] <= payload["steps"] or len(payload["history"]) != payload["step"]
            or plan_sha256 is not None and payload["plan_sha256"] != plan_sha256):
        raise ValueError("Paired training checkpoint identity or cursor differs")
    if runtime and (payload["torch_version"] != str(torch.__version__) or payload["numpy_version"] != np.__version__
                    or torch.cuda.is_initialized() or torch.get_num_threads() != 1):
        raise ValueError("Exact recovery requires the original single-thread CPU runtime")
    for arm in recipe.arms:
        store = payload["arms"][arm]
        if checkpoint.state_sha256(store["parameters"]) != store["state_sha256"]:
            raise ValueError("Bass output head checksum differs")
        opt = store["adam"]
        states = opt["state"]
        groups = opt["param_groups"]
        indices = list(range(len(store["parameters"])))
        if (list(states) != indices or len(groups) != 1 or groups[0]["params"] != indices
                or groups[0]["lr"] != LR or tuple(groups[0]["betas"]) != (.9, .999)
                or groups[0]["eps"] != 1e-8 or groups[0]["weight_decay"] != 0):
            raise ValueError("Bass head Adam configuration differs")
        for i, p in zip(indices, store["parameters"].values(), strict=True):
            state = states[i]
            if (state["step"].item() != payload["step"]
                    or any(t.shape != p.shape or t.dtype != p.dtype or not torch.isfinite(t).all()
                           for t in (state["exp_avg"], state["exp_avg_sq"]))
                    or (state["exp_avg_sq"] < 0).any()):
                raise ValueError("Invalid Adam moments or cursor")
    return payload


def save_step(path, models, optimizers, parents, base_state_sha256, plan_sha256, role, steps, history,
              *, recipe=BASS_RECIPE, parameters=mask_parameters):
    arms = {}
    for arm in recipe.arms:
        tensors = {name: p.detach() for name, p in parameters(models[arm]).items()}
        arms[arm] = {"parameters": tensors, "state_sha256": checkpoint.state_sha256(tensors),
                     "adam": optimizers[arm].state_dict()}
    payload = {"schema": recipe.schema, "plan_sha256": plan_sha256, "parents": parents,
               "base_state_sha256": base_state_sha256, "target": role, "step": len(history),
               "steps": steps, "history": history, "arms": arms, "rng": checkpoint._rng_state(),
               "backend": checkpoint._backend_state(), "torch_version": str(torch.__version__),
               "numpy_version": str(np.__version__)}
    return checkpoint._atomic_save(path, payload)


def apply_heads(model, saved, *, parameters=mask_parameters):
    target = parameters(model)
    checkpoint._validate_tensors(saved, target)
    if list(target) != list(saved):
        raise ValueError("Changed output head order")
    with torch.no_grad():
        for name, p in target.items():
            p.copy_(saved[name])


def restore_students(path, models, optimizers, *, plan_sha256, target, parents, base_state_sha256, schedule,
                     recipe=BASS_RECIPE, parameters=mask_parameters):
    saved = validated_payload(torch.load(path, map_location="cpu", weights_only=True),
                              plan_sha256=plan_sha256, runtime=True, recipe=recipe)
    if (saved["base_state_sha256"] != base_state_sha256 or saved["parents"] != parents
            or saved["target"] != target or saved["steps"] != len(schedule)
            or [r["address"] for r in saved["history"]] != schedule[:saved["step"]]):
        raise ValueError("Parents, teaching target or scheduled addresses changed at recovery")
    for arm in recipe.arms:
        apply_heads(models[arm], saved["arms"][arm]["parameters"], parameters=parameters)
        optimizers[arm].load_state_dict(saved["arms"][arm]["adam"])
    checkpoint._restore_rng(saved["rng"], saved["backend"])
    return saved["history"]


def assemble_endpoint(parent, path, digest, arm, parent_digests):
    """Load the final raw output heads onto the authenticated Bass parent for streaming."""
    if arm not in ARMS or checkpoint.file_sha256(path) != digest:
        raise ValueError("Invalid paired endpoint role or checkpoint digest")
    saved = validated_payload(torch.load(path, map_location="cpu", weights_only=True))
    if (saved["step"] != saved["steps"] or saved["parents"] != parent_digests
            or checkpoint.state_sha256(parent.state_dict()) != saved["base_state_sha256"]):
        raise ValueError("Require a completed pilot on the supplied frozen parents")
    model = copy.deepcopy(parent).eval().requires_grad_(False)
    apply_heads(model, saved["arms"][arm]["parameters"])
    parent_updates = parent.provenance["training_updates"]
    model.provenance = {**checkpoint._raw_provenance(model.provenance),
                        "parent_training_updates": parent_updates, "training_updates": parent_updates+saved["step"],
                        "bass_mask_pilot": {"arm": arm, "target": saved["target"] if arm == "student" else "ground_truth",
                                           "plan_sha256": saved["plan_sha256"], "parents": saved["parents"],
                                           "updates": saved["step"], "paired_checkpoint_sha256": digest}}
    return model
