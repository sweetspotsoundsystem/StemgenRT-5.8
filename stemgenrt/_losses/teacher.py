"""Optional teacher term for ordinary ground-truth-active source windows.

This module adds no teacher term to auxiliary views and changes no student
model or existing ground-truth loss. A production coefficient is not selected.
"""
from dataclasses import dataclass

import torch

from ..losses import BatchReduction, WINDOW, ACTIVITY_POWER, eligible_source_mean

VERSION = "prospective-ordinary-gt-active-relative-teacher-l1-v1"


@dataclass(frozen=True)
class TeacherTerm:
    total: torch.Tensor
    per_stem: torch.Tensor
    active_window_counts: torch.Tensor


def contribution(deployed, teacher_targets, targets, mixture, reduction, *, target_source=None):
    """Sum microbatch contributions using full logical-batch GT denominators.

Teacher audio is a fixed FP32 constant with physical alignment and source order
established by the caller. Ground truth controls eligibility and normalization;
teacher silence or leakage cannot change the support of the term.
"""
    if (not isinstance(reduction, BatchReduction) or deployed.ndim != 4
            or deployed.shape != teacher_targets.shape or deployed.shape != targets.shape
            or deployed.shape[1:3] != (4, 2) or not 0 < deployed.shape[0] <= reduction.examples
            or deployed.shape[-1] != reduction.samples or reduction.samples < WINDOW
            or mixture.shape != (deployed.shape[0], 2, deployed.shape[-1])
            or any(x.dtype != torch.float32 or x.device != deployed.device
                   or not bool(torch.isfinite(x).all()) for x in (deployed, teacher_targets, targets, mixture))
            or any(x.requires_grad or x.grad_fn is not None or torch.is_inference(x)
                   for x in (teacher_targets, targets, mixture))
            or any(x.shape != (4,) or x.dtype != torch.int64 or x.device != deployed.device
                   or x.requires_grad or bool((x < 0).any()) for x in (reduction.active, reduction.absent))
            or not bool(torch.all(reduction.active + reduction.absent ==
                                  reduction.examples * (reduction.samples // WINDOW)))):
        raise ValueError("Require aligned finite FP32 estimates, detached normal-tensor references and full GT counts")
    windows = reduction.samples // WINDOW
    with torch.autocast(deployed.device.type, enabled=False):
        estimate = deployed[..., :windows * WINDOW].unflatten(-1, (windows, WINDOW))
        teacher = teacher_targets[..., :windows * WINDOW].unflatten(-1, (windows, WINDOW))
        truth = targets[..., :windows * WINDOW].unflatten(-1, (windows, WINDOW))
        physical = mixture[..., :windows * WINDOW].unflatten(-1, (windows, WINDOW))
        power = truth.square().sum((2, 4)) / (2 * WINDOW)
        active = power > ACTIVITY_POWER
        active_counts = active.sum((0, 2))
        if bool((active_counts > reduction.active).any()) or bool(((~active).sum((0, 2)) > reduction.absent).any()):
            raise ValueError("Microbatch GT support exceeds declared full-group counts")
        scale = torch.maximum(power.sqrt(), .1 * physical.square().mean((1, 3)).sqrt()[:, None]).clamp_min(1e-3)
        distance = (estimate - teacher).abs().mean((2, 4)) / scale
        per_stem = torch.where(active, distance, 0).sum((0, 2)) / reduction.active.clamp_min(1)
        total = eligible_source_mean(per_stem, reduction.active, target_source)
    if not bool(torch.isfinite(total)):
        raise FloatingPointError("Nonfinite teacher contribution")
    return TeacherTerm(total, per_stem, active_counts)


def policy():
    return {"version": VERSION, "ordinary_only": True, "auxiliary_teacher_weight": 0.,
        "target": "Detached FP32 teacher output on the exact final augmented warmup-plus-scored crop",
        "target_output_policy": "Native Drums/Bass/Vocals; Other equals physical mixture minus DBV",
        "source_order": ["drums", "bass", "vocals", "other"],
        "alignment": "Teacher zero-shift physical samples; score suffix after the existing detached student warmup",
        "eligibility": "Ground-truth-active complete one-second windows, unchanged power threshold 1e-5",
        "distance": "Channel/sample mean absolute deployed-student minus teacher waveform",
        "normalization": "max(GT window RMS, 0.1 mixture window RMS, 1e-3)",
        "reduction": "Full ordinary-group GT eligible-window count per stem, then eligible-stem mean",
        "partial_tail": "Excluded from this term; existing waveform/spectral supervision remains unchanged",
        "ground_truth_loss_changed": False, "absence_loss_changed": False,
        "residual_coupling_limitation": "No direct absent-stem teacher term; indirect native-head gradients through residual Other remain possible",
        "student_architecture_changed": False, "production_coefficient_selected": False,
        "production_recipe_selected": False}
