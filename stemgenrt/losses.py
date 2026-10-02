"""The supported reconstruction/SDR objective and complete-group VJP replay.

Ordinary examples and auxiliary source views have independent denominators.
The auxiliary instrumental/vocals contributions retain their joint two-view
reduction before weights 1 and .25 and the outer .1 coefficient are applied.
Neural outputs are captured and replayed so the scalar and its output gradients
are evaluated once per complete group, without retaining all neural graphs.
"""
from contextlib import nullcontext
from dataclasses import dataclass
import math

import torch

WINDOW = 44100
ACTIVITY_POWER = 1e-5
ABSENCE_WEIGHT = .5
ANCHOR_WEIGHT = .05
SDR_WEIGHT = .2
FFT_SIZES = (512, 1024, 2048)
AUXILIARY_WEIGHT = .1
VIEW_INDICES = (14, 15)
VIEW_NAMES = ("instrumental", "vocals_only")
VIEW_WEIGHTS = (1., .25)
VERSION = "whole-group-wave-spectral-sdr-weighted-source-views-v1"
SOURCE_NAMES = ("drums", "bass", "vocals", "other")


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def source_index(target_source):
    """None retains joint training; a source name selects every loss component."""
    if target_source is None:
        return None
    if not isinstance(target_source, str) or target_source not in SOURCE_NAMES:
        raise ValueError("target_source must be None or a supported source name")
    return SOURCE_NAMES.index(target_source)


def source_mean(values, target_source, *, dim=1):
    index = source_index(target_source)
    return values.mean() if index is None else values.select(dim, index).mean()


def eligible_source_mean(values, counts, target_source):
    index = source_index(target_source)
    if index is None:
        return values.sum() / (counts > 0).sum().clamp_min(1)
    # Per-source values already use clamped active/absent denominators. No
    # eligible windows means zero contribution, not an extra division by four.
    return values[index]


@dataclass(frozen=True)
class DirectSDRLoss:
    total: torch.Tensor
    negative_sdr_db: torch.Tensor
    per_stem_negative_sdr_db: torch.Tensor
    absence_db: torch.Tensor
    relative_l1_anchor: torch.Tensor
    active_window_counts: torch.Tensor
    absent_window_counts: torch.Tensor


def direct_sdr(raw, deployed, targets, mixture, *, target_source=None):
    """Use full one-second windows and the evaluation's fixed activity rule.

The main loss is negative scale-dependent SDR in dB, with equal stem weight.
Inactive windows receive a mixture-relative leakage penalty. A small relative
L1 term also anchors the pre-residual heads, including the learned Other head.
All coefficients affect training only. No teacher or fitted gain is used.
"""
    if (raw.ndim != 4 or raw.shape != deployed.shape or raw.shape != targets.shape
            or raw.shape[1:3] != (4, 2) or raw.shape[-1] < WINDOW
            or mixture.shape != (raw.shape[0], 2, raw.shape[-1])
            or any(v.dtype != torch.float32 or v.device != raw.device
                   for v in (raw, deployed, targets, mixture))
            or targets.requires_grad or mixture.requires_grad):
        raise ValueError("Require aligned FP32 raw/deployed/truth [B,4,2,T] and fixed physical mixture")
    windows = raw.shape[-1] // WINDOW
    with torch.autocast(raw.device.type, enabled=False):
        reference = targets[..., :windows * WINDOW].unflatten(-1, (windows, WINDOW))
        estimate = deployed[..., :windows * WINDOW].unflatten(-1, (windows, WINDOW))
        raw_windows = raw[..., :windows * WINDOW].unflatten(-1, (windows, WINDOW))
        physical = mixture[..., :windows * WINDOW].unflatten(-1, (windows, WINDOW))
        signal = reference.square().sum(dim=(2, 4))
        error = (estimate - reference).square().sum(dim=(2, 4))
        active = signal / (2 * WINDOW) > ACTIVITY_POWER
        values = (10 * torch.log10((error + 1e-12) / (signal + 1e-12))).clamp(-60, 60)
        active_counts = active.sum(dim=(0, 2))
        per_stem = torch.where(active, values, 0).sum(dim=(0, 2)) / active_counts.clamp_min(1)
        primary = eligible_source_mean(per_stem, active_counts, target_source)
        mixture_power = physical.square().mean(dim=(1, 3))
        leakage = estimate.square().mean(dim=(2, 4))
        absent_counts = (~active).sum(dim=(0, 2))
        absence_values = 10 * torch.log10(1 + leakage / mixture_power[:, None].clamp_min(ACTIVITY_POWER))
        absence_per_stem = torch.where(~active, absence_values, 0).sum(dim=(0, 2)) / absent_counts.clamp_min(1)
        absence = eligible_source_mean(absence_per_stem, absent_counts, target_source)
        scale = torch.maximum(signal / (2 * WINDOW), .01 * mixture_power[:, None]).clamp_min(ACTIVITY_POWER).sqrt()
        anchor = source_mean((raw_windows - reference).abs().mean(dim=(2, 4)) / scale, target_source)
        total = primary + ABSENCE_WEIGHT * absence + ANCHOR_WEIGHT * anchor
    if not bool(torch.isfinite(total)):
        raise FloatingPointError("Nonfinite direct SDR objective")
    return DirectSDRLoss(total, primary, per_stem, absence, anchor, active_counts, absent_counts)


@dataclass(frozen=True)
class ReconstructionLoss:
    total: torch.Tensor
    waveform: torch.Tensor
    spectral: torch.Tensor
    raw_anchor: torch.Tensor
    negative_sdr_db: torch.Tensor
    per_stem_negative_sdr_db: torch.Tensor
    active_window_counts: torch.Tensor
    absent_window_counts: torch.Tensor


def reconstruction(raw, deployed, targets, mixture, *, target_source=None):
    if raw.shape != deployed.shape or raw.shape != targets.shape or raw.ndim != 4 or raw.shape[1:3] != (4, 2):
        raise ValueError("Require aligned four-stem stereo estimates and references")
    if mixture.shape != (raw.shape[0], 2, raw.shape[-1]) or raw.shape[-1] < 44100:
        raise ValueError("Require an aligned physical mixture and at least one scored second")
    if any(t.dtype != torch.float32 or t.device != raw.device for t in (raw, deployed, targets, mixture)):
        raise ValueError("Use aligned FP32 reconstruction tensors")
    if targets.requires_grad or mixture.requires_grad:
        raise ValueError("Training references must remain fixed")
    with torch.autocast(raw.device.type, enabled=False):
        reference_rms = targets.square().mean((2, 3)).sqrt()
        mixture_rms = mixture.square().mean((1, 2)).sqrt()
        scale = torch.maximum(reference_rms, .1 * mixture_rms[:, None]).clamp_min(1e-3)
        waveform = source_mean((deployed - targets).abs().mean((2, 3)) / scale, target_source)
        raw_anchor = source_mean((raw - targets).abs().mean((2, 3)) / scale, target_source)
        components = []
        for size in FFT_SIZES:
            window = torch.hann_window(size, device=raw.device, dtype=torch.float32)
            def spectrum(value):
                return torch.stft(value.reshape(-1, value.shape[-1]), n_fft=size, hop_length=size // 4,
                    window=window, center=False, return_complex=True).unflatten(0, value.shape[:-1])
            estimate = spectrum(deployed)
            reference = spectrum(targets)
            physical = spectrum(mixture)
            # Complex distance retains phase sensitivity; normalization averages
            # channels, bins and frames independently for each sample and stem.
            denominator = torch.maximum(reference.abs().mean((2, 3, 4)),
                .1 * physical.abs().mean((1, 2, 3))[:, None]).clamp_min(1e-3)
            components.append(source_mean((estimate - reference).abs().mean((2, 3, 4)) / denominator,
                                          target_source))
        spectral = torch.stack(components).mean()
        total = waveform + .25 * spectral + .25 * raw_anchor
        with torch.no_grad():
            metric = direct_sdr(raw, deployed, targets, mixture, target_source=target_source)
    if not bool(torch.isfinite(total)):
        raise FloatingPointError("Nonfinite waveform/spectral loss")
    return ReconstructionLoss(total, waveform, spectral, raw_anchor, metric.negative_sdr_db,
        metric.per_stem_negative_sdr_db, metric.active_window_counts, metric.absent_window_counts)


@dataclass(frozen=True)
class BlendedLoss:
    total: torch.Tensor
    waveform: torch.Tensor
    spectral: torch.Tensor
    raw_anchor: torch.Tensor
    negative_sdr_db: torch.Tensor
    per_stem_negative_sdr_db: torch.Tensor
    active_window_counts: torch.Tensor
    absent_window_counts: torch.Tensor
    reconstruction_loss: torch.Tensor
    direct_sdr_loss: torch.Tensor
    absence_db: torch.Tensor
    direct_raw_anchor: torch.Tensor


def _extra_primary_weight(value):
    if type(value) not in (float, int) or value not in (0., .2):
        raise ValueError("extra_ordinary_primary_sdr_weight must be 0 or 0.2")
    return float(value)


def objective(raw, deployed, targets, mixture, *, extra_ordinary_primary_sdr_weight=0., target_source=None):
    """Both summands carry gradients; the coefficient is fixed before training.

    Reconstruction is wave L1 + .25 complex-STFT + .25 raw-head L1.
    The direct term is negative scale-dependent SDR + .5 absence + .05
    window-normalized raw-head L1. Inference and evaluation are unchanged.
    The optional .2 increment applies only to primary SDR; the absence and
    raw-anchor coefficients and the auxiliary objective retain their weights.
    """
    extra = _extra_primary_weight(extra_ordinary_primary_sdr_weight)
    base = reconstruction(raw, deployed, targets, mixture, target_source=target_source)
    direct = direct_sdr(raw, deployed, targets, mixture, target_source=target_source)
    total = base.total + SDR_WEIGHT * direct.total
    if extra:
        total = total + extra * direct.negative_sdr_db
    if not bool(torch.isfinite(total)):
        raise FloatingPointError("Nonfinite blended reconstruction/SDR loss")
    return BlendedLoss(total, base.waveform, base.spectral, base.raw_anchor,
                       direct.negative_sdr_db, direct.per_stem_negative_sdr_db,
                       direct.active_window_counts, direct.absent_window_counts,
                       base.total, direct.total, direct.absence_db,
                       direct.relative_l1_anchor)


@dataclass(frozen=True)
class BatchReduction:
    examples: int
    samples: int
    active: torch.Tensor
    absent: torch.Tensor


def activity_counts(targets):
    if (targets.ndim != 4 or targets.shape[0] < 1 or targets.shape[1:3] != (4, 2)
            or targets.shape[-1] < WINDOW or targets.dtype != torch.float32 or targets.requires_grad):
        raise ValueError("Require fixed FP32 four-stem stereo references")
    windows = targets.shape[-1] // WINDOW
    with torch.no_grad(), torch.autocast(targets.device.type, enabled=False):
        reference = targets[..., :windows * WINDOW].unflatten(-1, (windows, WINDOW))
        signal = reference.square().sum((2, 4))
        active = signal / (2 * WINDOW) > ACTIVITY_POWER
        return active.sum((0, 2)), (~active).sum((0, 2))


def prepare_reduction(targets):
    active, absent = activity_counts(targets)
    return BatchReduction(targets.shape[0], targets.shape[-1], active, absent)


def contribution(raw, deployed, targets, mixture, reduction, *, target_source=None):
    """Return this microbatch's contribution; sum contributions before Adam.

The reduction is computed on the training device from every reference in the
logical batch. Neither the returned loss nor its gradient needs a further
division by the number of microbatches.
"""
    if (not isinstance(reduction, BatchReduction) or raw.ndim != 4
            or raw.shape != deployed.shape or raw.shape != targets.shape
            or raw.shape[1:3] != (4, 2) or not 0 < raw.shape[0] <= reduction.examples
            or raw.shape[-1] != reduction.samples or reduction.samples < WINDOW
            or mixture.shape != (raw.shape[0], 2, raw.shape[-1])
            or any(v.dtype != torch.float32 or v.device != raw.device for v in (raw, deployed, targets, mixture))
            or targets.requires_grad or mixture.requires_grad
            or any(v.shape != (4,) or v.dtype != torch.int64 or v.device != raw.device
                   or v.requires_grad or bool((v < 0).any()) for v in (reduction.active, reduction.absent))
            or not bool(torch.all(reduction.active + reduction.absent ==
                                  reduction.examples * (reduction.samples // WINDOW)))):
        raise ValueError("Require matching microbatch audio and whole-batch activity counts")
    fraction = raw.shape[0] / reduction.examples
    base = reconstruction(raw, deployed, targets, mixture, target_source=target_source)
    windows = raw.shape[-1] // WINDOW
    with torch.autocast(raw.device.type, enabled=False):
        reference = targets[..., :windows * WINDOW].unflatten(-1, (windows, WINDOW))
        estimate = deployed[..., :windows * WINDOW].unflatten(-1, (windows, WINDOW))
        raw_windows = raw[..., :windows * WINDOW].unflatten(-1, (windows, WINDOW))
        physical = mixture[..., :windows * WINDOW].unflatten(-1, (windows, WINDOW))
        signal = reference.square().sum((2, 4))
        error = (estimate - reference).square().sum((2, 4))
        active = signal / (2 * WINDOW) > ACTIVITY_POWER
        active_counts, absent_counts = active.sum((0, 2)), (~active).sum((0, 2))
        if bool((active_counts > reduction.active).any()) or bool((absent_counts > reduction.absent).any()):
            raise ValueError("Microbatch activity exceeds the declared logical batch")
        values = (10 * torch.log10((error + 1e-12) / (signal + 1e-12))).clamp(-60, 60)
        per_stem = torch.where(active, values, 0).sum((0, 2)) / reduction.active.clamp_min(1)
        primary = eligible_source_mean(per_stem, reduction.active, target_source)
        mixture_power = physical.square().mean((1, 3))
        leakage = estimate.square().mean((2, 4))
        absence_values = 10 * torch.log10(1 + leakage / mixture_power[:, None].clamp_min(ACTIVITY_POWER))
        absence_per_stem = torch.where(~active, absence_values, 0).sum((0, 2)) / reduction.absent.clamp_min(1)
        absence = eligible_source_mean(absence_per_stem, reduction.absent, target_source)
        scale = torch.maximum(signal / (2 * WINDOW), .01 * mixture_power[:, None]).clamp_min(ACTIVITY_POWER).sqrt()
        anchor = source_mean((raw_windows - reference).abs().mean((2, 4)) / scale, target_source) * fraction
        direct = primary + ABSENCE_WEIGHT * absence + ANCHOR_WEIGHT * anchor
        reconstruction_loss = base.total * fraction
        total = reconstruction_loss + SDR_WEIGHT * direct
    if not bool(torch.isfinite(total)):
        raise FloatingPointError("Nonfinite globally normalized microbatch loss")
    return BlendedLoss(total, base.waveform * fraction, base.spectral * fraction,
                       base.raw_anchor * fraction, primary, per_stem, active_counts, absent_counts,
                       reconstruction_loss, direct, absence, anchor)


@dataclass(frozen=True)
class Groups:
    ordinary: BatchReduction
    auxiliary: BatchReduction


def source_views(mixture, targets):
    """Derive additional full-context inputs after ordinary augmentation.

Call on the entire warmup-plus-scored crop. Every view needs freshly computed
warmup states; never remove a source only in the scored suffix or reuse states
from the corresponding ordinary mixture. Inputs are never modified in place.
"""
    if (targets.ndim != 4 or targets.shape[:3] != (16, 4, 2)
            or mixture.shape != (16, 2, targets.shape[-1])
            or targets.dtype != torch.float32 or mixture.dtype != torch.float32
            or targets.requires_grad or mixture.requires_grad or targets.device != mixture.device):
        raise ValueError("Require sixteen fixed FP32 augmented stereo mixtures and references")
    if not bool(torch.isfinite(targets).all()) or not bool(torch.isfinite(mixture).all()):
        raise ValueError("Nonfinite source-view inputs")
    selected = targets[list(VIEW_INDICES)].clone()
    keep = torch.tensor([[True, True, False, True], [False, False, True, False]],
                        dtype=torch.bool, device=targets.device)
    selected *= keep[:, :, None, None]
    return selected.sum(1), selected


def prepare_groups(ordinary_targets, auxiliary_targets):
    if ordinary_targets.shape[0] != 16 or auxiliary_targets.shape[0] != 2:
        raise ValueError("Keep sixteen ordinary examples and two auxiliary views")
    if ordinary_targets.shape[1:] != auxiliary_targets.shape[1:]:
        raise ValueError("Both groups must share scored geometry")
    return Groups(prepare_reduction(ordinary_targets), prepare_reduction(auxiliary_targets))


@dataclass(frozen=True)
class WeightedAuxiliaryLoss:
    total: torch.Tensor
    active_window_counts: torch.Tensor
    absent_window_counts: torch.Tensor
    unweighted_view_contributions: torch.Tensor
    weighted_view_contributions: torch.Tensor


def auxiliary_objective(raw, deployed, targets, mixture, *, weights=VIEW_WEIGHTS, target_source=None):
    """Weight the two complete-group contributions without changing counts.

    Reconstruction keeps its original example denominator of two. Direct SDR
    and absence keep the active/absent counts and eligible-stem counts of both
    views. The outer group coefficient of 0.1 is applied by the accumulator.
    Explicit weights support CPU controls; production uses the fixed default.
    """
    _require(raw.ndim == 4 and raw.shape == deployed.shape == targets.shape
            and raw.shape[:3] == (2, 4, 2) and mixture.shape == (2, 2, raw.shape[-1])
            and type(weights) is tuple and len(weights) == 2
            and all(type(w) is float and math.isfinite(w) for w in weights)
            and weights[0] == 1. and 0 <= weights[1] <= 1.,
            "Require exactly the ordered source-view pair and valid fixed view weights")
    _require(torch.count_nonzero(targets[0, 2]).item() == 0
            and torch.count_nonzero(targets[1, [0, 1, 3]]).item() == 0,
            "Source-view target ordering or removed stems changed")
    reduction = prepare_reduction(targets)
    terms = [contribution(raw[i:i + 1], deployed[i:i + 1], targets[i:i + 1], mixture[i:i + 1], reduction,
                          target_source=target_source)
             for i in (0, 1)]
    unweighted = torch.stack([term.total for term in terms])
    weighted = unweighted * unweighted.new_tensor(weights)
    total = weighted.sum()
    _require(bool(torch.isfinite(total)), "Nonfinite weighted source-view objective")
    return WeightedAuxiliaryLoss(total, reduction.active, reduction.absent, unweighted, weighted)


def _teacher_weight(value):
    if type(value) is not float or not math.isfinite(value) or value < 0:
        raise ValueError("Require a finite nonnegative float teacher coefficient")
    return value


def policy(*, extra_ordinary_primary_sdr_weight=0., teacher_coefficient=0., target_source=None):
    source_index(target_source)
    extra = _extra_primary_weight(extra_ordinary_primary_sdr_weight)
    teacher = _teacher_weight(teacher_coefficient)
    result = {"version": VERSION, "loss": "Unchanged ordinary16 plus 0.1 joint source-view2 with view multipliers [1,0.25]",
            "ordinary_examples": 16, "auxiliary_examples": 2, "auxiliary_weight": AUXILIARY_WEIGHT,
            "view_indices": list(VIEW_INDICES), "view_weights": list(VIEW_WEIGHTS),
            "direct_sdr_weight": SDR_WEIGHT,
            "capture": "Grad-enabled microbatch renders, detached output coordinates only; release each neural graph",
            "output_derivatives": "One FP32 complete-group scalar per group; auxiliary view weights retain complete-group denominators",
            "backward": "Replay each microbatch render with its exact whole-group output derivatives",
            "replay_outputs": "Require bit-exact raw and deployed outputs before every backward",
            "optimizer": "Clear gradients once, complete both groups, clip once, Adam once, EMA once",
            "inference_changed": False, "extra_forward_pass_per_update": True}
    if extra:
        result.update(version="whole-group-primary-sdr-two-fifths-v1",
            loss="Ordinary primary SDR weight .4; unchanged .1 joint source-view loss with weights [1,.25]",
            ordinary_primary_sdr_weight=.4, extra_ordinary_primary_sdr_weight=extra)
    if teacher:
        from ._losses.teacher import policy as teacher_policy
        result.update(version="optional-ordinary-teacher-grouped-update-v1",
            teacher_coefficient=teacher, teacher_term=teacher_policy(),
            teacher_targets="Caller supplies complete ordinary-group detached FP32 scored targets",
            teacher_auxiliary_weight=0., production_recipe_selected=False)
    if target_source is not None:
        result.update(parent_objective_version=result["version"], version="single-source-grouped-update-v1",
            target_source=target_source,
            loss="Selected-source waveform, complex-STFT, raw anchors, SDR, absence and optional teacher terms",
            source_reduction="One selected source; retain complete-group example and window denominators",
            auxiliary_source_selection="Select the same output on original instrumental/vocals-only inputs")
    return result


def accumulate_group(model, mixture, targets, *, group, microbatch, warmup_samples,
                     check_continue=None, verify_input_gradients=False, progress=None,
                     extra_ordinary_primary_sdr_weight=0., teacher_coefficient=0., teacher_targets=None,
                     target_source=None):
    from .model import render_scored_context

    extra = _extra_primary_weight(extra_ordinary_primary_sdr_weight)
    teacher = _teacher_weight(teacher_coefficient)
    if teacher:
        if group != "ordinary":
            raise ValueError("Teacher supervision is ordinary-only")
        if (not isinstance(teacher_targets, torch.Tensor)
                or teacher_targets.shape != targets[..., warmup_samples:].shape
                or teacher_targets.dtype != torch.float32 or teacher_targets.device != targets.device
                or teacher_targets.requires_grad or teacher_targets.grad_fn is not None
                or torch.is_inference(teacher_targets) or not bool(torch.isfinite(teacher_targets).all())):
            raise ValueError("Require aligned finite normal detached teacher targets for the complete ordinary group")

    _require(group in ("ordinary", "auxiliary") and len(mixture) == (16 if group == "ordinary" else 2),
            "Require a complete named source group")
    _require(type(microbatch) is int and 0 < microbatch <= len(mixture)
            and mixture.device == targets.device and not mixture.requires_grad and not targets.requires_grad,
            "Require fixed group inputs and a valid microbatch")
    captured = [[], []]
    for offset in range(0, len(mixture), microbatch):
        if check_continue is not None:
            check_continue()
        output = render_scored_context(model, mixture[offset:offset + microbatch],
                                       warmup_samples=warmup_samples, carry_state=True)
        _require(output.initial_state_detached and output.flush_hops == 1, "Capture context contract changed")
        captured[0].append(output.raw.detach().clone())
        captured[1].append(output.deployed.detach().clone())
        del output
        if progress is not None:
            progress("canonical_capture", group, offset)
    raw, deployed = (torch.cat(values).requires_grad_() for values in captured)
    del captured
    loss_function = objective if group == "ordinary" else auxiliary_objective
    ordinary_options = {"extra_ordinary_primary_sdr_weight": extra} if group == "ordinary" else {}
    terms = loss_function(raw, deployed, targets[..., warmup_samples:], mixture[..., warmup_samples:],
                          target_source=target_source, **ordinary_options)
    value = terms.total if group == "ordinary" else AUXILIARY_WEIGHT * terms.total
    details = {}
    if teacher:
        from ._losses.teacher import contribution as teacher_contribution
        reference = targets[..., warmup_samples:]
        term = teacher_contribution(deployed, teacher_targets, reference, mixture[..., warmup_samples:],
                                    prepare_reduction(reference), target_source=target_source)
        value = terms.total + teacher * term.total
        if not bool(torch.isfinite(value)):
            raise FloatingPointError("Nonfinite combined ordinary teacher loss")
        details = {"teacher_coefficient": teacher, "teacher_loss_unweighted": float(term.total.detach())}
        del term
    loss = float(value.detach())
    if group == "auxiliary":
        details = {
            "view_contribution_multipliers": list(VIEW_WEIGHTS),
            "weighted_view_losses": (AUXILIARY_WEIGHT * terms.weighted_view_contributions).detach().cpu().tolist(),
            "original_weight_view_losses": (AUXILIARY_WEIGHT * terms.unweighted_view_contributions).detach().cpu().tolist(),
        }
    active, absent = terms.active_window_counts.cpu().tolist(), terms.absent_window_counts.cpu().tolist()
    derivatives = tuple(v.detach() for v in torch.autograd.grad(value, (raw, deployed)))
    expected = raw.detach(), deployed.detach()
    del raw, deployed, value, terms
    rows = []
    for offset in range(0, len(mixture), microbatch):
        if check_continue is not None:
            check_continue()
        end = offset + microbatch
        audio = mixture[offset:end]
        if verify_input_gradients:
            audio = audio.detach().clone().requires_grad_()
        output = render_scored_context(model, audio, warmup_samples=warmup_samples, carry_state=True)
        _require(output.initial_state_detached and output.flush_hops == 1
                and torch.equal(output.raw, expected[0][offset:end])
                and torch.equal(output.deployed, expected[1][offset:end]), "Canonical replay outputs differ")
        torch.autograd.backward((output.raw, output.deployed), tuple(v[offset:end] for v in derivatives))
        if verify_input_gradients:
            _require(audio.grad is not None and bool(torch.isfinite(audio.grad).all())
                    and torch.count_nonzero(audio.grad[..., :warmup_samples]) == 0
                    and torch.count_nonzero(audio.grad[..., warmup_samples:]) > 0,
                    "Canonical warmup or scored-input gradients differ")
        rows.append({"offset": offset, "examples": len(audio), "whole_group_output_derivatives_replayed": True})
        del audio, output
        if progress is not None:
            progress("canonical_backward", group, offset)
    return {"examples": len(mixture), "weighted_loss": loss, "active_windows": active,
            "absent_windows": absent, "microbatches": rows, "replay_outputs_bit_exact": True,
            "whole_group_objective_evaluations": 1, **details}


def accumulate_groups(model, mixture_cpu, targets_cpu, *, warmup_samples, ordinary_microbatch=4,
                      auxiliary_microbatch=2, check_continue=None, after_group=None,
                      verify_input_gradients=False, progress=None, extra_ordinary_primary_sdr_weight=0.,
                      teacher_coefficient=0., teacher_targets=None, target_source=None):
    teacher = _teacher_weight(teacher_coefficient)
    auxiliary_mix, auxiliary_targets = source_views(mixture_cpu, targets_cpu)
    device = next(model.parameters()).device
    inputs = {"ordinary": (mixture_cpu.to(device), targets_cpu.to(device)),
              "auxiliary": (auxiliary_mix.to(device), auxiliary_targets.to(device))}
    groups = prepare_groups(inputs["ordinary"][1][..., warmup_samples:], inputs["auxiliary"][1][..., warmup_samples:])
    rows = {}
    for group, microbatch in (("ordinary", ordinary_microbatch), ("auxiliary", auxiliary_microbatch)):
        row = accumulate_group(model, *inputs[group], group=group, microbatch=microbatch,
            warmup_samples=warmup_samples, check_continue=check_continue,
            verify_input_gradients=verify_input_gradients, progress=progress,
            extra_ordinary_primary_sdr_weight=extra_ordinary_primary_sdr_weight,
            teacher_coefficient=teacher if group == "ordinary" else 0.,
            teacher_targets=teacher_targets if group == "ordinary" else None, target_source=target_source)
        reduction = getattr(groups, group)
        _require(row["active_windows"] == reduction.active.cpu().tolist()
                and row["absent_windows"] == reduction.absent.cpu().tolist(), "Canonical group activity differs")
        rows[group] = row
        if after_group is not None:
            after_group(group, row)
    return rows


def _grouped_update(model, optimizer, ema, mixture_cpu, targets_cpu, *, step,
                   warmup_samples, ordinary_microbatch=4, auxiliary_microbatch=2,
                   check_continue=None, after_group=None, extra_ordinary_primary_sdr_weight=0.,
                   teacher_coefficient=0., teacher_targets=None, target_source=None):
    """Accumulate both independently normalized groups, then clip/update once.

    Optional callbacks run only before the final update, and may raise to stop
    accumulation without changing weights, Adam moments or EMA. A later call
    clears any partial gradients. An exception during Adam/EMA is fatal: do
    not retry against a potentially partially updated endpoint.
    """
    from .checkpoint import ParameterEMA, state_sha256

    extra = _extra_primary_weight(extra_ordinary_primary_sdr_weight)
    teacher = _teacher_weight(teacher_coefficient)
    parameters = list(model.parameters())
    _require(type(ema) is ParameterEMA and type(step) is int and step == ema.updates + 1,
            "Grouped update must follow the contiguous EMA endpoint")
    _require(type(optimizer) is torch.optim.Adam and len(parameters) == model.parameter_tensor_count
            and model.training and all(p.requires_grad and p.dtype == torch.float32 for p in parameters)
            and len(optimizer.param_groups) == 1
            and [id(p) for p in optimizer.param_groups[0]["params"]] == [id(p) for p in parameters]
            and set(optimizer.state) == (set() if step == 1 else set(parameters))
            and all(state["step"].item() == step - 1 for state in optimizer.state.values()),
            "Grouped update requires all model parameters at one Adam endpoint")
    _require(type(warmup_samples) is int and warmup_samples > 0 and warmup_samples % 128 == 0
            and mixture_cpu.device.type == targets_cpu.device.type == "cpu"
            and type(ordinary_microbatch) is int and 0 < ordinary_microbatch <= 16
            and type(auxiliary_microbatch) is int and 0 < auxiliary_microbatch <= 2,
            "Invalid grouped update input or microbatch geometry")
    if check_continue is not None:
        check_continue()
    optimizer.zero_grad(set_to_none=True)
    rows = accumulate_groups(model, mixture_cpu, targets_cpu, warmup_samples=warmup_samples,
        ordinary_microbatch=ordinary_microbatch, auxiliary_microbatch=auxiliary_microbatch,
        check_continue=check_continue, after_group=after_group, extra_ordinary_primary_sdr_weight=extra,
        teacher_coefficient=teacher, teacher_targets=teacher_targets, target_source=target_source)
    gradient_norms = {}
    for name, parameter in model.named_parameters():
        _require(parameter.grad is not None and bool(torch.isfinite(parameter.grad).all()),
                "Missing or nonfinite grouped gradient: " + name)
        gradient_norms[name] = float(parameter.grad.norm())
    if check_continue is not None:
        check_continue()
    norm = torch.nn.utils.clip_grad_norm_(parameters, 5., foreach=False, error_if_nonfinite=True)
    optimizer.step()
    ema.update(model, step=step)
    endpoint = {"ema_updates": ema.updates,
                "raw_model_state_sha256": state_sha256(model.state_dict()),
                "ema_parameters_sha256": state_sha256(ema.parameters)}
    _require(ema.updates == step and len(optimizer.state) == model.parameter_tensor_count
            and all(state["step"].item() == step for state in optimizer.state.values()),
            "Grouped update advanced Adam or EMA incorrectly")
    return {"step": step, "accumulation_policy": policy(extra_ordinary_primary_sdr_weight=extra,
            teacher_coefficient=teacher, target_source=target_source), "groups": rows, "weighted_loss": sum(r["weighted_loss"] for r in rows.values()),
            "gradient_norm_before_clip": float(norm), "parameter_gradient_norms": gradient_norms, **endpoint}


def grouped_update(model, optimizer, ema, mixture_cpu, targets_cpu, *, step,
                   warmup_samples, ordinary_microbatch=4, auxiliary_microbatch=2,
                   check_continue=None, after_group=None, share_gru_weights=True,
                   extra_ordinary_primary_sdr_weight=0., teacher_coefficient=0., teacher_targets=None,
                   target_source=None):
    """Complete both loss groups, then clip, advance Adam and update EMA once.

    CUDA BF16 replay shares exactly equal saved GRU transposes by default.
    This only avoids redundant saved copies; arithmetic and precision do not
    change. CPU/FP32 runs use the same replay without the CUDA optimization.
    An interrupted accumulation leaves all parameter/Adam/EMA values unchanged;
    the next call clears partial gradients before starting over.
    """
    from ._losses.saved_gru import share_saved_gru_weights

    source_index(target_source)
    parameter = next(model.parameters())
    sharing = (share_gru_weights and parameter.is_cuda
               and getattr(model, "training_precision", "fp32") == "bf16")
    scope = share_saved_gru_weights(model) if sharing else nullcontext()
    with scope:
        return _grouped_update(model, optimizer, ema, mixture_cpu, targets_cpu, step=step,
            warmup_samples=warmup_samples, ordinary_microbatch=ordinary_microbatch,
            auxiliary_microbatch=auxiliary_microbatch, check_continue=check_continue,
            after_group=after_group, extra_ordinary_primary_sdr_weight=extra_ordinary_primary_sdr_weight,
            teacher_coefficient=teacher_coefficient, teacher_targets=teacher_targets, target_source=target_source)
