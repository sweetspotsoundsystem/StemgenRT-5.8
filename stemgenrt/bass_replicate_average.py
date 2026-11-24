"""Equal-weight Bass inference models from authenticated, matched training continuations."""
import re

import torch

from .checkpoint import load_training_checkpoint, state_sha256
from .specialist import SpecialistSeparator


def _load_replicate(path, digest):
    if not isinstance(digest, str) or not re.fullmatch("[0-9a-f]{64}", digest):
        raise ValueError("Supply each training checkpoint SHA-256 digest")
    state = load_training_checkpoint(path, sha256=digest, device="cpu", restore_rng=False)
    model = state.model
    start = state.config.get("data_start")
    batch = state.config.get("batch_size")
    if (type(model) is not SpecialistSeparator or model.sources != ("bass",)
            or state.config.get("precision") != "fp32" or model.provenance.get("parent_training_updates", 0) <= 0
            or type(start) is not int or start < 0 or type(batch) is not int or batch <= 0
            or state.step <= 0 or state.step != state.config.get("steps")
            or state.next_sample_index != start + state.step * batch):
        raise ValueError("Replicate averaging requires complete FP32 Bass continuation checkpoints")
    return start, state, digest


def average_bass_replicates(first_checkpoint, second_checkpoint, *, first_sha256, second_sha256):
    """Authenticate both training runs and average their final CPU EMA parameters."""
    replicas = [_load_replicate(first_checkpoint, first_sha256), _load_replicate(second_checkpoint, second_sha256)]
    replicas.sort(key=lambda entry: entry[0])
    (start_a, first, _), (start_b, second, _) = replicas
    if start_b < first.next_sample_index:
        raise ValueError("Bass replicate training address ranges overlap")
    if (first.config != {**second.config, "data_start": start_a}
            or first.data_identity != second.data_identity or first.ema.decay != second.ema.decay):
        raise ValueError("Bass replicate training recipes and dataset identities must match")
    if (first.model.architecture_metadata != second.model.architecture_metadata
            or first.model.provenance["parent_training_updates"] != second.model.provenance["parent_training_updates"]
            or first.ema.base_state_sha256 != second.ema.base_state_sha256):
        raise ValueError("Bass replicates require identical geometry and the same initial parent state")
    if (first.ema.buffers.keys() != second.ema.buffers.keys()
            or any(not torch.equal(t.view(torch.int32), second.ema.buffers[n].view(torch.int32))
                   for n, t in first.ema.buffers.items())):
        raise ValueError("Bass replicate fixed buffers differ")
    model = first.ema.inference_copy(first.model)
    with torch.no_grad():
        for name, p in model.named_parameters():
            p.copy_((p.double() + second.ema.parameters[name].double()) * 0.5)
    model.provenance.pop("weight_averaging")
    model.provenance.update(checkpoint_weight_role="replicate_average_inference", quality_measured=False,
                            bass_replicate_average={
                                "version": "equal-bass-replicate-ema-v1",
                                "base_state_sha256": first.ema.base_state_sha256,
                                "replicates": [{"checkpoint_sha256": digest, "weight": 0.5, "updates": state.step,
                                                "data_range": [start, state.next_sample_index],
                                                "ema_parameters_sha256": state_sha256(state.ema.parameters)}
                                               for start, state, digest in replicas],
                                "output_state_sha256": state_sha256(model.state_dict()),
                                "additional_optimizer_updates": 0,
                            })
    return model


def average_bass_sampling_runs(checkpoints):
    """Create one equally weighted Bass model from two schedules on each of two streams.

    Supply four (full training checkpoint, expected SHA-256) pairs in any order.
    Every common training setting is checked. The addressed streams must be
    disjoint, and each must contain one fixed and one alternating hard-solo run.
    """
    from ._losses.solo_sampling import VERSION as alternating_version

    records = list(checkpoints)
    if len(records) != 4:
        raise ValueError("Supply four authenticated Bass training checkpoints")
    cells = {}
    for path, digest in records:
        start, state, _ = _load_replicate(path, digest)
        sampler = state.config.get("bass_solo_sampling")
        if sampler not in (None, alternating_version):
            raise ValueError("Sampling average requires fixed and alternating hard-solo training")
        schedule = "fixed" if sampler is None else "alternating"
        cell = (start, schedule)
        if cell in cells:
            raise ValueError("Each stream needs exactly one checkpoint per sampling schedule")
        cells[cell] = (state, digest)
    starts = sorted({start for start, schedule in cells})
    if len(starts) != 2 or set(cells) != {(s, k) for s in starts for k in ("fixed", "alternating")}:
        raise ValueError("Require both sampling schedules on the same two training streams")

    first = cells[starts[0], "fixed"][0]
    if starts[1] < first.next_sample_index:
        raise ValueError("Bass training stream ranges overlap")
    common_config = {k: v for k, v in first.config.items() if k not in ("data_start", "bass_solo_sampling")}
    ordered = []
    parents = []
    for start in starts:
        for schedule in ("fixed", "alternating"):
            state, digest = cells[start, schedule]
            config = {k: v for k, v in state.config.items() if k not in ("data_start", "bass_solo_sampling")}
            if config != common_config or state.data_identity != first.data_identity or state.ema.decay != first.ema.decay:
                raise ValueError("Sampling average requires matching training recipes and dataset identities")
            if (state.model.architecture_metadata != first.model.architecture_metadata
                    or state.model.provenance["parent_training_updates"] != first.model.provenance["parent_training_updates"]
                    or state.ema.base_state_sha256 != first.ema.base_state_sha256):
                raise ValueError("Sampling average requires identical geometry and the same initial parent state")
            if (state.ema.buffers.keys() != first.ema.buffers.keys()
                    or any(not torch.equal(t.view(torch.int32), first.ema.buffers[n].view(torch.int32))
                           for n, t in state.ema.buffers.items())):
                raise ValueError("Bass sampling checkpoints have different fixed buffers")
            optimizer_states = state.optimizer.state
            if (state.ema.updates != state.step or len(optimizer_states) != len(list(state.model.parameters()))
                    or any(v["step"].item() != state.step for v in optimizer_states.values())):
                raise ValueError("Require the complete Adam and EMA state for each final update")
            parents.append({"sampling": schedule, "data_range": [start, state.next_sample_index],
                            "updates": state.step, "weight": 0.25, "checkpoint_sha256": digest,
                            "ema_parameters_sha256": state_sha256(state.ema.parameters)})
            ordered.append(state.ema)

    model = first.ema.inference_copy(first.model)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            total = torch.stack([ema.parameters[name].double() for ema in ordered]).sum(0)
            parameter.copy_(total * 0.25)
    model.provenance.pop("weight_averaging")
    model.provenance.update(checkpoint_weight_role="balanced_sampling_average_inference", quality_measured=False,
                            bass_sampling_average={"version": "equal-two-stream-two-schedule-bass-ema-v1",
                                                   "base_state_sha256": first.ema.base_state_sha256,
                                                   "parents": parents, "output_state_sha256": state_sha256(model.state_dict()),
                                                   "additional_optimizer_updates": 0})
    return model
