"""Equal-weight inference model from two matched, independently sampled Bass continuations."""
import re

import torch

from .checkpoint import load_training_checkpoint, state_sha256
from .specialist import SpecialistSeparator


def average_bass_replicates(first_checkpoint, second_checkpoint, *, first_sha256, second_sha256):
    """Authenticate both training runs and average their final CPU EMA parameters."""
    replicas = []
    for path, digest in ((first_checkpoint, first_sha256), (second_checkpoint, second_sha256)):
        if not isinstance(digest, str) or not re.fullmatch("[0-9a-f]{64}", digest):
            raise ValueError("Supply both training checkpoint SHA-256 digests")
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
        replicas.append((start, state, digest))
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
