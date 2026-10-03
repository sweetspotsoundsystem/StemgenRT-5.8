"""Portable checkpoints for the current streaming model.

Training snapshots own raw weights, Adam, FP32 parameter EMA, all RNG streams,
the absolute data cursor and the complete configuration/data identity. Packed
snapshots are lossless and self-contained.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import tempfile

import numpy as np
import torch

from stemgenrt.model import StemgenRT58, VERSION
from stemgenrt.compact import CompactSeparator, VERSION as COMPACT_VERSION, SOURCE_ORDER
from stemgenrt._checkpoint.codec import CODEC, pack, unpack

SCHEMA = "hs-tasnet-eight-state-training-v1"
COMPACT_SCHEMA = "stemgenrt-compact-training-v1"
NATIVE_SCHEMA = "latency58-branch-memory-inference-v1"
EMA_SCHEMA = "latency58-branch-parameter-ema-v1"


def _validate_geometry_config(config, model):
    if type(model) is CompactSeparator:
        expected_sources = ("vocals",) if config.get("target_source") == "vocals" else SOURCE_ORDER
        _require(config.get("model_family") == "compact"
                 and config.get("target_source") in (None, "vocals")
                 and config.get("compact_hidden_size") == model.hidden_size
                 and config.get("compact_layers") == model.layers
                 and model.sources == expected_sources
                 and config.get("teacher_coefficient", 0.) == 0.,
                 "Training configuration differs from the compact architecture")
        _require(config.get("past_filter", False) is False and config.get("attention_window", 32) == 32,
                 "Compact configuration contains unsupported attention/filter overrides")
        return
    _require(config.get("model_family", "released") == "released", "Training model family differs")
    _require(type(config.get("past_filter", False)) is bool
             and config.get("past_filter", False) is False,
             "Training configuration past filter differs from the model")
    window = config.get("attention_window", 32)
    _require(type(window) is int and window == model.attention_window,
             "Training configuration attention window differs from the model")


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def state_sha256(state):
    """Historical tensor fingerprint, retained byte-for-byte for native imports."""
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str((tuple(tensor.shape), tensor.dtype)).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu_tree(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(_cpu_tree(item) for item in value)
    return copy.deepcopy(value)


def _json_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _raw_provenance(provenance):
    result = copy.deepcopy(provenance)
    if "weight_averaging" in result:
        result["initial_parent_weight_averaging"] = result.pop("weight_averaging")
    result["checkpoint_weight_role"] = "raw_optimizer_endpoint"
    return result


def _validate_tensors(stored, current):
    _require(isinstance(stored, dict) and set(stored) == set(current), "Checkpoint tensor inventory differs")
    _require(all(isinstance(value, torch.Tensor) and value.dtype == torch.float32
                 and value.shape == current[name].shape and bool(torch.isfinite(value).all())
                 for name, value in stored.items()), "Invalid checkpoint tensor shape, dtype or values")


class ParameterEMA:
    """One FP32 parameter update after each Adam update; fixed buffers stay fixed."""

    def __init__(self, model, *, decay=.995, base_state_sha256=None):
        _require(type(model) in (StemgenRT58, CompactSeparator) and type(decay) is float
                 and math.isfinite(decay) and 0 <= decay < 1, "Invalid EMA model or decay")
        self.decay, self.updates = decay, 0
        self.architecture = copy.deepcopy(model.architecture_metadata)
        self.parameters = {name: p.detach().clone() for name, p in model.named_parameters()}
        self.buffers = {name: b.detach().clone() for name, b in model.named_buffers()}
        self.raw_state_sha256 = state_sha256(model.state_dict())
        self.base_state_sha256 = base_state_sha256 or self.raw_state_sha256
        _require(self.base_state_sha256 == self.raw_state_sha256, "EMA base model differs")
        self._validate_model(model)

    def _validate_model(self, model):
        _require(type(model) in (StemgenRT58, CompactSeparator) and model.architecture_metadata == self.architecture,
                 "EMA architecture differs")
        parameters, buffers = dict(model.named_parameters()), dict(model.named_buffers())
        _require(list(parameters) == list(self.parameters) and len(parameters) == model.parameter_tensor_count
                 and list(buffers) == list(self.buffers), "EMA tensor inventory differs")
        for stored, current in ((self.parameters, parameters), (self.buffers, buffers)):
            _validate_tensors(stored, current)
            _require(all(value.device == current[name].device for name, value in stored.items()),
                     "EMA tensor devices differ")
        _require(all(torch.equal(value.view(torch.int32), buffers[name].view(torch.int32))
                     for name, value in self.buffers.items()), "EMA fixed buffers changed")
        _require(all(bool(torch.isfinite(p).all()) for p in parameters.values()), "Raw parameters are non-finite")

    @torch.no_grad()
    def update(self, model, *, step):
        _require(type(step) is int and step == self.updates + 1, "EMA updates must be contiguous")
        self._validate_model(model)
        for name, parameter in model.named_parameters():
            if self.decay == 0:
                self.parameters[name].copy_(parameter)
            else:
                self.parameters[name].mul_(self.decay).add_(parameter.detach(), alpha=1 - self.decay)
        self.raw_state_sha256 = state_sha256(model.state_dict())
        self.updates = step

    def state_dict(self, model):
        self._validate_model(model)
        _require(state_sha256(model.state_dict()) == self.raw_state_sha256,
                 "Raw model changed since the last EMA update")
        parameters, buffers = _cpu_tree(self.parameters), _cpu_tree(self.buffers)
        return {"schema": EMA_SCHEMA, "decay": self.decay, "updates": self.updates,
                "base_state_sha256": self.base_state_sha256, "raw_state_sha256": self.raw_state_sha256,
                "architecture": copy.deepcopy(self.architecture), "parameters": parameters, "buffers": buffers,
                "parameter_names": list(parameters), "ema_parameters_sha256": state_sha256(parameters),
                "fixed_buffers_sha256": state_sha256(buffers)}

    @classmethod
    def from_state_dict(cls, model, payload, *, expected_step, decay=None, base_state_sha256=None):
        parameters, buffers = dict(model.named_parameters()), dict(model.named_buffers())
        _require(payload["schema"] == EMA_SCHEMA and payload["architecture"] == model.architecture_metadata
                 and payload["parameter_names"] == list(parameters) and payload["updates"] == expected_step
                 and type(expected_step) is int and expected_step >= 0
                 and type(payload["decay"]) is float and math.isfinite(payload["decay"])
                 and 0 <= payload["decay"] < 1
                 and (decay is None or payload["decay"] == decay)
                 and (base_state_sha256 is None or payload["base_state_sha256"] == base_state_sha256),
                 "EMA endpoint or policy differs")
        _validate_tensors(payload["parameters"], parameters)
        _validate_tensors(payload["buffers"], buffers)
        _require(payload["raw_state_sha256"] == state_sha256(model.state_dict())
                 and payload["ema_parameters_sha256"] == state_sha256(payload["parameters"])
                 and payload["fixed_buffers_sha256"] == state_sha256(payload["buffers"]) == state_sha256(buffers),
                 "EMA tensor fingerprints differ")
        result = cls.__new__(cls)
        result.decay, result.updates = payload["decay"], expected_step
        result.base_state_sha256 = payload["base_state_sha256"]
        result.raw_state_sha256 = payload["raw_state_sha256"]
        result.architecture = copy.deepcopy(payload["architecture"])
        result.parameters = {name: p.detach().to(parameters[name].device).clone()
                             for name, p in payload["parameters"].items()}
        result.buffers = {name: b.detach().to(buffers[name].device).clone() for name, b in payload["buffers"].items()}
        result._validate_model(model)
        return result

    def inference_copy(self, model):
        state = self.state_dict(model)
        result = copy.deepcopy(model).eval().requires_grad_(False)
        result.load_state_dict({**state["parameters"], **state["buffers"]}, strict=True)
        result.provenance = {**result.provenance, "checkpoint_weight_role": "averaged_inference",
            "weight_averaging": {key: state[key] for key in
                ("schema", "decay", "updates", "base_state_sha256", "raw_state_sha256", "ema_parameters_sha256")},
            "quality_measured": False}
        return result


def _new_model(architecture):
    # Constructing a loader must not consume the training RNG being restored.
    with torch.random.fork_rng(devices=[]), torch.device("cpu"):
        if architecture.get("version") == COMPACT_VERSION:
            model = CompactSeparator(sources=architecture["source_order"],
                hidden_size=architecture["hidden_size"], layers=architecture["layers"])
        else:
            model = StemgenRT58()
    _require(architecture == model.architecture_metadata, "Checkpoint architecture differs")
    return model


def _read(path, digest=None):
    path = Path(path)
    _require(path.is_file() and not path.is_symlink(), "Checkpoint must be a regular file")
    if digest is not None:
        _require(file_sha256(path) == digest, "Checkpoint SHA-256 differs")
    return torch.load(path, map_location="cpu", weights_only=True)


def _native_payload_model(payload):
    _require(payload["schema"] == NATIVE_SCHEMA
             and type(payload["step"]) is int and payload["step"] > 0,
             "Unsupported native checkpoint schema")
    model = _new_model(payload["architecture"])
    _require(type(model) is StemgenRT58, "Native historical checkpoints require the released architecture")
    _require(payload["architecture"] == model.architecture_metadata
             and payload["parameter_names"] == [name for name, _ in model.named_parameters()],
             "Native checkpoint architecture differs")
    _validate_tensors(payload["model"], model.state_dict())
    _require(state_sha256(payload["model"]) == payload["model_state_sha256"], "Native tensor fingerprint differs")
    model.load_state_dict(payload["model"], strict=True)
    provenance = payload["provenance"]
    prefix = "branch_memory"
    _require(state_sha256(dict(model.named_buffers())) == payload["fixed_buffers_sha256"]
             and model.fixed_residual_share.item() == 1 / 16
             and provenance[prefix + "_version"] == VERSION
             and provenance[prefix + "_updates"] == payload["step"]
             and provenance["training_updates"] == provenance[prefix + "_parent_updates"] + payload["step"]
             and provenance[prefix + "_training_plan_sha256"] == payload["plan_sha256"]
             and provenance[prefix + "_all_neural_parameters_trained"] is True,
             "Native fixed buffers or training lineage differs")
    model.provenance = {**copy.deepcopy(provenance), "checkpoint_weight_role": _native_role(payload)}
    return model.eval().requires_grad_(False)


def _native_role(payload):
    provenance = payload["provenance"]
    declared = provenance.get("checkpoint_weight_role")
    if declared is None:
        # The first current-model native serializer preceded separate EMA files.
        # Its inference payload held the raw optimizer endpoint.
        _require("weight_averaging" not in provenance, "Native averaging role is ambiguous")
        return "raw_optimizer_endpoint"
    _require(declared in ("raw_optimizer_endpoint", "averaged_inference"), "Unknown native checkpoint weight role")
    return declared


def load_native_checkpoint(path, *, sha256, device="cpu"):
    """Load an authenticated historical native checkpoint of the current model.

    Historical packed/XOR recovery archives require their archived converter.
    An integer ONNX graph cannot reconstruct the original FP32 weights.
    """
    _require(isinstance(sha256, str) and len(sha256) == 64, "Supply the native checkpoint SHA-256")
    payload = _read(path, sha256)
    return _native_payload_model(payload).to(device), payload


def _validate_optimizer(model, optimizer, step):
    _require(type(optimizer) is torch.optim.Adam, "Only Adam checkpoints are supported")
    parameters = list(model.parameters())
    _require(len(parameters) == model.parameter_tensor_count and len(optimizer.param_groups) == 1
             and [id(p) for p in optimizer.param_groups[0]["params"]] == [id(p) for p in parameters],
             "Adam must own every model parameter in their original order")
    _require(set(optimizer.state) == (set(parameters) if step else set()), "Adam state inventory differs")
    for parameter in parameters:
        if not step:
            continue
        state = optimizer.state[parameter]
        _require(state["step"].item() == step, "Adam step differs from checkpoint step")
        for key in ("exp_avg", "exp_avg_sq"):
            value = state[key]
            _require(value.shape == parameter.shape and value.dtype == torch.float32
                     and bool(torch.isfinite(value).all()), "Invalid Adam moment")
        _require(bool((state["exp_avg_sq"] >= 0).all()), "Invalid Adam squared moment")


def _rng_state():
    numpy_state = np.random.get_state()
    return {"python": random.getstate(), "numpy": [numpy_state[0],
        torch.from_numpy(numpy_state[1].astype(np.int64)), *numpy_state[2:]],
        "cpu": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else []}


def _backend_state():
    return {"deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32}


def _restore_rng(state, backend):
    _require(len(state["cuda"]) == (torch.cuda.device_count() if state["cuda"] else 0),
             "CUDA RNG device inventory differs")
    torch.use_deterministic_algorithms(backend["deterministic_algorithms"],
                                      warn_only=backend["deterministic_warn_only"])
    torch.backends.cudnn.deterministic = backend["cudnn_deterministic"]
    torch.backends.cudnn.benchmark = backend["cudnn_benchmark"]
    torch.backends.cudnn.allow_tf32 = backend["cudnn_allow_tf32"]
    torch.backends.cuda.matmul.allow_tf32 = backend["matmul_allow_tf32"]
    numpy_state = state["numpy"]
    random.setstate(state["python"])
    np.random.set_state((numpy_state[0], numpy_state[1].numpy().astype(np.uint32), *numpy_state[2:]))
    torch.set_rng_state(state["cpu"])
    if state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])


def _validate_cursor(step, cursor, config):
    _require(type(step) is int and step >= 0 and type(cursor) is int and cursor >= 0,
             "Invalid checkpoint step or sample cursor")
    if "data_start" in config and "batch_size" in config:
        _require(cursor == config["data_start"] + step * config["batch_size"],
                 "Sample cursor differs from the completed updates")


def _atomic_save(path, payload):
    path = Path(path)
    _require(path.parent.is_dir() and not path.is_symlink(), "Checkpoint directory must exist; aliases are forbidden")
    descriptor, name = tempfile.mkstemp(prefix=path.name + ".", suffix=".pending", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        digest = file_sha256(temporary)
        os.replace(temporary, path)
        # Directory fsync makes the rename durable on POSIX filesystems.
        if os.name == "posix":
            directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        return digest
    finally:
        temporary.unlink(missing_ok=True)


def _validate_training_provenance(config, provenance):
    from .data import validate_checkpoint as validate_data_checkpoint
    validate_data_checkpoint(config, provenance)
    if ("teacher_coefficient" in config or "teacher_supervision" in config
            or "branch_memory_current_stage_teacher_supervision" in provenance
            or provenance.get("branch_memory_online_teacher_used")
            or provenance.get("branch_memory_teacher_generated_targets_in_current_stage")):
        from .teacher import validate_checkpoint
        validate_checkpoint(config, provenance)


def save_training_checkpoint(path, model, optimizer, ema, *, step, next_sample_index,
                             config, data_identity, metadata=None, compressed=True):
    """Atomically replace one checkpoint after a complete Adam/EMA update."""
    _require(type(model) in (StemgenRT58, CompactSeparator) and isinstance(config, dict)
             and type(ema) is ParameterEMA and ema.updates == step, "Invalid training checkpoint objects")
    _validate_training_provenance(config, model.provenance)
    _validate_geometry_config(config, model)
    _validate_cursor(step, next_sample_index, config)
    _validate_optimizer(model, optimizer, step)
    state = _cpu_tree(model.state_dict())
    _validate_tensors(state, model.state_dict())
    schema = COMPACT_SCHEMA if type(model) is CompactSeparator else SCHEMA
    provenance = _raw_provenance(model.provenance)
    if type(model) is CompactSeparator:
        parent_updates = provenance.get("parent_training_updates", 0)
        _require(type(parent_updates) is int and parent_updates >= 0, "Invalid compact parent update count")
        provenance.update(current_stage_updates=step, training_updates=parent_updates + step)
    payload = {"schema": schema, "architecture": copy.deepcopy(model.architecture_metadata),
        "parameter_names": [name for name, _ in model.named_parameters()], "model": state,
        "model_state_sha256": state_sha256(state), "provenance": provenance,
        "optimizer": _cpu_tree(optimizer.state_dict()), "ema": ema.state_dict(model),
        "step": step, "next_sample_index": next_sample_index, "config": copy.deepcopy(config),
        "config_sha256": _json_sha(config), "data_identity": copy.deepcopy(data_identity),
        "data_identity_sha256": _json_sha(data_identity), "metadata": copy.deepcopy(metadata or {}),
        "torch_version": str(torch.__version__), "device_type": next(model.parameters()).device.type,
        "precision": model.training_precision, "backend": _backend_state(), "rng": _rng_state()}
    _json_sha(payload["metadata"])
    _require(sys.byteorder == "little", "Checkpoint byte-plane encoding requires little-endian tensors")
    envelope = {"schema": payload["schema"], "codec": CODEC if compressed else None,
                "payload": pack(payload) if compressed else payload}
    digest = _atomic_save(path, envelope)
    return {"path": str(Path(path)), "sha256": digest, "step": step}


def _training_payload(envelope):
    _require(isinstance(envelope, dict) and envelope.get("schema") in (SCHEMA, COMPACT_SCHEMA)
             and envelope.get("codec") in (None, CODEC), "Unsupported training checkpoint schema or codec")
    _require(sys.byteorder == "little", "Checkpoint decoding requires little-endian tensors")
    payload = unpack(envelope["payload"]) if envelope["codec"] else envelope["payload"]
    _require(payload["schema"] == envelope["schema"] and payload["config_sha256"] == _json_sha(payload["config"])
             and payload["data_identity_sha256"] == _json_sha(payload["data_identity"]),
             "Checkpoint configuration or dataset fingerprint differs")
    _validate_training_provenance(payload["config"], payload["provenance"])
    _validate_cursor(payload["step"], payload["next_sample_index"], payload["config"])
    if payload["schema"] == COMPACT_SCHEMA:
        provenance = payload["provenance"]
        parent = provenance.get("parent_training_updates", 0)
        _require(type(parent) is int and parent >= 0
                 and provenance.get("current_stage_updates") == payload["step"]
                 and provenance.get("training_updates") == parent + payload["step"],
                 "Compact training history differs from checkpoint endpoint")
    return payload


def _training_model(payload):
    model = _new_model(payload["architecture"])
    expected_schema = COMPACT_SCHEMA if type(model) is CompactSeparator else SCHEMA
    _require(payload["schema"] == expected_schema, "Training schema and architecture differ")
    _validate_geometry_config(payload["config"], model)
    _require(payload["architecture"] == model.architecture_metadata
             and payload["parameter_names"] == [name for name, _ in model.named_parameters()],
             "Training checkpoint architecture differs")
    _validate_tensors(payload["model"], model.state_dict())
    _require(state_sha256(payload["model"]) == payload["model_state_sha256"],
             "Training checkpoint tensor fingerprint differs")
    model.load_state_dict(payload["model"], strict=True)
    model.provenance = _raw_provenance(payload["provenance"])
    return model


@dataclass
class TrainingState:
    model: StemgenRT58 | CompactSeparator
    optimizer: torch.optim.Adam
    ema: ParameterEMA
    step: int
    next_sample_index: int
    config: dict
    data_identity: object
    metadata: dict


def load_training_checkpoint(path, *, sha256=None, config=None, data_identity=None,
                             device="cpu", precision=None, restore_rng=True):
    """Restore raw weights and optimizer before restoring the saved RNG streams.

    Exact continuation requires the same PyTorch build, device type, precision,
    dataset and configuration. Pass their expected identities to reject drift.
    Use ``load_model`` for inference across devices without changing global RNG.
    """
    payload = _training_payload(_read(path, sha256))
    _require(config is None or _json_sha(config) == payload["config_sha256"], "Training configuration changed")
    _require(data_identity is None or _json_sha(data_identity) == payload["data_identity_sha256"],
             "Training data identity changed")
    precision = precision or payload["precision"]
    target = torch.device(device)
    if restore_rng:
        _require(payload["torch_version"] == str(torch.__version__) and payload["device_type"] == target.type
                 and payload["precision"] == precision, "Exact resume requires the original runtime and precision")
    _require(precision in ("fp32", "bf16") and (precision != "bf16" or target.type == "cuda"),
             "Unsupported training precision on the selected device")
    model = _training_model(payload).to(target).train().requires_grad_(True)
    model.training_precision = precision
    groups = payload["optimizer"]["param_groups"]
    _require(len(groups) == 1 and groups[0]["params"] == list(range(model.parameter_tensor_count)), "Saved Adam parameter order differs")
    optimizer = torch.optim.Adam(model.parameters(), lr=groups[0]["lr"], foreach=False)
    optimizer.load_state_dict(payload["optimizer"])
    _validate_optimizer(model, optimizer, payload["step"])
    ema = ParameterEMA.from_state_dict(model, payload["ema"], expected_step=payload["step"])
    if restore_rng:
        _restore_rng(payload["rng"], payload["backend"])
    return TrainingState(model, optimizer, ema, payload["step"], payload["next_sample_index"],
                         payload["config"], payload["data_identity"], payload["metadata"])


def load_model(path, *, expected_sha256, role=None, device="cpu"):
    """Load a verified native inference file or a portable training snapshot.

    ``role`` selects raw or EMA weights from portable training snapshots
    (default: EMA). Native files already contain their declared role; an
    explicit role must match that provenance.
    """
    _require(isinstance(expected_sha256, str) and len(expected_sha256) == 64,
             "Supply the checkpoint SHA-256")
    _require(role in (None, "raw", "ema"), "Select the raw or EMA inference role")
    envelope = _read(path, expected_sha256)
    if envelope.get("schema") == NATIVE_SCHEMA:
        if role is not None:
            _require(_native_role(envelope) ==
                     {"raw": "raw_optimizer_endpoint", "ema": "averaged_inference"}[role],
                     "Requested weight role differs from the native checkpoint")
        return _native_payload_model(envelope).to(device)
    role = role or "ema"
    payload = _training_payload(envelope)
    model = _training_model(payload)
    ema = ParameterEMA.from_state_dict(model, payload["ema"], expected_step=payload["step"])
    if role == "ema":
        model = ema.inference_copy(model)
    else:
        model.provenance = {**model.provenance, "checkpoint_weight_role": "raw_optimizer_endpoint"}
    return model.to(device).eval().requires_grad_(False)
