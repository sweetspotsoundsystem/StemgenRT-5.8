"""Portable training for the current streaming model and weighted objective.

Start a new run or resume a complete training checkpoint with its saved recipe.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import json
import math
import os
from pathlib import Path
import random
import signal
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from .checkpoint import ParameterEMA, load_model, load_training_checkpoint, save_training_checkpoint, state_sha256
from .data import (AbsoluteIndexSampler, CROP_SAMPLES, WARMUP_SAMPLES, audio_sha,
                   batch_recipes, load_manifest, make_dataset, remix_batch, worker_init, policy as data_policy)
from .losses import grouped_update, _teacher_weight, source_index
from .model import StemgenRT58
from .compact import CompactSeparator, SOURCE_ORDER


@dataclass(frozen=True)
class TrainingConfig:
    model_family: str = "released"
    compact_hidden_size: int = 384
    compact_layers: int = 3
    target_source: str | None = None
    attention_window: int = 32
    past_filter: bool = False
    steps: int = 2000
    batch_size: int = 16
    microbatch_size: int = 16
    auxiliary_microbatch_size: int = 2
    crop_samples: int = 264576
    warmup_samples: int = 88064
    lr: float = 3e-5
    min_lr: float = 3e-6
    warmup: int = 100
    checkpoint_every: int = 50
    seed: int = 20261102
    data_seed: int = 60
    data_start: int = 0
    track_sampling: str = "uniform"
    root_weights: dict[str, float] | None = None
    vocal_active_probability: float = .85
    ema_decay: float = .995
    precision: str = "bf16"
    device: str = "cuda"
    workers: int = 2
    extra_ordinary_primary_sdr_weight: float = .2
    teacher_coefficient: float = 0.
    teacher_checkpoint: str | None = None

    def validate(self):
        source_index(self.target_source)
        if self.model_family not in ("released", "compact"):
            raise ValueError("Unknown model_family")
        if (type(self.compact_hidden_size) is not int or not 16 <= self.compact_hidden_size <= 1024
                or type(self.compact_layers) is not int or not 1 <= self.compact_layers <= 4):
            raise ValueError("Invalid compact backbone geometry")
        if self.model_family == "released" and (self.compact_hidden_size != 384 or self.compact_layers != 3):
            raise ValueError("Compact geometry cannot configure the released model")
        if self.model_family == "compact" and (self.target_source not in (None, "vocals")
                                               or self.teacher_coefficient != 0.):
            raise ValueError("Compact experiments support joint or vocal supervised training without an online teacher")
        if self.past_filter is not False:
            raise ValueError("The current model does not use a past filter")
        if type(self.attention_window) is not int or self.attention_window != 32:
            raise ValueError("attention_window must be 32")
        if self.track_sampling != "uniform":
            raise ValueError("track_sampling must be uniform")
        _teacher_weight(self.teacher_coefficient)
        if self.teacher_coefficient and (not self.teacher_checkpoint
                or self.extra_ordinary_primary_sdr_weight != .2):
            raise ValueError("Teacher training requires its checkpoint and extra_ordinary_primary_sdr_weight=0.2")
        if (type(self.extra_ordinary_primary_sdr_weight) not in (float, int)
                or self.extra_ordinary_primary_sdr_weight not in (0., .2)):
            raise ValueError("extra_ordinary_primary_sdr_weight must be 0 or 0.2")
        for name in ("steps", "batch_size", "microbatch_size", "auxiliary_microbatch_size", "checkpoint_every"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(name + " must be a positive integer")
        for name in ("seed", "data_seed", "data_start", "warmup", "workers"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(name + " must be a nonnegative integer")
        if (self.batch_size != 16 or self.microbatch_size > 16 or self.auxiliary_microbatch_size > 2
                or self.crop_samples != CROP_SAMPLES or self.warmup_samples != WARMUP_SAMPLES):
            raise ValueError("Use the current B16 ordinary / B2 auxiliary, six-second context geometry")
        if self.data_start % self.batch_size:
            raise ValueError("data_start must align with the 16-example remix group")
        if self.warmup >= self.steps:
            raise ValueError("Learning-rate warmup must end before the configured horizon")
        for name in ("lr", "min_lr", "vocal_active_probability", "ema_decay"):
            value = getattr(self, name)
            if type(value) not in (float, int) or not math.isfinite(value):
                raise ValueError(name + " must be finite")
        if not (0 < self.min_lr <= self.lr and 0 <= self.vocal_active_probability <= 1 and 0 <= self.ema_decay < 1):
            raise ValueError("Invalid learning-rate, activity or EMA configuration")
        device = torch.device(self.device)
        if device.type not in ("cpu", "cuda") or self.precision not in ("fp32", "bf16"):
            raise ValueError("Use CPU FP32 or CUDA FP32/BF16")
        if device.type == "cpu" and self.precision != "fp32":
            raise ValueError("CPU training requires precision=fp32")
        if self.root_weights is not None and (not self.root_weights or any(
                not isinstance(k, str) or type(v) not in (float, int) or not math.isfinite(v) or v <= 0
                for k, v in self.root_weights.items())):
            raise ValueError("Root weights must be positive finite numbers")
        return self


def learning_rate(step, config):
    """Original warmup/cosine schedule indexed by completed updates."""
    if not 0 <= step < config.steps:
        raise ValueError("Step lies outside the configured schedule")
    if step < config.warmup:
        return config.lr * (step + 1) / config.warmup
    phase = (step - config.warmup) / max(1, config.steps - 1 - config.warmup)
    return config.min_lr + .5 * (config.lr - config.min_lr) * (1 + math.cos(math.pi * phase))


def configure_determinism(config):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(config.seed)
    np.random.seed(config.seed % 2**32)
    torch.manual_seed(config.seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    device = torch.device(config.device)
    if device.type == "cuda":
        if os.environ["CUBLAS_WORKSPACE_CONFIG"] not in (":4096:8", ":16:8"):
            raise ValueError("Set CUBLAS_WORKSPACE_CONFIG=:4096:8 before CUDA initialization")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable")
        device = torch.device("cuda", torch.cuda.current_device() if device.index is None else device.index)
        torch.cuda.set_device(device)
        if config.precision == "bf16" and not torch.cuda.is_bf16_supported():
            raise RuntimeError("The selected CUDA device does not support BF16")
    return device


def _json(path, value):
    with Path(path).open("x") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def train(config, manifest, output, *, checkpoint=None, resume=None, sha256=None,
          role="ema", stop_after=None, validation_manifest=None,
          replay_journal=None, replay_sha256=None):
    """Start or exactly resume a finite run into a new output directory.

    Initializing from a checkpoint starts fresh Adam/EMA; --resume restores all
    optimizer, average, RNG and absolute crop-address state. A signal finishes
    the current optimizer update before saving a resumable endpoint.
    """
    config.validate()
    if checkpoint is not None and resume is not None:
        raise ValueError("Choose one checkpoint initialization or exact resume")
    if (checkpoint is not None or resume is not None) and sha256 is None:
        raise ValueError("Supply the checkpoint's expected SHA-256")
    if (replay_journal is None) != (replay_sha256 is None) or (replay_journal is not None and resume is None):
        raise ValueError("Replay verification requires --resume, --replay-journal and --replay-sha256")
    if role not in ("raw", "ema"):
        raise ValueError("Checkpoint role must be raw or ema")
    stop = config.steps if stop_after is None else stop_after
    if type(stop) is not int or not 0 < stop <= config.steps:
        raise ValueError("stop_after must lie within the original schedule")
    output = Path(output)
    if output.exists():
        raise FileExistsError("Use a new output directory; preserve previous runs: " + str(output))
    corpus = load_manifest(manifest, expected_split="train")
    root_weights = config.root_weights or corpus.root_weights
    config = replace(config, root_weights=dict(root_weights)).validate()
    config_dict = asdict(config)
    if config.model_family == "released":
        for key in ("model_family", "compact_hidden_size", "compact_layers"):
            config_dict.pop(key)
    if config.target_source is None:
        config_dict.pop("target_source")
    # Preserve the exact configuration identity of existing baseline checkpoints.
    if not config.past_filter:
        config_dict.pop("past_filter")
    if config.attention_window == 32:
        config_dict.pop("attention_window")
    if config.track_sampling == "uniform":
        config_dict.pop("track_sampling")
    if config.extra_ordinary_primary_sdr_weight == 0:
        config_dict.pop("extra_ordinary_primary_sdr_weight")
    config_dict.pop("teacher_checkpoint")
    teacher_provider = None
    if config.teacher_coefficient:
        from .teacher import CPUTrainingTeacher
        teacher_provider = CPUTrainingTeacher(config.teacher_checkpoint, coefficient=config.teacher_coefficient)
        teacher_provider._load()
        config_dict["teacher_supervision"] = teacher_provider.specification
    else:
        config_dict.pop("teacher_coefficient")
    # Dict equality/canonical JSON hashes ignore insertion order, but the
    # counter-addressed sampler assigns intervals in this explicit order.
    data_identity = {"manifest_sha256": corpus.sha256,
                     "sampling_root_order": list(root_weights)}
    if validation_manifest is not None:
        from .data import require_disjoint
        validation = load_manifest(validation_manifest, expected_split="valid")
        require_disjoint(corpus.tracks, validation.tracks)
        data_identity["validation_manifest_sha256"] = validation.sha256
    device = configure_determinism(config)
    step = 0
    if resume is not None:
        restored = load_training_checkpoint(resume, sha256=sha256, config=config_dict,
            data_identity=data_identity, device=device, precision=config.precision, restore_rng=True)
        model, optimizer, ema = restored.model, restored.optimizer, restored.ema
        step = restored.step
        if restored.next_sample_index != config.data_start + step * config.batch_size:
            raise ValueError("Restored sample cursor disagrees with its schedule")
        if not 0 <= step < stop:
            raise ValueError("Resume must precede the requested stopping point")
    else:
        if checkpoint:
            model = load_model(checkpoint, expected_sha256=sha256, role=role)
            from .checkpoint import _validate_geometry_config
            _validate_geometry_config(config_dict, model)
        elif config.model_family == "compact":
            model = CompactSeparator(sources=("vocals",) if config.target_source == "vocals" else SOURCE_ORDER,
                hidden_size=config.compact_hidden_size, layers=config.compact_layers)
        else:
            model = StemgenRT58()
        model.to(device).train().requires_grad_(True)
        model.training_precision = config.precision
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, foreach=False)
        ema = ParameterEMA(model, decay=float(config.ema_decay))
        model.provenance = {**getattr(model, "provenance", {}),
                           "branch_memory_current_stage_augmentation": data_policy(config.track_sampling)}
        if teacher_provider is not None or getattr(model, "provenance", {}).get("branch_memory_current_stage_teacher_supervision"):
            from .teacher import attach
            attach(model, teacher_provider.specification if teacher_provider is not None else None)
    model.train().requires_grad_(True)
    model.training_precision = config.precision
    initial_step = step
    replay = None
    if replay_journal is not None:
        from .recovery import ReplayVerifier
        replay = ReplayVerifier(replay_journal, sha256=replay_sha256,
                                resume_step=step, schedule_steps=config.steps)
    fixed = {name: tensor.detach().clone() for name, tensor in model.named_buffers()}
    final_index = config.data_start + stop * config.batch_size
    dataset = make_dataset(corpus, config_dict, final_index)
    loader = DataLoader(dataset, batch_size=config.batch_size, num_workers=config.workers,
        sampler=AbsoluteIndexSampler(config.data_start + step * config.batch_size, final_index),
        pin_memory=device.type == "cuda", worker_init_fn=worker_init,
        generator=torch.Generator().manual_seed(config.seed + 1),
        **({"multiprocessing_context": "spawn", "prefetch_factor": 2} if config.workers else {}))
    output.mkdir(parents=True, exist_ok=False)
    _json(output / "config.json", config_dict)
    _json(output / "inputs.json", {"data": data_identity, "initialization": str(resume or checkpoint or "scratch"),
        "checkpoint_sha256": sha256, "resumed_from_step": step, "device": str(device),
        "torch_version": str(torch.__version__), "model_state_sha256": state_sha256(model.state_dict())})
    requested_stop = False
    def request_stop(signum, frame):
        nonlocal requested_stop
        requested_stop = True
    previous_handlers = {}
    # Signal handlers are only legal in the main thread; library callers may
    # own cancellation themselves. No global handlers are left behind.
    import threading
    if threading.current_thread() is threading.main_thread():
        for sig in (signal.SIGTERM, signal.SIGINT):
            previous_handlers[sig] = signal.signal(sig, request_stop)
    began = time.monotonic()
    saved = None
    def save():
        if replay is not None:
            replay.verify_source()
        return save_training_checkpoint(output / "checkpoint.pt", model, optimizer, ema,
            step=step, next_sample_index=config.data_start + step * config.batch_size,
            config=config_dict, data_identity=data_identity,
            metadata={"torch_version": str(torch.__version__), "device_type": device.type,
                      **({"replay_verification": replay.report()} if replay is not None else {})})
    try:
        with (output / "metrics.jsonl").open("x", buffering=1) as journal:
            for mixture, targets in loader:
                if requested_stop:
                    break
                first = config.data_start + step * config.batch_size
                optimizer.param_groups[0]["lr"] = learning_rate(step, config)
                before = audio_sha(mixture, targets)
                mixture, targets, _, _ = remix_batch(mixture, targets, seed=config.seed, first_sample_index=first)
                after = audio_sha(mixture, targets)
                teacher_options, teacher_metadata = {}, {}
                if teacher_provider is not None:
                    from .teacher import supervision_sha, validate_checkpoint
                    validate_checkpoint(config_dict, model.provenance)
                    teacher_targets = teacher_provider.render(mixture)
                    teacher_metadata = {
                        "teacher_supervision_sha256": supervision_sha(teacher_provider.specification),
                        "teacher_targets_sha256": state_sha256({"teacher_targets": teacher_targets})}
                    teacher_options = {"teacher_coefficient": config.teacher_coefficient,
                                       "teacher_targets": teacher_targets.to(device)}
                    del teacher_targets
                update = grouped_update(model, optimizer, ema, mixture, targets, step=step + 1,
                    warmup_samples=config.warmup_samples, ordinary_microbatch=config.microbatch_size,
                    auxiliary_microbatch=config.auxiliary_microbatch_size,
                    extra_ordinary_primary_sdr_weight=config.extra_ordinary_primary_sdr_weight,
                    target_source=config.target_source, **teacher_options)
                del teacher_options
                step += 1
                if any(not torch.equal(tensor, fixed[name]) for name, tensor in model.named_buffers()):
                    raise RuntimeError("Training modified a fixed model buffer")
                row = {**update, **teacher_metadata, "lr": optimizer.param_groups[0]["lr"], "first_sample_index": first,
                    "next_sample_index": first + config.batch_size, "pitch_tempo_recipes": batch_recipes(config_dict, first),
                    "before_remix_audio_sha256": before, "after_remix_audio_sha256": after,
                    "elapsed_seconds": time.monotonic() - began}
                if replay is not None:
                    replay.compare(row)
                journal.write(json.dumps(row, allow_nan=False) + "\n")
                journal.flush()
                os.fsync(journal.fileno())
                print(json.dumps({"step": step, "loss": update["weighted_loss"], "lr": row["lr"]}), flush=True)
                if step % config.checkpoint_every == 0 or requested_stop or step == stop:
                    saved = save()
                if requested_stop:
                    break
        if step > initial_step and (saved is None or saved["step"] != step):
            saved = save()
        result = {"status": "interrupted" if requested_stop else "completed", "step": step,
            "schedule_steps": config.steps, "resumed_from_step": initial_step,
            "next_sample_index": config.data_start + step * config.batch_size, "checkpoint": saved,
            "elapsed_seconds": time.monotonic() - began}
        if not requested_stop and step != stop:
            raise RuntimeError("Input data ended before the requested training endpoint")
        if replay is not None:
            replay.verify_source()
            result["replay_verification"] = replay.report()
        _json(output / "result.json", result)
        return result
    finally:
        for sig, previous in previous_handlers.items():
            signal.signal(sig, previous)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint", type=Path)
    source.add_argument("--resume", type=Path)
    source.add_argument("--scratch", action="store_true")
    parser.add_argument("--sha256", help="Required expected digest for --checkpoint or --resume")
    parser.add_argument("--role", choices=("raw", "ema"), default="ema", help="Weight role for fresh-Adam initialization")
    parser.add_argument("--stop-after", type=int)
    parser.add_argument("--replay-journal", type=Path, help="Verify repeated updates against a previous run's journal")
    parser.add_argument("--replay-sha256", help="Expected digest of the complete original journal file")
    args = parser.parse_args()
    torch.set_num_threads(1)
    result = train(TrainingConfig(**json.loads(args.config.read_text())), args.manifest, args.output,
        checkpoint=args.checkpoint, resume=args.resume, sha256=args.sha256, role=args.role,
        stop_after=args.stop_after, validation_manifest=args.validation_manifest,
        replay_journal=args.replay_journal, replay_sha256=args.replay_sha256)
    print(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
