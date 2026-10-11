"""Checkpoint contracts exercised with the maintained streaming model."""
from __future__ import annotations

import gc
import hashlib
import random

import numpy as np
import pytest
import torch

from stemgenrt import checkpoint
from stemgenrt.checkpoint import (
    ParameterEMA, file_sha256, load_model, load_native_checkpoint,
    load_training_checkpoint, save_training_checkpoint, state_sha256,
)
from stemgenrt.model import StemgenRT58, VERSION, render_scored_context


def tree_fingerprint(value):
    digest = hashlib.sha256()
    def visit(item):
        if isinstance(item, torch.Tensor):
            tensor = item.detach().cpu().contiguous()
            digest.update(str((tensor.dtype, tuple(tensor.shape))).encode())
            digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item, dict):
            for key, child in item.items():
                digest.update(str(key).encode())
                visit(child)
        elif isinstance(item, (list, tuple)):
            for child in item:
                visit(child)
        else:
            digest.update(repr(item).encode())
    visit(value)
    return digest.hexdigest()


def model():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(617)
        result = StemgenRT58().train()
        # Exercise the initially zero branch output projections as well.
        with torch.no_grad():
            for parameter in result.parameters():
                if torch.count_nonzero(parameter) == 0:
                    parameter.normal_(std=.001)
    result.training_precision = "fp32"
    return result


def advance(candidate, optimizer, ema, step):
    audio = torch.randn(1, 2, 384) * (.02 + .001 * random.random() + .001 * np.random.random())
    target = torch.randn(1, 4, 2, 256) * .01
    optimizer.zero_grad(set_to_none=True)
    output = render_scored_context(candidate, audio, warmup_samples=128, carry_state=True)
    loss = (output.raw - target).square().mean()
    loss.backward()
    assert all(parameter.grad is not None for parameter in candidate.parameters())
    torch.nn.utils.clip_grad_norm_(candidate.parameters(), 5., foreach=False, error_if_nonfinite=True)
    optimizer.step()
    ema.update(candidate, step=step)
    return float(loss.detach())


@pytest.mark.parametrize("with_teacher", [False, True])
def test_exact_next_update_after_portable_packed_restore(tmp_path, with_teacher):
    torch.set_num_threads(1)
    random.seed(192); np.random.seed(192); torch.manual_seed(192)
    candidate = model()
    optimizer = torch.optim.Adam(candidate.parameters(), lr=3e-5, foreach=False)
    ema = ParameterEMA(candidate)
    config = {"steps": 3, "batch_size": 1, "data_start": 400, "lr": 3e-5,
              "precision": "fp32", "seed": 192}
    if with_teacher:
        from stemgenrt import teacher
        specification = teacher.specification(1.)
        teacher.attach(candidate, specification)
        config.update(teacher_coefficient=1., teacher_supervision=specification, target_source="vocals")
        invalid = {**config, 'teacher_coefficient': .5}
        with pytest.raises(ValueError, match='teacher identity'):
            save_training_checkpoint(tmp_path / 'invalid.pt', candidate, optimizer, ema,
                step=0, next_sample_index=400, config=invalid, data_identity={})
        assert not (tmp_path / 'invalid.pt').exists()
    data = {"manifest_sha256": "a" * 64, "validation_manifest_sha256": "b" * 64}
    advance(candidate, optimizer, ema, 1)
    saved_optimizer = tree_fingerprint(optimizer.state_dict())
    saved_ema = tree_fingerprint(ema.state_dict(candidate))
    saved_raw = state_sha256(candidate.state_dict())
    rng = tree_fingerprint(checkpoint._rng_state())
    path = tmp_path / "checkpoint.pt"
    bound = save_training_checkpoint(path, candidate, optimizer, ema, step=1, next_sample_index=401,
                                     config=config, data_identity=data, metadata={"note": "test"})
    assert tree_fingerprint(checkpoint._rng_state()) == rng
    assert bound["sha256"] == file_sha256(path)
    expected_loss = advance(candidate, optimizer, ema, 2)
    expected_raw = state_sha256(candidate.state_dict())
    expected_optimizer = tree_fingerprint(optimizer.state_dict())
    expected_ema = tree_fingerprint(ema.state_dict(candidate))
    expected_rng = tree_fingerprint(checkpoint._rng_state())
    del candidate, optimizer, ema
    gc.collect()

    with pytest.raises(ValueError, match="data identity"):
        load_training_checkpoint(path, sha256=bound["sha256"], config=config,
                                 data_identity={**data, "manifest_sha256": "c" * 64})
    with pytest.raises(ValueError, match="configuration"):
        load_training_checkpoint(path, config={**config, "lr": 1e-3}, data_identity=data)
    with pytest.raises(ValueError, match="configuration"):
        load_training_checkpoint(path, config={**config, "target_source": "bass"}, data_identity=data)
    with pytest.raises(ValueError, match="configuration"):
        load_training_checkpoint(path, config={**config, "track_sampling":
            "duration"}, data_identity=data)
    with pytest.raises(ValueError, match="configuration"):
        load_training_checkpoint(path, config={**config, "extra_ordinary_primary_sdr_weight": .2},
                                 data_identity=data)
    restored = load_training_checkpoint(path, sha256=bound["sha256"], config=config, data_identity=data)
    assert restored.step == 1 and restored.next_sample_index == 401 and restored.metadata == {"note": "test"}
    assert restored.model.parameter_tensor_count == 40
    assert restored.model.attention_window == 32
    assert state_sha256(restored.model.state_dict()) == saved_raw
    assert tree_fingerprint(restored.optimizer.state_dict()) == saved_optimizer
    assert tree_fingerprint(restored.ema.state_dict(restored.model)) == saved_ema
    assert advance(restored.model, restored.optimizer, restored.ema, 2) == expected_loss
    assert state_sha256(restored.model.state_dict()) == expected_raw
    assert tree_fingerprint(restored.optimizer.state_dict()) == expected_optimizer
    assert tree_fingerprint(restored.ema.state_dict(restored.model)) == expected_ema
    assert tree_fingerprint(checkpoint._rng_state()) == expected_rng
    del restored
    gc.collect()

    inference_rng = tree_fingerprint(checkpoint._rng_state())
    raw = load_model(path, expected_sha256=bound["sha256"], role="raw")
    assert state_sha256(raw.state_dict()) == saved_raw
    if with_teacher:
        assert raw.provenance[teacher.PROVENANCE_KEY] == specification
    assert not raw.training and not any(p.requires_grad for p in raw.parameters())
    del raw
    averaged = load_model(path, expected_sha256=bound["sha256"], role="ema")
    assert averaged.provenance["checkpoint_weight_role"] == "averaged_inference"
    if with_teacher:
        assert averaged.provenance[teacher.PROVENANCE_KEY] == specification
    assert tree_fingerprint(checkpoint._rng_state()) == inference_rng
    assert not torch.cuda.is_initialized()
    path.unlink()


def test_native_checkpoint_authentication_and_role(tmp_path):
    torch.set_num_threads(1)
    candidate = model()
    plan_sha = "d" * 64
    prefix = "branch_memory"
    provenance = {**candidate.provenance, prefix + "_version": VERSION,
        prefix + "_updates": 2, prefix + "_parent_updates": 10, "training_updates": 12,
        prefix + "_training_plan_sha256": plan_sha, prefix + "_all_neural_parameters_trained": True}
    payload = {"schema": checkpoint.NATIVE_SCHEMA,
        "step": 2, "model": candidate.state_dict(),
        "model_state_sha256": state_sha256(candidate.state_dict()), "architecture": candidate.architecture_metadata,
        "fixed_buffers_sha256": state_sha256(dict(candidate.named_buffers())),
        "parameter_names": [name for name, _ in candidate.named_parameters()], "provenance": provenance,
        "plan_sha256": plan_sha}
    path = tmp_path / "native.pt"
    torch.save(payload, path)
    digest = file_sha256(path)
    expected = payload["model_state_sha256"]
    del candidate, payload
    restored, metadata = load_native_checkpoint(path, sha256=digest)
    assert state_sha256(restored.state_dict()) == expected == metadata["model_state_sha256"]
    assert restored.provenance["checkpoint_weight_role"] == "raw_optimizer_endpoint"
    assert restored.attention_window == 32
    del restored, metadata
    with pytest.raises(ValueError, match="SHA-256"):
        load_native_checkpoint(path, sha256="0" * 64)
    with pytest.raises(ValueError, match="weight role"):
        load_model(path, expected_sha256=digest, role="ema")
    payload = torch.load(path, weights_only=True)
    for schema in ("stemgenrt-attention128-inference-v1", "stemgenrt-shared-mask-inference-v1"):
        payload["schema"] = schema
        with pytest.raises(ValueError, match="Unsupported native checkpoint schema"):
            checkpoint._native_payload_model(payload)
    path.unlink()


def test_atomic_failure_preserves_previous_checkpoint(tmp_path, monkeypatch):
    path = tmp_path / "checkpoint.pt"
    path.write_bytes(b"previous checkpoint")
    def fail_replace(*args):
        raise OSError("simulated publication failure")
    monkeypatch.setattr(checkpoint.os, "replace", fail_replace)
    with pytest.raises(OSError, match="publication failure"):
        checkpoint._atomic_save(path, {"tensor": torch.ones(4)})
    assert path.read_bytes() == b"previous checkpoint"
    assert list(tmp_path.iterdir()) == [path]


def test_raw_continuation_retains_parent_averaging_as_history():
    source = {"checkpoint_weight_role": "averaged_inference", "weight_averaging": {"updates": 2000}}
    raw = checkpoint._raw_provenance(source)
    assert raw["checkpoint_weight_role"] == "raw_optimizer_endpoint"
    assert raw["initial_parent_weight_averaging"] == source["weight_averaging"]
    assert "weight_averaging" not in raw and "weight_averaging" in source


def test_codec_roundtrip_bits_and_corruption():
    from stemgenrt._checkpoint.codec import pack, unpack
    source = {"values": torch.tensor([0., -0., float("nan"), float("inf")]),
              "indices": torch.tensor([-(2**62), 2**62], dtype=torch.int64),
              "empty": torch.empty(0), "scalar": torch.tensor(1.),
              "rng": torch.arange(255, dtype=torch.uint8)}
    packed = pack(source)
    assert tree_fingerprint(unpack(packed)) == tree_fingerprint(source)
    packed["values"]["data"] = torch.cat((packed["values"]["data"], torch.tensor([1], dtype=torch.uint8)))
    with pytest.raises(ValueError, match="byte length"):
        unpack(packed)
