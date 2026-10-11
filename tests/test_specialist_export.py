"""Complete ONNX FFT, recurrent-state and waveform parity for both specialists."""
import json

import numpy as np
import pytest
import torch

from stemgenrt.specialist import SpecialistSeparator
from stemgenrt.specialist_export import export_fp32, make_export_copy
from stemgenrt.checkpoint import state_sha256


@pytest.mark.parametrize("source", ["bass", "drums"])
def test_specialist_full_onnx_stream_matches_every_state_without_mutating_source(tmp_path, source):
    ort = pytest.importorskip("onnxruntime")
    onnx = pytest.importorskip("onnx")
    torch.manual_seed(842)
    model = SpecialistSeparator(source=source).train()
    with torch.no_grad():
        for head in model.backbone.mask_heads:
            head.weight.normal_(std=.003)
        if source == "drums":
            model.waveform_decode.weight.mul_(25)
    fingerprint, rng = state_sha256(model.state_dict()), torch.get_rng_state()
    flags = [p.requires_grad for p in model.parameters()]
    path = tmp_path / (source + ".onnx")
    report = export_fp32(model, path)
    assert model.training and flags == [p.requires_grad for p in model.parameters()]
    assert state_sha256(model.state_dict()) == fingerprint and torch.equal(torch.get_rng_state(), rng)
    graph = onnx.load(str(path))
    assert sum(n.op_type == "DFT" for n in graph.graph.node) == (3 if source == "bass" else 2)
    assert not any(n.op_type in ("GRU", "Loop", "Scan", "Softmax") for n in graph.graph.node)
    metadata = json.loads({m.key: m.value for m in graph.metadata_props}["stemgenrt.experimental"])
    assert metadata["architecture"]["source_order"] == [source]
    assert metadata["native_host_qualified"] is False
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
    assert [i.name for i in session.get_inputs()] == report["interface"]["input_names"]
    assert [i.shape for i in session.get_inputs()] == report["interface"]["input_shapes"]
    assert [o.name for o in session.get_outputs()] == report["interface"]["output_names"]
    assert [o.shape for o in session.get_outputs()] == report["interface"]["output_shapes"]
    audio = torch.randn(1, 2, 128 * 128) * .07
    audio[..., :1024] = 0
    audio[..., 1025] = 1
    audio[..., 2048:4096] += .02
    audio[..., -4096:] = 0
    state = model.initial_state(1)
    runtime = [s.numpy().copy() for s in state]
    with torch.no_grad():
        for chunk in audio.split(128, -1):
            expected, state = model.forward_chunk(chunk, state)
            actual = session.run(None, dict(zip(report["interface"]["input_names"], [chunk.numpy(), *runtime])))
            np.testing.assert_allclose(actual[0], expected.numpy(), rtol=2e-4, atol=1e-5)
            assert np.isfinite(actual[0]).all()
            for found, reference in zip(actual[1:], state):
                np.testing.assert_allclose(found, reference.numpy(), rtol=2e-4, atol=5e-4)
                assert np.isfinite(found).all()
            runtime = actual[1:]
    with pytest.raises(ValueError, match="new path"):
        export_fp32(model, path)


def test_specialist_export_rejects_nonfinite_weights():
    model = SpecialistSeparator(source="bass", band_width=16, global_width=16, layers=1)
    with torch.no_grad():
        next(model.parameters()).flatten()[0] = float("nan")
    with pytest.raises(ValueError, match="finite FP32"):
        make_export_copy(model)
