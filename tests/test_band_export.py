"""Complete one-hop DSP and recurrent-state equivalence in ONNX Runtime."""
import json

import numpy as np
import pytest
import torch

from stemgenrt.banded import BandSeparator, SOURCE_ORDER
from stemgenrt.band_export import export_fp32, make_export_copy
from stemgenrt.checkpoint import state_sha256


@pytest.mark.parametrize("sources", [("vocals",), SOURCE_ORDER])
def test_full_fp32_onnx_continuous_audio_all_states_and_source_preservation(tmp_path, sources):
    ort = pytest.importorskip("onnxruntime")
    onnx = pytest.importorskip("onnx")
    torch.manual_seed(42)
    model = BandSeparator(sources=sources).train()
    # Exercise masks beyond their near-constant initialization.
    with torch.no_grad():
        for head in model.backbone.mask_heads:
            head.weight.normal_(std=.003)
    fingerprint, rng = state_sha256(model.state_dict()), torch.get_rng_state()
    flags = [p.requires_grad for p in model.parameters()]
    path = tmp_path / "band.onnx"
    report = export_fp32(model, path)
    assert model.training and flags == [p.requires_grad for p in model.parameters()]
    assert state_sha256(model.state_dict()) == fingerprint and torch.equal(torch.get_rng_state(), rng)
    graph = onnx.load(str(path))
    assert sum(n.op_type == "DFT" for n in graph.graph.node) == 2
    assert not any(n.op_type in ("GRU", "Loop", "Scan", "Softmax") for n in graph.graph.node)
    metadata = json.loads({m.key: m.value for m in graph.metadata_props}["stemgenrt.experimental"])
    assert metadata["architecture"]["source_order"] == list(sources)
    assert metadata["native_host_qualified"] is False
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
    assert [i.name for i in session.get_inputs()] == report["interface"]["input_names"]
    assert [i.shape for i in session.get_inputs()] == report["interface"]["input_shapes"]
    assert [o.name for o in session.get_outputs()] == report["interface"]["output_names"]
    assert [o.shape for o in session.get_outputs()] == report["interface"]["output_shapes"]
    audio = torch.randn(1, 2, 96 * 128) * .07
    audio[..., :512] = 0
    audio[..., 512:1024] = 0
    audio[..., 513] = 1
    audio[..., 1024:4096] += .02  # DC plus noise.
    audio[..., -1024:] = 0
    state = model.initial_state(1)
    runtime = [s.numpy().copy() for s in state]
    with torch.no_grad():
        for chunk in audio.split(128, -1):
            output, state = model.forward_chunk(chunk, state)
            actual = session.run(None, dict(zip(report["interface"]["input_names"], [chunk.numpy(), *runtime])))
            np.testing.assert_allclose(actual[0], output.numpy(), rtol=2e-4, atol=1e-5)
            assert np.isfinite(actual[0]).all()
            for found, expected in zip(actual[1:], state):
                np.testing.assert_allclose(found, expected.numpy(), rtol=2e-4, atol=5e-4)
                assert np.isfinite(found).all()
            runtime = actual[1:]
    with pytest.raises(ValueError, match="new path"):
        export_fp32(model, path)


def test_export_rejects_nonfinite_weights():
    model = BandSeparator(band_width=16, global_width=16, layers=1)
    with torch.no_grad():
        next(model.parameters()).flatten()[0] = float("nan")
    with pytest.raises(ValueError, match="finite FP32"):
        make_export_copy(model)
