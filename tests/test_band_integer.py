"""Independent INT8 reconstruction, precision boundaries and stream parity."""
import copy
import json

import numpy as np
import pytest
import torch

from stemgenrt.banded import BandSeparator, SOURCE_ORDER
from stemgenrt.band_export import export_fp32
from stemgenrt.band_integer import export_int8, make_integer_reference
from stemgenrt.checkpoint import file_sha256, state_sha256


@pytest.mark.parametrize("sources", [("vocals",), SOURCE_ORDER])
def test_integer_complete_stream_matches_independent_reference_and_preserves_source(tmp_path, sources):
    ort = pytest.importorskip("onnxruntime")
    onnx = pytest.importorskip("onnx")
    torch.manual_seed(137)
    model = BandSeparator(sources=sources, band_width=32, global_width=48, layers=2).train()
    # Strengthen masks so agreement does not depend on near-constant outputs.
    with torch.no_grad():
        for head in model.backbone.mask_heads:
            head.weight.normal_(std=.025)
    digest, rng = state_sha256(model.state_dict()), torch.get_rng_state()
    flags = [p.requires_grad for p in model.parameters()]
    fp, integer = tmp_path / "fp32.onnx", tmp_path / "int8.onnx"
    source = export_fp32(model, fp)
    report = export_int8(model, fp, integer, expected_fp32_sha256=source["onnx_sha256"])
    reference = make_integer_reference(model, integer, report)
    assert state_sha256(model.state_dict()) == digest and torch.equal(torch.get_rng_state(), rng)
    assert model.training and [p.requires_grad for p in model.parameters()] == flags
    assert len(report["converted_matrices"]) == 27
    assert all(p["zero_point_shape"] == p["scale_shape"] == [] for p in report["converted_matrices"])
    assert report["bytes"] < source["bytes"] * .4
    graph = onnx.shape_inference.infer_shapes(onnx.load(str(integer)))
    types = {v.name: v.type.tensor_type.elem_type for v in [*graph.graph.input, *graph.graph.output, *graph.graph.value_info]}
    assert sum(n.op_type == "MatMulInteger" for n in graph.graph.node) == 27
    assert not any(n.op_type in ("MatMul", "Gemm", "GRU") for n in graph.graph.node)
    assert sorted(types[n.input[0]] for n in graph.graph.node if n.op_type == "DFT") == [onnx.TensorProto.FLOAT, onnx.TensorProto.DOUBLE]
    assert all(v.type.tensor_type.elem_type == onnx.TensorProto.FLOAT for v in [*graph.graph.input, *graph.graph.output])
    metadata = json.loads({v.key: v.value for v in graph.metadata_props}["stemgenrt.experimental"])
    assert metadata["quality_measured"] is metadata["native_host_qualified"] is False
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    options.add_session_config_entry("mlas.disable_kleidiai", "1")
    session = ort.InferenceSession(str(integer), options, providers=["CPUExecutionProvider"])
    state = model.initial_state(1)
    runtime = [s.numpy().copy() for s in state]
    audio = torch.randn(1, 2, 192 * 128) * .1
    audio[..., :512] = 0
    audio[..., 1024] = 1
    audio[..., -1024:] = 0
    with torch.no_grad():
        for hop, chunk in enumerate(audio.split(128, -1)):
            expected = reference(chunk, *state)
            state = expected[1:]
            actual = session.run(None, dict(zip(report["interface"]["input_names"], [chunk.numpy(), *runtime])))
            for value, truth in zip(actual, expected):
                assert np.isfinite(value).all()
                np.testing.assert_allclose(value, truth.numpy(), atol=5e-5, rtol=2e-5)
            if hop < 4:
                assert not np.count_nonzero(actual[0])
            runtime = actual[1:]
    with pytest.raises(ValueError, match="new path"):
        export_int8(model, fp, integer, expected_fp32_sha256=source["onnx_sha256"])


def test_integer_rejects_wrong_source_and_independently_detects_changed_weight(tmp_path):
    onnx = pytest.importorskip("onnx")
    from onnx import numpy_helper as nh
    model = BandSeparator(band_width=16, global_width=16, layers=1)
    fp, integer = tmp_path / "fp32.onnx", tmp_path / "int8.onnx"
    source = export_fp32(model, fp)
    with pytest.raises(ValueError, match="digest differs"):
        export_int8(model, fp, integer, expected_fp32_sha256="0" * 64)
    wrong_model = copy.deepcopy(model)
    with torch.no_grad():
        next(wrong_model.parameters()).add_(1)
    with pytest.raises(ValueError, match="identity differs"):
        export_int8(wrong_model, fp, integer, expected_fp32_sha256=source["onnx_sha256"])
    report = export_int8(model, fp, integer, expected_fp32_sha256=source["onnx_sha256"])
    graph = onnx.load(str(integer))
    name = report["converted_matrices"][0]["initializer"] + "_quantized"
    weight = next(v for v in graph.graph.initializer if v.name == name)
    altered = nh.to_array(weight).copy()
    altered.flat[0] = 0 if altered.flat[0] else 1
    weight.CopyFrom(nh.from_array(altered, name))
    corrupted = tmp_path / "corrupted.onnx"
    onnx.save(graph, str(corrupted))
    with pytest.raises(ValueError, match="Independent weight reconstruction"):
        make_integer_reference(model, corrupted, {**report, "onnx_sha256": file_sha256(corrupted)})
