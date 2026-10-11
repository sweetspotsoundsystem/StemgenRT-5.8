"""Check complete combined ONNX state, delay, residual and source preservation."""
import json

import numpy as np
import pytest
import torch

from stemgenrt.banded import BandSeparator
from stemgenrt.specialist import SpecialistSeparator
from stemgenrt.independent import IndependentStems
from stemgenrt.independent_export import export_fp32, make_export_copy
from stemgenrt.checkpoint import state_sha256


@pytest.mark.parametrize("vocal_feature_n_fft", [1024, 4096])
def test_combined_onnx_matches_native_stream_flush_reset_and_preserves_source(tmp_path, vocal_feature_n_fft):
    ort = pytest.importorskip("onnxruntime")
    onnx = pytest.importorskip("onnx")
    torch.manual_seed(20261010)
    children = {source: SpecialistSeparator(source=source).eval() for source in ("drums", "bass")}
    children["vocals"] = BandSeparator(sources=("vocals",), feature_n_fft=vocal_feature_n_fft).eval()
    with torch.no_grad():
        for child in children.values():
            for head in child.backbone.mask_heads:
                head.weight.normal_(std=.003)
        children["drums"].waveform_decode.weight.mul_(25)
        if vocal_feature_n_fft == 4096:
            for module in children["vocals"].backbone.long_encoders:
                module.weight.normal_(std=.01)
    model = IndependentStems(**children)
    digest, rng = state_sha256(model.state_dict()), torch.get_rng_state()
    flags = [(m.training, [p.requires_grad for p in m.parameters()]) for m in model.modules()]
    path = tmp_path / "combined.onnx"
    report = export_fp32(model, path)
    assert state_sha256(model.state_dict()) == digest and torch.equal(torch.get_rng_state(), rng)
    assert flags == [(m.training, [p.requires_grad for p in m.parameters()]) for m in model.modules()]
    graph = onnx.load(path)
    assert sum(node.op_type == "DFT" for node in graph.graph.node) == (8 if vocal_feature_n_fft == 4096 else 7)
    metadata = json.loads({m.key: m.value for m in graph.metadata_props}["stemgenrt.experimental"])
    assert metadata["state_count"] == 13 and metadata["native_host_qualified"] is False
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.add_session_config_entry("mlas.disable_kleidiai", "1")
    session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
    contract = report["interface"]
    for nodes, names, shapes in ((session.get_inputs(), contract["input_names"], contract["input_shapes"]),
                                 (session.get_outputs(), contract["output_names"], contract["output_shapes"])):
        assert [node.name for node in nodes] == names
        assert [node.shape for node in nodes] == shapes
        assert all(node.type == "tensor(float)" for node in nodes)
    audio = torch.randn(1, 2, 128 * 192) * .07
    audio[..., :1024] = 0
    audio[..., 1025] = 1
    audio[..., -4096:] = 0
    audio = torch.cat((audio, torch.zeros(1, 2, 128)), -1)  # One true EOF flush.
    native, runtime, first = None, None, None
    with torch.inference_mode():
        for index, chunk in enumerate(audio.split(128, -1)):
            if index in (0, 64):  # A complete transport reset, including every child's state.
                native = None
                runtime = [np.zeros(shape, np.float32) for shape in contract["input_shapes"][1:]]
            expected = model.render(chunk, native)
            native = expected.state
            actual = session.run(None, dict(zip(contract["input_names"], [chunk.numpy(), *runtime])))
            np.testing.assert_allclose(actual[0], expected.deployed.numpy(), atol=1e-5, rtol=2e-4)
            for found, want in zip(actual[1:], [v for child in native for v in child], strict=True):
                np.testing.assert_allclose(found, want.numpy(), atol=5e-4, rtol=2e-4)
            physical = expected.delayed_mixture.numpy()
            residual = physical - ((actual[0][:, 0] + actual[0][:, 1]) + actual[0][:, 2])
            np.testing.assert_array_equal(actual[0][:, 3], residual)
            np.testing.assert_allclose(actual[0].sum(1), physical, atol=2e-7, rtol=0)
            if index == 0:
                first = actual
            runtime = actual[1:]
    zeros = [np.zeros(shape, np.float32) for shape in contract["input_shapes"]]
    reset = session.run(None, dict(zip(contract["input_names"], zeros)))
    for got, want in zip(reset, first, strict=True):
        np.testing.assert_array_equal(got, want)
    with pytest.raises(ValueError, match="new path"):
        export_fp32(model, path)


def test_combined_export_rejects_nonfinite_child():
    model = IndependentStems(drums=SpecialistSeparator(source="drums").eval(),
        bass=SpecialistSeparator(source="bass").eval(), vocals=BandSeparator(sources=("vocals",)).eval())
    with torch.no_grad():
        next(model.bass.parameters()).flatten()[0] = float("nan")
    with pytest.raises(ValueError, match="finite FP32"):
        make_export_copy(model)
