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


@pytest.mark.parametrize("vocal_feature_n_fft,bass_waveform_basis,drums_feature_n_fft", [(1024, 0, 1024), (4096, 0, 1024), (1024, 128, 1024), (1024, 0, 4096)])
@pytest.mark.parametrize("batch_carrier_analysis", [False, True])
@pytest.mark.parametrize("batch_gru_gates", [False, True])
def test_combined_onnx_matches_native_stream_flush_reset_and_preserves_source(tmp_path, vocal_feature_n_fft, bass_waveform_basis, drums_feature_n_fft, batch_carrier_analysis, batch_gru_gates):
    ort = pytest.importorskip("onnxruntime")
    onnx = pytest.importorskip("onnx")
    torch.manual_seed(20261010)
    children = {source: SpecialistSeparator(source=source, waveform_basis=bass_waveform_basis if source == "bass" else 256, feature_n_fft=4096 if source == "bass" else drums_feature_n_fft).eval() for source in ("drums", "bass")}
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
    report = export_fp32(model, path, batch_carrier_analysis=batch_carrier_analysis,
                         batch_gru_gates=batch_gru_gates)
    assert state_sha256(model.state_dict()) == digest and torch.equal(torch.get_rng_state(), rng)
    assert flags == [(m.training, [p.requires_grad for p in m.parameters()]) for m in model.modules()]
    graph = onnx.load(path)
    assert sum(node.op_type == "DFT" for node in graph.graph.node) == (7 - 2 * batch_carrier_analysis + (vocal_feature_n_fft == 4096) + (drums_feature_n_fft == 4096))
    metadata = json.loads({m.key: m.value for m in graph.metadata_props}["stemgenrt.experimental"])
    assert metadata["state_count"] == 13 + bool(bass_waveform_basis) and metadata["native_host_qualified"] is False
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


@pytest.mark.parametrize("optimization", ["carrier_analysis", "gru_gates"])
def test_batched_operations_preserve_unequal_public_histories_windows_and_weights(tmp_path, optimization):
    ort = pytest.importorskip("onnxruntime")
    onnx = pytest.importorskip("onnx")
    torch.manual_seed(20261023)
    model = IndependentStems(drums=SpecialistSeparator(source="drums").eval(),
        bass=SpecialistSeparator(source="bass").eval(), vocals=BandSeparator(sources=("vocals",)).eval())
    with torch.no_grad():
        for child in (model.drums, model.bass, model.vocals):
            for head in child.backbone.mask_heads:
                head.weight.normal_(std=.01)
        # The fast path must not silently assume identical registered windows.
        model.vocals.analysis_window.mul_(.9)
    before = state_sha256(model.state_dict())
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.add_session_config_entry("mlas.disable_kleidiai", "1")
    sessions, graphs, reports = [], [], []
    for batched in (False, True):
        path = tmp_path / f"{optimization}-{batched}.onnx"
        reports.append(export_fp32(model, path, **{"batch_" + optimization: batched}))
        graphs.append(onnx.load(path))
        sessions.append(ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"]))
    assert reports[0]["interface"] == reports[1]["interface"]
    assert reports[0]["weight_sha256"] == reports[1]["weight_sha256"] == before
    assert reports[0]["child_weight_sha256"] == reports[1]["child_weight_sha256"]
    assert reports[1]["execution_variant"] == "batched-" + optimization.replace("_", "-") + "-fp32-v1"
    assert state_sha256(model.state_dict()) == before
    # Every learned initializer retains its literal name, dtype, shape and bytes.
    weights = [{v.name: onnx.numpy_helper.to_array(v) for v in g.graph.initializer} for g in graphs]
    for name, value in weights[0].items():
        if "backbone" in name or "waveform" in name:
            np.testing.assert_array_equal(weights[1][name], value)
    if optimization == "gru_gates":
        # Constant-folded affine initializers can have generated names. Match
        # each operation by its semantic node path, not that incidental name.
        affines = [{node.name: node for node in graph.graph.node
                    if node.op_type in ("Gemm", "MatMul")} for graph in graphs]
        assert set(affines[0]) == set(affines[1])
        for name, first in affines[0].items():
            second = affines[1][name]
            assert first.attribute == second.attribute
            for a, b in zip(first.input[1:], second.input[1:], strict=True):
                assert (a in weights[0]) == (b in weights[1])
                if a in weights[0]:
                    np.testing.assert_array_equal(weights[0][a], weights[1][b])
    rng = np.random.default_rng(23)
    for _ in range(8):
        feeds = {node.name: rng.normal(0, .2, node.shape).astype(np.float32)
                 for node in sessions[0].get_inputs()}
        expected, actual = (s.run(None, feeds) for s in sessions)
        for index, (a, b) in enumerate(zip(actual, expected, strict=True)):
            if optimization == "gru_gates":
                np.testing.assert_array_equal(a, b)
                continue
            # Batched feature reductions may change FP32 rounding. Use the
            # established absolute waveform/state limits, without widening
            # them or claiming bitwise-identical execution.
            np.testing.assert_allclose(a, b, atol=1e-5 if index == 0 else 5e-4, rtol=0)
