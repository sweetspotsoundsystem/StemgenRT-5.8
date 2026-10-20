"""Native recurrence fidelity, integer arithmetic, and conversion rejection."""
import json

import numpy as np
import pytest
import torch

onnx = pytest.importorskip("onnx")
ort = pytest.importorskip("onnxruntime")
from onnx import helper as h, numpy_helper as nh, TensorProto as T
from stemgenrt.banded import BandSeparator
from stemgenrt.specialist import SpecialistSeparator
from stemgenrt.independent import IndependentStems
from stemgenrt.independent_export import export_fp32
from stemgenrt.independent_optimization import optimize, fuse_grus, sha, VARIANTS


def session(path):
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.add_session_config_entry("mlas.disable_kleidiai", "1")
    return ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    directory = tmp_path_factory.mktemp("independent-optimization")
    torch.manual_seed(20261020)
    model = IndependentStems(drums=SpecialistSeparator(source="drums").eval(),
        bass=SpecialistSeparator(source="bass").eval(), vocals=BandSeparator(sources=("vocals",)).eval())
    with torch.no_grad():
        for stem in ("drums", "bass", "vocals"):
            for head in getattr(model, stem).backbone.mask_heads:
                head.weight.normal_(std=.01)
    model.eval().requires_grad_(False)
    original = directory / "original.onnx"
    export_fp32(model, original)
    digest = sha(original)
    paths, reports = {}, {}
    for variant in VARIANTS:
        paths[variant] = directory / (variant + ".onnx")
        reports[variant] = optimize(original, paths[variant], expected_sha256=digest, variant=variant)
    return model, original, digest, paths, reports


@pytest.mark.parametrize("variant", ["layout-fp32", "fused-fp32", "fused-layout-fp32"])
def test_fp32_variants_match_native_with_continuous_state_reset_and_flush(exported, variant):
    model, original, digest, paths, reports = exported
    runtime = session(paths[variant])
    names = [v.name for v in runtime.get_inputs()]
    shapes = [v.shape for v in runtime.get_inputs()]
    audio = torch.randn(1, 2, 128 * 96, generator=torch.Generator().manual_seed(22)) * .1
    audio[..., :1024] = 0
    audio[..., 1025] = .8
    audio[..., -4096:] = 0
    first = None
    with torch.inference_mode():
        for index, chunk in enumerate(audio.split(128, dim=-1)):
            if index in (0, 48):
                native = None
                states = [np.zeros(shape, np.float32) for shape in shapes[1:]]
            expected = model.render(chunk, native)
            native = expected.state
            actual = runtime.run(None, dict(zip(names, [chunk.numpy(), *states])))
            np.testing.assert_allclose(actual[0], expected.deployed.numpy(), atol=1e-5, rtol=2e-4)
            for found, wanted in zip(actual[1:], [x for child in native for x in child], strict=True):
                np.testing.assert_allclose(found, wanted.numpy(), atol=5e-5, rtol=2e-4)
            physical = expected.delayed_mixture.numpy()
            np.testing.assert_array_equal(actual[0][:, 3], physical - ((actual[0][:, 0] + actual[0][:, 1]) + actual[0][:, 2]))
            if index == 0:
                first = actual
            states = actual[1:]
    reset = runtime.run(None, dict(zip(names, [np.zeros(shape, np.float32) for shape in shapes])))
    for a, b in zip(reset, first, strict=True):
        np.testing.assert_array_equal(a, b)
    assert sha(original) == digest
    assert reports[variant]["public_tensor_interface_unchanged"]


@pytest.mark.parametrize("variant", ["layout-fp32", "fused-layout-fp32"])
def test_fp32_preserves_nonzero_public_state_contract(exported, variant):
    _, original, _, paths, _ = exported
    source, candidate = session(original), session(paths[variant])
    rng = np.random.default_rng(83)
    for _ in range(8):
        feeds = {v.name: rng.normal(0, .2, v.shape).astype(np.float32) for v in source.get_inputs()}
        expected, actual = source.run(None, feeds), candidate.run(None, feeds)
        for a, b in zip(actual, expected, strict=True):
            np.testing.assert_allclose(a, b, atol=5e-5, rtol=2e-4)


def test_selective_quantized_product_matches_independent_integer_dot(exported, tmp_path):
    _, original, _, paths, reports = exported
    graph = onnx.load(paths["fused-layout-global-int8"])
    originals = {w.name: nh.to_array(w) for w in onnx.load(original).graph.initializer}
    stored = {w.name: w for w in graph.graph.initializer}
    for index, row in enumerate(reports["fused-layout-global-int8"]["transformation_proof"]["quantization"]):
        prefix = row["node"] + "/u8s8"
        selected = [n for n in graph.graph.node if n.name.startswith(prefix + "/")]
        names = {name for n in selected for name in n.input if name in stored}
        shape = row["shape"]
        probe = h.make_model(h.make_graph(selected, "integer-product", [h.make_tensor_value_info(selected[0].input[0], T.FLOAT, [1, 1, shape[0]])],
            [h.make_tensor_value_info(selected[-1].output[0], T.FLOAT, [1, 1, shape[1]])],
            initializer=[stored[n] for n in names]), opset_imports=[h.make_opsetid("", 17)], ir_version=graph.ir_version)
        path = tmp_path / (str(index) + ".onnx")
        onnx.save(probe, path)
        runtime = session(path)
        rng = np.random.default_rng(index)
        for zero in (False, True):
            values = np.zeros((1, 1, shape[0]), np.float32) if zero else rng.normal(0, .3, (1, 1, shape[0])).astype(np.float32)
            weight = originals[row["source_initializer"]]
            weight_scale = np.float32(np.max(np.abs(weight)) / 64.)
            quantized_w = np.clip(np.rint(weight / weight_scale), -64, 64).astype(np.int32)
            lo, hi = min(float(values.min()), 0.), max(float(values.max()), 0.)
            input_scale = np.float32((np.float32(hi) - np.float32(lo)) / np.float32(255.)) if hi != lo else np.float32(1.)
            input_zero = np.int32(np.clip(np.rint(-np.float32(lo) / input_scale), 0, 255))
            quantized_x = np.clip(np.rint(values / input_scale) + input_zero, 0, 255).astype(np.int32)
            expected = ((quantized_x - input_zero) @ quantized_w).astype(np.float32) * np.float32(input_scale * weight_scale)
            actual = runtime.run(None, {runtime.get_inputs()[0].name: values})[0]
            np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=1e-6)
    assert len(reports["fused-layout-global-int8"]["transformation_proof"]["quantization"]) == 12


def test_conversion_rejects_different_recurrence_and_preserves_existing_files(exported, tmp_path):
    _, original, digest, paths, _ = exported
    graph = onnx.load(original)
    node = next(n for n in graph.graph.node if n.name == "/drums/backbone/local.0/Sub")
    node.input.reverse()
    with pytest.raises(ValueError, match="gate arithmetic"):
        fuse_grus(graph)
    with pytest.raises(ValueError, match="identity differs"):
        optimize(original, tmp_path / "bad.onnx", expected_sha256="0" * 64)
    with pytest.raises(ValueError, match="new candidate"):
        optimize(original, paths["fused-fp32"], expected_sha256=digest)
    with pytest.raises(ValueError, match="original independent FP32"):
        optimize(paths["fused-fp32"], tmp_path / "twice.onnx", expected_sha256=sha(paths["fused-fp32"]))
    assert sha(original) == digest
