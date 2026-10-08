"""Physical alignment, EOF and output-policy checks for the complete joint graph."""
import numpy as np
import pytest
import soundfile as sf
import torch

from examples.separate_joint import JointOnnxRenderer, separate_file
from stemgenrt.banded import BandSeparator, SOURCE_ORDER
from stemgenrt.band_export import export_fp32
from stemgenrt.evaluation import EvaluationTrack, NativeRenderer, stream_track


@pytest.fixture
def graph(tmp_path):
    torch.manual_seed(751)
    model = BandSeparator(sources=SOURCE_ORDER, band_width=16, global_width=16, layers=1).eval()
    with torch.no_grad():
        for head in model.backbone.mask_heads:
            head.weight.normal_(std=.01)
    path = tmp_path / "joint.onnx"
    report = export_fp32(model, path)
    return model, path, report


def test_joint_onnx_continuous_prefix_gaps_and_partial_eof(graph, tmp_path):
    model, path, report = graph
    audio = np.random.default_rng(17).normal(0, .05, (2, 2107)).astype(np.float32)
    mixture = tmp_path / "mixture.wav"
    sf.write(mixture, audio.T, 44100, subtype="FLOAT")
    track = EvaluationTrack("test", mixture, (mixture,) * 4, ((331, 900), (1801, 2107)))
    renderer = JointOnnxRenderer(path, report["onnx_sha256"])
    actual, metadata = stream_track(renderer, track, frames=2107, unroll_hops=3)
    expected, reference = stream_track(NativeRenderer(model), track, frames=2107, unroll_hops=7)
    assert metadata == reference and metadata["coverage_complete"] and metadata["zero_input_samples"] > 0
    for found, want in zip(actual, expected):
        np.testing.assert_allclose(found, want, atol=1e-5, rtol=2e-5)
    repeated, _ = stream_track(renderer, track, frames=2107, unroll_hops=11)
    for found, want in zip(actual, repeated):
        np.testing.assert_array_equal(found, want)


def test_joint_onnx_every_state_and_reset(graph):
    model, path, report = graph
    renderer = JointOnnxRenderer(path, report["onnx_sha256"])
    audio = torch.randn(1, 2, 128 * 8) * .04
    initial = model.initial_state(1)
    state = initial
    with torch.no_grad():
        for chunk in audio.split(128, -1):
            output, state = model.forward_chunk(chunk, state)
            actual = renderer.render(chunk[0].numpy())
            np.testing.assert_allclose(actual, output[0].numpy(), atol=1e-5, rtol=2e-5)
            for found, expected in zip(renderer.state, state):
                np.testing.assert_allclose(found, expected.numpy(), atol=5e-5, rtol=2e-5)
    renderer.reset()
    assert all(np.count_nonzero(value) == 0 for value in renderer.state)


def test_joint_file_native_outputs_and_optional_physical_other(graph, tmp_path):
    _, path, report = graph
    audio = np.random.default_rng(32).normal(0, .2, (517, 2)).astype(np.float32)
    mixture = tmp_path / "input.wav"
    sf.write(mixture, audio, 44100, subtype="FLOAT")
    renderer = JointOnnxRenderer(path, report["onnx_sha256"])
    native = tmp_path / "native"
    residual = tmp_path / "residual"
    first = separate_file(renderer, mixture, native)
    second = separate_file(renderer, mixture, residual, residual_other=True)
    assert first["output_policy"] != second["output_policy"]
    stems = []
    for name in SOURCE_ORDER:
        left, rate = sf.read(native / (name + ".wav"), dtype="float32", always_2d=True)
        right, _ = sf.read(residual / (name + ".wav"), dtype="float32", always_2d=True)
        assert rate == 44100 and left.shape == right.shape == audio.shape
        if name != "other":
            np.testing.assert_array_equal(left, right)
        stems.append(right)
    np.testing.assert_array_equal(stems[3], audio - ((stems[0] + stems[1]) + stems[2]))
    with pytest.raises(ValueError, match="new output"):
        separate_file(renderer, mixture, native)


def test_joint_onnx_rejects_digest_and_single_source(graph, tmp_path):
    _, path, _ = graph
    with pytest.raises(ValueError, match="SHA256"):
        JointOnnxRenderer(path, "0" * 64)
    vocals = tmp_path / "vocals.onnx"
    report = export_fp32(BandSeparator(band_width=16, global_width=16, layers=1).eval(), vocals)
    with pytest.raises(ValueError, match="four-source"):
        JointOnnxRenderer(vocals, report["onnx_sha256"])


def test_native_metadata_fallback_keeps_conflicting_attribute_rejection():
    model = BandSeparator(sources=SOURCE_ORDER, band_width=16, global_width=16, layers=1).eval()
    model.hop_samples = 256
    with pytest.raises(ValueError, match="geometry"):
        NativeRenderer(model)
