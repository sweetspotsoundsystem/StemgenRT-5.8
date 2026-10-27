"""Compare deployed ONNX output with an independent CPU PyTorch oracle."""

import hashlib
import json
from pathlib import Path
import struct

import numpy as np
import pytest

from stemgenrt.streaming import MODEL_SHA256, StreamingSeparator


ROOT = Path(__file__).resolve().parents[1]


def references(name="independent-pytorch", expected_sha256=MODEL_SHA256):
    path = ROOT / "tests/fixtures" / (name + ".bin")
    data = path.read_bytes()
    metadata = json.loads(path.with_suffix(".json").read_text())
    assert hashlib.sha256(data).hexdigest() == metadata["fixture_sha256"]
    assert metadata["deployment_graph_sha256"] == expected_sha256
    assert data[:8] == b"SGRTG001"
    count, = struct.unpack_from("<I", data, 8)
    offset = 12
    result = []
    for _ in range(count):
        frames, = struct.unpack_from("<I", data, offset)
        offset += 4
        audio = np.frombuffer(data, dtype="<f4", count=2 * frames, offset=offset).reshape(2, frames)
        offset += 8 * frames
        expected = np.frombuffer(data, dtype="<f4", count=8 * frames, offset=offset).reshape(4, 2, frames)
        offset += 32 * frames
        result.append((audio, expected))
    assert offset == len(data)
    assert [audio.shape[-1] for audio, _ in result] == metadata["frames"]
    return result


@pytest.fixture(scope="module")
def separator():
    return StreamingSeparator(ROOT / "models/independent.onnx")


@pytest.mark.parametrize("audio,expected", references(), ids=lambda x: str(x.shape))
def test_partial_eof_and_all_four_stems(separator, audio, expected):
    output = separator.separate(audio)
    assert output.shape == expected.shape
    assert np.isfinite(output).all()
    np.testing.assert_allclose(output, expected, rtol=0, atol=1e-5)
    np.testing.assert_allclose(output.sum(axis=0), audio, rtol=0, atol=1e-6)
    np.testing.assert_array_equal(separator.separate(audio), output)
    assert separator.flush() is None


def test_chunk_alignment_flush_and_reset(separator):
    audio, expected = references()[4]  # 255 frames: one complete and one partial hop.
    separator.reset()
    assert separator.process_chunk(audio[:, :128]) is None
    first = separator.process_chunk(np.pad(audio[:, 128:], ((0, 0), (0, 1))))
    final = separator.flush()
    np.testing.assert_allclose(first, expected[..., :128], rtol=0, atol=1e-5)
    np.testing.assert_allclose(final[..., :127], expected[..., 128:], rtol=0, atol=1e-5)
    assert separator.flush() is None
    assert separator.process_chunk(audio[:, :128]) is None
    separator.reset()
    assert separator.flush() is None


def test_invalid_audio_resets_stream(separator):
    valid = np.ones((2, 128), dtype=np.float32) * .1
    for invalid in (valid.astype(np.float64), valid[:, :127], valid[:1], valid * np.nan):
        separator.reset()
        assert separator.process_chunk(valid) is None
        with pytest.raises(ValueError):
            separator.process_chunk(invalid)
        assert separator.flush() is None
    assert separator.separate(np.empty((2, 0), dtype=np.float32)).shape == (4, 2, 0)


def test_reject_wrong_rate_and_wrong_model(tmp_path):
    path = tmp_path / "wrong.onnx"
    path.write_bytes(b"wrong model")
    with pytest.raises(ValueError, match="44100"):
        StreamingSeparator(path, sample_rate=48000)
    with pytest.raises(ValueError, match="SHA-256"):
        StreamingSeparator(path)


def test_runtime_uses_the_checked_m4_backend_configuration(separator):
    options = separator._session.get_session_options()
    assert options.intra_op_num_threads == options.inter_op_num_threads == 1
    assert options.get_session_config_entry("session.intra_op.allow_spinning") == "0"
    assert options.get_session_config_entry("session.inter_op.allow_spinning") == "0"
    assert options.get_session_config_entry("mlas.disable_kleidiai") == "1"


@pytest.mark.parametrize("changed_name,changed_shape", [
    ("drums_local_hidden", (2, 19, 96)),
    ("bass_audio_history", (1, 2, 896)),
    ("vocals_global_hidden", None),
    ("drums_waveform_numerator_tail", (1, 4, 2, 128)),
    ("unexpected_state", (1, 1)),
    ("past_carrier_history", (1, 2, 2, 513, 2)),
])
def test_rejects_incomplete_or_changed_state_interface(separator, monkeypatch, changed_name, changed_shape):
    from types import SimpleNamespace
    import onnxruntime as ort

    inputs = {node.name: tuple(node.shape) for node in separator._session.get_inputs()}
    if changed_shape is None:
        inputs.pop(changed_name)
    else:
        inputs[changed_name] = changed_shape
    nodes = [SimpleNamespace(name=name, shape=shape, type="tensor(float)")
             for name, shape in inputs.items()]
    monkeypatch.setattr(ort, "InferenceSession", lambda *args, **kwargs:
                        SimpleNamespace(get_inputs=lambda: nodes,
                                        get_outputs=separator._session.get_outputs))
    with pytest.raises(ValueError, match="interface"):
        StreamingSeparator(ROOT / "models/independent.onnx")


def test_retained_v062_all_four_stem_parity_and_reset():
    models = json.loads((ROOT / "stemgenrt/streaming_models.json").read_text())
    legacy = models["v0.6.2"]
    path = ROOT / "models" / legacy["filename"]
    if not path.is_file():
        pytest.skip("Legacy weights withdrawn; exact parity requires an existing local copy")
    separator = StreamingSeparator(path, expected_sha256=legacy["sha256"])
    for audio, expected in references("hop128-pytorch", legacy["sha256"]):
        output = separator.separate(audio)
        np.testing.assert_allclose(output, expected, rtol=0, atol=1e-5)
        np.testing.assert_array_equal(separator.separate(audio), output)
        np.testing.assert_allclose(output.sum(axis=0), audio, rtol=0, atol=1e-6)
        assert separator.flush() is None
