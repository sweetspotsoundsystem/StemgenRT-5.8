"""CPU inference for the released stereo, hop-128 ONNX model.

This convenience API is synchronous and allocates memory. Use a dedicated
worker when integrating it with audio playback; instances are not thread-safe.
"""

from hashlib import sha256
import json
from pathlib import Path

import numpy as np


_MODELS = json.loads(Path(__file__).with_name("streaming_models.json").read_text())
MODEL_SHA256 = _MODELS["current"]["sha256"]
SAMPLE_RATE = 44100
HOP_SAMPLES = 128
SOURCE_ORDER = ("drums", "bass", "vocals", "other")


class StreamingSeparator:
    """Separate float32 audio at 44.1 kHz, preserving every recurrent state.

    The current model has thirteen independent Drums/Bass/Vocals states.
    The v0.6.2 eight-state model remains supported with its explicit checksum.
    Custom exports require an explicit checksum and one of these exact ABIs.

    ``process_chunk`` accepts [2, 128] and returns [4, 2, 128], aligned to
    the previous input hop. Its first result after reset is ``None``.
    ``flush`` emits the last pending hop and resets the stream.
    """

    def __init__(self, model_path, *, sample_rate=SAMPLE_RATE, expected_sha256=MODEL_SHA256):
        if sample_rate != SAMPLE_RATE:
            raise ValueError("The released model requires 44100 Hz stereo audio")
        if (not isinstance(expected_sha256, str) or len(expected_sha256) != 64
                or any(character not in "0123456789abcdef" for character in expected_sha256)):
            raise ValueError("expected_sha256 must be a lowercase SHA-256 digest")
        path = Path(model_path)
        digest = sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != expected_sha256:
            raise ValueError("Model SHA-256 differs from the expected model")

        import onnxruntime as ort

        ort.disable_telemetry_events()
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.add_session_config_entry("session.intra_op.allow_spinning", "0")
        options.add_session_config_entry("session.inter_op.allow_spinning", "0")
        options.add_session_config_entry("mlas.disable_kleidiai", "1")
        self._session = ort.InferenceSession(
            str(path), sess_options=options, providers=["CPUExecutionProvider"]
        )
        actual_inputs = {node.name: tuple(node.shape) for node in self._session.get_inputs()}
        pinned = next((model for model in _MODELS.values()
                       if model["sha256"] == expected_sha256), None)
        contracts = [pinned] if pinned is not None else list(_MODELS.values())
        self._state_shapes = None
        for contract in contracts:
            states = {name: tuple(shape) for name, shape in contract["states"].items()}
            if actual_inputs == {"audio_chunk": (1, 2, 128), **states}:
                self._state_shapes = states
                break
        if self._state_shapes is None:
            raise ValueError("Model streaming interface differs")
        inputs = {"audio_chunk": (1, 2, 128), **self._state_shapes}
        outputs = {"separated_chunk": (1, 4, 2, 128)}
        outputs.update({"next_" + name: shape for name, shape in self._state_shapes.items()})
        for nodes, expected in ((self._session.get_inputs(), inputs),
                                (self._session.get_outputs(), outputs)):
            actual = {node.name: tuple(node.shape) for node in nodes}
            if actual != expected or any(node.type != "tensor(float)" for node in nodes):
                raise ValueError("Model streaming interface differs")
        self._output_names = list(outputs)
        self.reset()

    def reset(self):
        """Clear every state after a seek, input gap, or new audio stream."""
        self._state = {name: np.zeros(shape, dtype=np.float32)
                       for name, shape in self._state_shapes.items()}
        self._pending = False

    @staticmethod
    def _audio(audio):
        audio = np.asarray(audio)
        if audio.dtype != np.float32 or audio.ndim != 2 or audio.shape[0] != 2:
            raise ValueError("Audio must be a float32 array with shape [2, samples]")
        if not np.isfinite(audio).all():
            raise ValueError("Audio must contain only finite samples")
        return np.ascontiguousarray(audio)

    def process_chunk(self, audio):
        """Advance one complete hop; return None only for initial pre-roll.

        Input and inference errors reset all state before being raised.
        No input gain normalization or output confidence fade is applied.
        """
        try:
            audio = self._audio(audio)
            if audio.shape[1] != HOP_SAMPLES:
                raise ValueError("process_chunk requires exactly 128 samples")
            values = self._session.run(
                self._output_names, {"audio_chunk": audio[None], **self._state}
            )
            if not all(np.isfinite(value).all() for value in values):
                raise RuntimeError("Model returned non-finite output or state")
            valid = self._pending
            self._state = dict(zip(self._state_shapes, values[1:]))
            self._pending = True
            return values[0][0] if valid else None
        except Exception:
            self.reset()
            raise

    def flush(self):
        """Emit the final pending hop with one zero hop, then clear state.

        Returns None for an empty stream or a repeated flush. Trim the result
        to the real input length if the caller padded the final input hop.
        """
        if not self._pending:
            return None
        try:
            return self.process_chunk(np.zeros((2, HOP_SAMPLES), dtype=np.float32))
        finally:
            self.reset()

    def separate(self, audio):
        """Render a complete [2, T] clip to aligned [4, 2, T] stem audio.

        Starts from zero state, pads a partial final hop, flushes exactly once,
        and removes padding. Empty clips return shape [4, 2, 0].
        """
        self.reset()
        try:
            audio = self._audio(audio)
            frames = audio.shape[1]
            if frames == 0:
                return np.empty((4, 2, 0), dtype=np.float32)
            rendered = []
            for offset in range(0, frames, HOP_SAMPLES):
                chunk = audio[:, offset:offset + HOP_SAMPLES]
                if chunk.shape[1] < HOP_SAMPLES:
                    chunk = np.pad(chunk, ((0, 0), (0, HOP_SAMPLES - chunk.shape[1])))
                output = self.process_chunk(chunk)
                if output is not None:
                    rendered.append(output)
            rendered.append(self.flush())
            return np.concatenate(rendered, axis=-1)[..., :frames]
        finally:
            self.reset()
