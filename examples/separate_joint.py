"""Separate a stereo 44.1 kHz file with an authenticated four-state joint graph.

This example uses the experimental band export, including its FFT and synthesis.
It preserves all four learned outputs unless residual Other is requested.
"""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import soundfile as sf

from stemgenrt.evaluation import EvaluationTrack, SOURCE_ORDER, stream_track


def require(condition, message):
    if not condition:
        raise ValueError(message)


class JointOnnxRenderer:
    """Raw delayed graph coordinates, suitable for continuous ``stream_track``."""

    def __init__(self, model_path, expected_sha256):
        import onnxruntime as ort

        path = Path(model_path)
        require(sha256(path.read_bytes()).hexdigest() == expected_sha256, "Joint ONNX SHA256 differs")
        options = ort.SessionOptions()
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.add_session_config_entry("session.intra_op.allow_spinning", "0")
        options.add_session_config_entry("session.inter_op.allow_spinning", "0")
        options.add_session_config_entry("mlas.enable_kleidiai", "0")
        self.session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
        self.metadata = json.loads(self.session.get_modelmeta().custom_metadata_map["stemgenrt.experimental"])
        arch = self.metadata["architecture"]
        require(self.metadata["schema"] == "stemgenrt-experimental-band-onnx-v1"
                and self.metadata["precision"] == "fp32"
                and arch["source_order"] == list(SOURCE_ORDER)
                and arch["sample_rate"] == 44100
                and arch["hop_samples"] == arch["graph_alignment_samples"] == 128
                and arch["feature_history_samples"] == 896,
                "Require the FP32 four-source band export")
        self.input_names = ["audio_chunk", "audio_history", "local_hidden", "global_hidden", "spectral_numerator_tail"]
        self.output_names = ["separated_chunk", *("next_" + name for name in self.input_names[1:])]
        self.state_shapes = [[1, 2, 896], [arch["layers"], 20, arch["band_width"]],
                             [arch["layers"], 1, arch["global_width"]], [1, 4, 2, 128]]
        inputs, outputs = self.session.get_inputs(), self.session.get_outputs()
        require([value.name for value in inputs] == self.input_names
                and [value.name for value in outputs] == self.output_names
                and [value.shape for value in inputs] == [[1, 2, 128], *self.state_shapes]
                and [value.shape for value in outputs] == [[1, 4, 2, 128], *self.state_shapes]
                and all(value.type == "tensor(float)" for value in (*inputs, *outputs)),
                "Joint ONNX interface differs")
        self.reset()

    def reset(self):
        self.state = [np.zeros(shape, dtype=np.float32) for shape in self.state_shapes]

    def render(self, audio):
        require(audio.dtype == np.float32 and audio.ndim == 2 and audio.shape[0] == 2
                and audio.shape[1] > 0 and audio.shape[1] % 128 == 0 and np.isfinite(audio).all(),
                "Require finite FP32 stereo audio in complete 128-sample hops")
        outputs = []
        for offset in range(0, audio.shape[1], 128):
            chunk = np.ascontiguousarray(audio[None, :, offset:offset + 128])
            result = self.session.run(self.output_names, dict(zip(self.input_names, [chunk, *self.state])))
            require(all(np.isfinite(value).all() for value in result), "Nonfinite ONNX output or state")
            outputs.append(result[0][0])
            self.state = result[1:]
        return np.concatenate(outputs, axis=-1)


def separate_file(renderer, input_path, output_directory, *, residual_other=False):
    input_path, output = Path(input_path).resolve(), Path(output_directory).resolve()
    info = sf.info(input_path)
    require(info.samplerate == 44100 and info.channels == 2 and info.frames > 0,
            "Input must be nonempty stereo 44.1 kHz audio")
    require(not output.exists(), "Use a new output directory")
    # Only the mixture path is read by stream_track. It pads and flushes true
    # EOF, then removes exactly one hop of graph alignment from the capture.
    track = EvaluationTrack(input_path.stem, input_path, (input_path,) * 4, ((0, info.frames),))
    captures, stream = stream_track(renderer, track, frames=info.frames)
    stems = captures[0]
    if residual_other:
        mixture, rate = sf.read(input_path, dtype="float32", always_2d=True)
        require(rate == 44100 and mixture.shape == (info.frames, 2) and np.isfinite(mixture).all(),
                "Input changed while separating")
        stems[3] = mixture.T - ((stems[0] + stems[1]) + stems[2])
    output.mkdir(parents=True)
    files = {}
    for name, stem in zip(SOURCE_ORDER, stems, strict=True):
        path = output / (name + ".wav")
        sf.write(path, stem.T, 44100, subtype="FLOAT")
        restored, rate = sf.read(path, dtype="float32", always_2d=True)
        require(rate == 44100 and np.array_equal(restored, stem.T), "WAV serialization changed samples")
        files[name] = {"sha256": sha256(path.read_bytes()).hexdigest(), "frames": info.frames}
    report = {"source_order": list(SOURCE_ORDER), "sample_rate": 44100, "stream": stream,
        "output_policy": "joint DBV with physical residual Other" if residual_other else "all four learned joint outputs",
        "normalization": "none; native float32 levels", "files": files,
        "native_mac_runtime_measured": False, "plugin_qualified": False}
    (output / "render.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--residual-other", action="store_true")
    args = parser.parse_args()
    renderer = JointOnnxRenderer(args.model, args.sha256)
    report = separate_file(renderer, args.input, args.output_directory, residual_other=args.residual_other)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
