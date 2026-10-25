"""Complete FP32 one-hop export for independent bass and drums research models.

The graph includes both FFT analyses when used, the recurrent core, spectral
synthesis and the drum waveform branch. This is a research interface; exporting
it does not qualify the released plugin or establish native Mac runtime.
"""
import copy
import json
from pathlib import Path

import torch
from torch import nn

from ._export.fp32 import _RFFT1024
from ._export.helpers import OneFrameGRUForONNX, replace_rmsnorm_layers, model_state_sha256, sha
from ._model.dsp import require
from .band_export import _IRFFT1024, _RFFT4096, carrier_features
from .specialist import SpecialistSeparator


def interface(model):
    require(type(model) is SpecialistSeparator, "Expected an independent stem specialist")
    states = model.initial_state(1)
    return {"state_names": list(model.state_names),
            "input_names": ["audio_chunk", *model.state_names],
            "output_names": ["separated_chunk", *("next_" + n for n in model.state_names)],
            "input_shapes": [[1, 2, 128], *[list(v.shape) for v in states]],
            "output_shapes": [[1, 1, 2, 128], *[list(v.shape) for v in states]]}


class _SpecialistStreamingWrapper(nn.Module):
    def __init__(self, copied):
        super().__init__()
        self.model = copied

    def forward(self, audio_chunk, *states, carrier=None):
        model = self.model
        audio_history, local_hidden, global_hidden, spectral_tail = states[:4]
        joined = torch.cat((audio_history, audio_chunk), -1)
        if carrier is None:
            spectrum = _RFFT1024.apply((joined[..., -1024:] * model.analysis_window).reshape(2, 1024))
            spectrum = spectrum.reshape(1, 2, 1, 513, 2)
            features = carrier_features(spectrum)
        else:
            spectrum, features = carrier
        long_features = None
        if model.feature_n_fft == 4096:
            long_spectrum = _RFFT4096.apply((joined * model.long_analysis_window).reshape(2, 4096))
            long_spectrum = long_spectrum.reshape(1, 2, 1, 2049, 2)
            long_power = long_spectrum[..., 0].square() + long_spectrum[..., 1].square()
            long_scale = long_power.mean((1, 3), keepdim=True).clamp_min(1e-8).sqrt()
            long_features = torch.log1p((long_power + 1e-12).sqrt() / long_scale).permute(0, 2, 1, 3)
        masks, local, glob, context = model.backbone(features, local_hidden, global_hidden, long_features)
        masks = masks.permute(0, 2, 3, 1, 4, 5)
        real, imag = spectrum[:, None, ..., 0], spectrum[:, None, ..., 1]
        separated = torch.stack((real * masks[..., 0] - imag * masks[..., 1],
                                 real * masks[..., 1] + imag * masks[..., 0]), -1)
        frames = _IRFFT1024.apply(separated.reshape(2, 513, 2))
        frames = frames[..., 768:].reshape(1, 1, 2, 256) * model.synthesis.spectral_window
        emitted = (frames[..., :128] + spectral_tail) / model.synthesis.spectral_denominator
        next_states = (joined[..., -model.history_samples:], local, glob, frames[..., 128:])
        if model.waveform_basis:
            wave_input = joined[..., -256:].reshape(1, 1, 512)
            basis = torch.tanh(model.waveform_encode(wave_input))
            gates = torch.sigmoid(model.waveform_gates(context))
            waveform = model.waveform_decode(basis * gates).reshape(1, 1, 2, 256) * model.synthesis.window
            emitted = emitted + (waveform[..., :128] + states[4])
            next_states += (waveform[..., 128:],)
        return emitted, *next_states


def make_export_copy(model):
    interface(model)
    require(all(v.dtype == torch.float32 and bool(torch.isfinite(v).all()) for v in model.state_dict().values()),
            "Export requires finite FP32 model tensors")
    copied = copy.deepcopy(model).cpu().eval().requires_grad_(False)
    replace_rmsnorm_layers(copied)
    for name in ("local", "global_memory"):
        setattr(copied.backbone, name, nn.ModuleList(OneFrameGRUForONNX(m) for m in getattr(copied.backbone, name)))
    return _SpecialistStreamingWrapper(copied)


def export_fp32(model, destination):
    """Export through a separate copy while preserving source state and RNG."""
    import onnx

    contract = interface(model)
    destination = Path(destination)
    require(not destination.exists(), "Use a new path for the experimental graph")
    destination.parent.mkdir(parents=True, exist_ok=True)
    before = model_state_sha256(model)
    devices = sorted({v.device.index for v in model.parameters() if v.is_cuda})
    with torch.random.fork_rng(devices=devices), torch.no_grad():
        wrapper = make_export_copy(model)
        inputs = (torch.zeros(1, 2, 128), *wrapper.model.initial_state(1))
        torch.onnx.export(wrapper, inputs, str(destination), input_names=contract["input_names"],
            output_names=contract["output_names"], opset_version=17, do_constant_folding=True, dynamo=False)
    require(model_state_sha256(model) == before, "Export changed source weights")
    graph = onnx.load(str(destination))
    onnx.checker.check_model(graph, full_check=True)
    metadata = {"schema": "stemgenrt-experimental-specialist-onnx-v1", "architecture": model.architecture_metadata,
                "interface": contract, "weight_sha256": before, "quality_measured": False,
                "native_host_qualified": False, "precision": "fp32", "provenance": copy.deepcopy(model.provenance)}
    onnx.helper.set_model_props(graph, {"stemgenrt.experimental": json.dumps(metadata, sort_keys=True)})
    onnx.save(graph, str(destination))
    return {**metadata, "onnx_sha256": sha(destination), "bytes": destination.stat().st_size}
