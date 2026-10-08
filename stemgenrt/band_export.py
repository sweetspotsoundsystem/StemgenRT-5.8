"""Experimental FP32 one-hop ONNX export, including FFT and synthesis.

This four-state research graph has its own interface. It is not compatible
with the released plugin's eight-state model and has no runtime qualification.
"""
import copy
import json
from pathlib import Path

import torch
from torch import nn

from .banded import BandSeparator
from ._export.fp32 import _RFFT1024
from ._export.helpers import OneFrameGRUForONNX, replace_rmsnorm_layers, model_state_sha256, sha
from ._model.dsp import require


class _IRFFT1024(torch.autograd.Function):
    @staticmethod
    def forward(ctx, spectrum):
        return torch.fft.irfft(torch.complex(spectrum[..., 0], spectrum[..., 1]), n=1024, dim=1)

    @staticmethod
    def symbolic(graph, spectrum):
        endpoints = torch.ones(513, 2, dtype=torch.float32)
        endpoints[0, 1] = endpoints[-1, 1] = 0
        values = graph.op("Mul", spectrum, graph.op("Constant", value_t=endpoints))
        indices = graph.op("Constant", value_t=torch.arange(511, 0, -1, dtype=torch.int64))
        reflected = graph.op("Gather", values, indices, axis_i=1)
        reflected = graph.op("Mul", reflected, graph.op("Constant", value_t=torch.tensor([1., -1.])))
        full = graph.op("Concat", values, reflected, axis_i=1)
        inverse = graph.op("DFT", full, graph.op("Constant", value_t=torch.tensor(1024, dtype=torch.int64)),
                           axis_i=1, inverse_i=1, onesided_i=0)
        result = graph.op("Gather", inverse, graph.op("Constant", value_t=torch.tensor(0, dtype=torch.int64)), axis_i=2)
        return result.setType(spectrum.type().with_sizes([spectrum.type().sizes()[0], 1024]))


def interface(model):
    require(type(model) is BandSeparator, "Expected the experimental band separator")
    states = model.initial_state(1)
    return {"state_names": list(model.state_names),
            "input_names": ["audio_chunk", *model.state_names],
            "output_names": ["separated_chunk", *("next_" + n for n in model.state_names)],
            "input_shapes": [[1, 2, 128], *[list(v.shape) for v in states]],
            "output_shapes": [[1, len(model.sources), 2, 128], *[list(v.shape) for v in states]]}


class _BandStreamingWrapper(nn.Module):
    def __init__(self, copied):
        super().__init__()
        self.model = copied

    def forward(self, audio_chunk, audio_history, local_hidden, global_hidden, spectral_numerator_tail):
        model = self.model
        joined = torch.cat((audio_history, audio_chunk), -1)
        spectrum = _RFFT1024.apply((joined * model.analysis_window).reshape(2, 1024))
        spectrum = spectrum.reshape(1, 2, 1, 513, 2)
        power = spectrum[..., 0].square() + spectrum[..., 1].square()
        scale = power.mean((1, 3), keepdim=True).clamp_min(1e-8).sqrt()
        normalized = spectrum / scale.unsqueeze(-1)
        features = torch.cat((normalized, torch.log1p((power + 1e-12).sqrt() / scale).unsqueeze(-1)), -1)
        features = features.permute(0, 2, 1, 3, 4)
        masks, local, glob = model.backbone(features, local_hidden, global_hidden)
        masks = masks.permute(0, 2, 3, 1, 4, 5)
        real, imag = spectrum[:, None, ..., 0], spectrum[:, None, ..., 1]
        separated = torch.stack((real * masks[..., 0] - imag * masks[..., 1],
                                 real * masks[..., 1] + imag * masks[..., 0]), -1)
        frames = _IRFFT1024.apply(separated.reshape(2 * len(model.sources), 513, 2))
        frames = frames[..., 768:].reshape(1, len(model.sources), 2, 256) * model.synthesis.spectral_window
        emitted = (frames[..., :128] + spectral_numerator_tail) / model.synthesis.spectral_denominator
        return emitted, joined[..., -896:], local, glob, frames[..., 128:]


def make_export_copy(model):
    interface(model)
    require(all(v.dtype == torch.float32 and bool(torch.isfinite(v).all()) for v in model.state_dict().values()),
            "Export requires finite FP32 model tensors")
    copied = copy.deepcopy(model).cpu().eval().requires_grad_(False)
    replace_rmsnorm_layers(copied)
    for name in ("local", "global_memory"):
        setattr(copied.backbone, name, nn.ModuleList(OneFrameGRUForONNX(m) for m in getattr(copied.backbone, name)))
    return _BandStreamingWrapper(copied)


def export_fp32(model, destination):
    """Export a separate CPU copy without changing weights, mode, flags or RNG."""
    import onnx

    contract = interface(model)
    destination = Path(destination)
    require(not destination.exists(), "Use a new path for the experimental graph")
    destination.parent.mkdir(parents=True, exist_ok=True)
    before = model_state_sha256(model)
    # Forking includes accelerator RNG when the source model lives on CUDA.
    devices = sorted({v.device.index for v in model.parameters() if v.is_cuda})
    with torch.random.fork_rng(devices=devices), torch.no_grad():
        wrapper = make_export_copy(model)
        inputs = (torch.zeros(1, 2, 128), *wrapper.model.initial_state(1))
        torch.onnx.export(wrapper, inputs, str(destination), input_names=contract["input_names"],
            output_names=contract["output_names"], opset_version=17, do_constant_folding=True,
            dynamo=False)
    require(model_state_sha256(model) == before, "Export changed source weights")
    graph = onnx.load(str(destination))
    onnx.checker.check_model(graph, full_check=True)
    metadata = {"schema": "stemgenrt-experimental-band-onnx-v1", "architecture": model.architecture_metadata,
                "interface": contract, "weight_sha256": before, "quality_measured": False,
                "native_host_qualified": False, "precision": "fp32", "provenance": copy.deepcopy(model.provenance)}
    onnx.helper.set_model_props(graph, {"stemgenrt.experimental": json.dumps(metadata, sort_keys=True)})
    onnx.save(graph, str(destination))
    return {**metadata, "onnx_sha256": sha(destination), "bytes": destination.stat().st_size}
