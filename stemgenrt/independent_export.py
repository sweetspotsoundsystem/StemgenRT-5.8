"""One complete FP32 ONNX graph for independent DBV and physical residual Other.

All three models retain their own analysis, recurrent memory and synthesis.
This research ABI has 13 states for the default specialist/band architecture;
it needs a matching host and is not a replacement for the released eight-state ABI.
"""
import copy
import json
from pathlib import Path

import torch
from torch import nn

from . import band_export, specialist_export
from .banded import BandSeparator
from .specialist import SpecialistSeparator
from .independent import IndependentStems, SOURCE_ORDER
from ._export.helpers import OneFrameGRUForONNX, model_state_sha256, sha
from ._model.dsp import require
from ._export.fp32 import _RFFT1024


def _exporter(child):
    if type(child) is SpecialistSeparator:
        return specialist_export
    require(type(child) is BandSeparator, "Require a band or specialist child model")
    return band_export


def interface(model):
    require(type(model) is IndependentStems, "Expected three independent stems")
    names, shapes, groups = [], [], {}
    for source in SOURCE_ORDER[:3]:
        child = getattr(model, source)
        contract = _exporter(child).interface(child)
        require(tuple(child.sources) == (source,), "Child source differs")
        groups[source] = {"offset": len(names), "count": len(contract["state_names"])}
        names.extend(source + "_" + name for name in contract["state_names"])
        shapes.extend(contract["input_shapes"][1:])
    return {"state_names": names, "state_groups": groups,
            "input_names": ["audio_chunk", *names],
            "output_names": ["separated_chunk", *("next_" + name for name in names)],
            "input_shapes": [[1, 2, 128], *shapes],
            "output_shapes": [[1, 4, 2, 128], *shapes]}


class _IndependentStreamingWrapper(nn.Module):
    def __init__(self, model, *, batch_carrier_analysis=False, batch_gru_gates=False):
        super().__init__()
        self.contract = interface(model)
        self.children_by_source = nn.ModuleDict({source: _exporter(getattr(model, source)).make_export_copy(
            getattr(model, source)) for source in SOURCE_ORDER[:3]})
        self.physical_history_index = self.contract["state_names"].index("vocals_audio_history")
        self.batch_carrier_analysis = batch_carrier_analysis
        for module in self.children_by_source.modules():
            if isinstance(module, OneFrameGRUForONNX):
                module.batch_gates = batch_gru_gates

    def initial_state(self):
        return tuple(value for child in self.children_by_source.values() for value in child.model.initial_state(1))

    def forward(self, audio_chunk, *states):
        estimates, next_states = [], []
        carriers = [None] * 3
        if self.batch_carrier_analysis:
            # Retain each child's own history and analysis window, including
            # deliberately unequal nonzero public states. Only dispatch and
            # pointwise feature operations are batched; FFT work is unchanged.
            frames = []
            for source in SOURCE_ORDER[:3]:
                child = self.children_by_source[source].model
                history = states[self.contract["state_names"].index(source + "_audio_history")]
                joined = torch.cat((history[..., -896:], audio_chunk), -1)
                frames.append((joined * child.analysis_window).reshape(2, 1024))
            spectra = _RFFT1024.apply(torch.cat(frames, dim=0)).reshape(3, 2, 1, 513, 2)
            features = band_export.carrier_features(spectra)
            carriers = [(spectra[i:i + 1], features[i:i + 1]) for i in range(3)]
        for source, carrier in zip(SOURCE_ORDER[:3], carriers):
            group = self.contract["state_groups"][source]
            start, count = group["offset"], group["count"]
            result = self.children_by_source[source](audio_chunk, *states[start:start + count], carrier=carrier)
            estimates.append(result[0])
            next_states.extend(result[1:])
        # Every child history ends at the previous received input sample.
        # Use its literal last hop, not the newly supplied audio_chunk.
        physical = states[self.physical_history_index][..., -128:]
        other = physical - ((estimates[0][:, 0] + estimates[1][:, 0]) + estimates[2][:, 0])
        return torch.cat((*estimates, other[:, None]), dim=1), *next_states


def make_export_copy(model, *, batch_carrier_analysis=False, batch_gru_gates=False):
    interface(model)
    require(all(value.dtype == torch.float32 and bool(torch.isfinite(value).all())
                for value in model.state_dict().values()), "Export requires finite FP32 model tensors")
    require(type(batch_carrier_analysis) is bool, "batch_carrier_analysis must be bool")
    require(type(batch_gru_gates) is bool, "batch_gru_gates must be bool")
    return _IndependentStreamingWrapper(model, batch_carrier_analysis=batch_carrier_analysis,
        batch_gru_gates=batch_gru_gates).eval().requires_grad_(False)


def export_fp32(model, destination, *, batch_carrier_analysis=False, batch_gru_gates=False):
    """Export all three children and residual, preserving source tensors and RNG."""
    import onnx

    contract = interface(model)
    destination = Path(destination)
    require(not destination.exists(), "Use a new path for the experimental graph")
    destination.parent.mkdir(parents=True, exist_ok=True)
    before = model_state_sha256(model)
    devices = sorted({value.device.index for value in model.parameters() if value.is_cuda})
    with torch.random.fork_rng(devices=devices), torch.no_grad():
        wrapper = make_export_copy(model, batch_carrier_analysis=batch_carrier_analysis,
                                   batch_gru_gates=batch_gru_gates)
        torch.onnx.export(wrapper, (torch.zeros(1, 2, 128), *wrapper.initial_state()), str(destination),
            input_names=contract["input_names"], output_names=contract["output_names"],
            opset_version=17, do_constant_folding=True, dynamo=False)
    require(model_state_sha256(model) == before, "Export changed source weights")
    graph = onnx.load(str(destination))
    onnx.checker.check_model(graph, full_check=True)
    metadata = {"schema": "stemgenrt-experimental-independent-onnx-v1",
        "architecture": model.architecture_metadata, "interface": contract, "weight_sha256": before,
        "precision": "fp32", "quality_measured": False, "native_host_qualified": False,
        "provenance": {source: copy.deepcopy(getattr(model, source).provenance) for source in SOURCE_ORDER[:3]},
        "child_weight_sha256": {source: model_state_sha256(getattr(model, source)) for source in SOURCE_ORDER[:3]},
        "initial_state": "all_zeros", "preroll": "one_invalid_hop", "flush_hops": 1,
        "output_alignment": "previous_input_hop", "residual": "physical_mixture_minus_((drums+bass)+vocals)",
        "input_level": "unchanged", "sample_rate": 44100, "hop_samples": 128,
        "graph_alignment_samples": 128, "state_count": len(contract["state_names"])}
    if batch_carrier_analysis:
        metadata["execution_variant"] = "batched-carrier-analysis-fp32-v1"
        metadata["analysis_policy"] = "independent histories/windows; batched carrier FFT and per-stem features"
    if batch_gru_gates:
        metadata["execution_variant"] = ("batched-carrier-and-gru-gates-fp32-v1" if batch_carrier_analysis
                                         else "batched-gru-gates-fp32-v1")
        metadata["gru_gate_policy"] = "independent reset/update elementwise operations batched; separate input/hidden affine operations and recurrence arithmetic preserved"
    onnx.helper.set_model_props(graph, {"stemgenrt.experimental": json.dumps(metadata, sort_keys=True)})
    onnx.save(graph, str(destination))
    return {**metadata, "onnx_sha256": sha(destination), "bytes": destination.stat().st_size}
