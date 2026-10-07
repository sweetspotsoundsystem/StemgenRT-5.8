"""Evaluate three independent stems with one physical residual Other.

Each module owns one output and its own causal state. No trunk, mask, hidden
state or learned output correction is shared between instruments. This assembly
does not alter or qualify the released plugin.
"""
from typing import NamedTuple

import torch
from torch import nn

from ._model.dsp import require
from .model import ModelOutput

SOURCE_ORDER = ("drums", "bass", "vocals", "other")


class IndependentState(NamedTuple):
    drums: object
    bass: object
    vocals: object


class IndependentStems(nn.Module):
    hop_samples = 128
    graph_alignment_samples = 128
    flush_hops = 1

    def __init__(self, *, drums, bass, vocals):
        super().__init__()
        children = (drums, bass, vocals)
        require(len({id(m) for m in children}) == 3, "Require three independent models")
        parameter_ids = [set(map(id, m.parameters())) for m in children]
        require(not any(parameter_ids[i] & parameter_ids[j] for i in range(3) for j in range(i)),
                "Independent models must not share parameters")
        for source, model in zip(SOURCE_ORDER[:3], children):
            info = model.architecture_metadata
            require(tuple(info["source_order"]) == (source,), "Each model must emit exactly its named source")
            require(info["hop_samples"] == info["graph_alignment_samples"] == 128
                    and info["sample_rate"] == 44100 and info["flush_hops"] == 1,
                    "Independent models must have the same physical alignment")
            require(not any(module.training for module in model.modules()), "Evaluate frozen models in eval mode")
        self.drums, self.bass, self.vocals = children
        self.register_buffer("output_source_scales", torch.ones(4))
        self.eval().requires_grad_(False)

    @property
    def architecture_metadata(self):
        return {"version": "independent-dbv-physical-residual-other-v1", "source_order": list(SOURCE_ORDER),
                "hop_samples": 128, "graph_alignment_samples": 128, "sample_rate": 44100,
                "flush_hops": 1, "intended_total_latency_samples": 256,
                "models": {name: getattr(self, name).architecture_metadata for name in SOURCE_ORDER[:3]},
                "native_host_qualified": False, "output_policy": "independent DBV; Other is mixture minus DBV"}

    def initial_state(self, batch_size, *, device=None):
        return IndependentState(*(getattr(self, name).initial_state(batch_size, device=device)
                                  for name in SOURCE_ORDER[:3]))

    def render(self, audio, state=None):
        state = self.initial_state(audio.shape[0], device=audio.device) if state is None else state
        require(type(state) is IndependentState, "Wrong independent model state")
        results = [getattr(self, name).render(audio, child_state)
                   for name, child_state in zip(SOURCE_ORDER[:3], state)]
        physical = results[0].delayed_mixture
        require(all(torch.equal(result.delayed_mixture, physical) for result in results[1:]),
                "Independent models disagree on physical mixture coordinates")
        drums, bass, vocals = (r.deployed[:, 0] for r in results)
        other = physical - ((drums + bass) + vocals)
        estimates = torch.stack((drums, bass, vocals, other), 1)
        return ModelOutput(estimates, estimates, estimates, torch.zeros_like(estimates), physical,
                           IndependentState(*(r.state for r in results)), estimates)

    def compute_budget(self):
        children = {name: getattr(self, name).compute_budget() for name in SOURCE_ORDER[:3]}
        return {"models": children,
                **{key: sum(row[key] for row in children.values()) for key in
                   ("parameters", "dense_macs_per_hop", "dense_macs_per_second", "persistent_state_bytes_fp32")},
                "excludes": ["FFT", "normalization", "pointwise operations", "copies", "bias additions"],
                "measured_runtime": False}
