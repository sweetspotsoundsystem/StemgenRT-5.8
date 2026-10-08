"""Evaluate a vocal candidate while preserving a fixed drums/bass baseline.

This assembly is an evaluation system, not a proposed deployed architecture.
Both models receive exactly the same continuous stream with independent state.
"""
from typing import NamedTuple

import torch
from torch import nn

from .model import ModelOutput
from ._model.dsp import require

SOURCE_ORDER = ("drums", "bass", "vocals", "other")


class ReplacementState(NamedTuple):
    parent: object
    candidate: object


class VocalReplacement(nn.Module):
    hop_samples = 128
    graph_alignment_samples = 128
    flush_hops = 1

    def __init__(self, parent, candidate):
        super().__init__()
        require(parent is not candidate, "Use independent parent and candidate state")
        for model in (parent, candidate):
            info = model.architecture_metadata
            require(info["hop_samples"] == info["graph_alignment_samples"] == 128
                    and info["sample_rate"] == 44100 and info["flush_hops"] == 1,
                    "Replacement models must use the same physical alignment")
            require(not any(module.training for module in model.modules()), "Evaluate frozen models in eval mode")
        require(tuple(parent.architecture_metadata["source_order"]) == SOURCE_ORDER,
                "Parent must provide DBVO estimates")
        candidate_sources = tuple(candidate.architecture_metadata["source_order"])
        require(candidate_sources in (("vocals",), SOURCE_ORDER), "Unsupported candidate source order")
        self.parent, self.candidate = parent, candidate
        self.vocal_index = candidate_sources.index("vocals")
        # Child models already apply their own output calibration. These unity
        # gains describe the assembled signals for source-view diagnostics.
        self.register_buffer("output_source_scales", torch.ones(4))
        self.eval().requires_grad_(False)

    @property
    def architecture_metadata(self):
        return {"version": "frozen-db-candidate-vocal-residual-other-evaluation-v1",
                "source_order": list(SOURCE_ORDER), "hop_samples": 128,
                "graph_alignment_samples": 128, "sample_rate": 44100, "flush_hops": 1,
                "parent": self.parent.architecture_metadata,
                "candidate": self.candidate.architecture_metadata,
                "deployment_candidate": False}

    def initial_state(self, batch_size, *, device=None):
        return ReplacementState(self.parent.initial_state(batch_size, device=device),
                                self.candidate.initial_state(batch_size, device=device))

    def render(self, audio, state=None):
        state = self.initial_state(audio.shape[0], device=audio.device) if state is None else state
        require(type(state) is ReplacementState, "Wrong replacement state")
        parent = self.parent.render(audio, state.parent)
        candidate = self.candidate.render(audio, state.candidate)
        require(torch.equal(parent.delayed_mixture, candidate.delayed_mixture),
                "Parent/candidate mixture coordinates differ")
        drums, bass = parent.deployed[:, 0], parent.deployed[:, 1]
        vocals = candidate.deployed[:, self.vocal_index]
        other = parent.delayed_mixture - ((drums + bass) + vocals)
        estimates = torch.stack((drums, bass, vocals, other), 1)
        return ModelOutput(estimates, estimates, estimates, torch.zeros_like(estimates),
                           parent.delayed_mixture, ReplacementState(parent.state, candidate.state), estimates)
