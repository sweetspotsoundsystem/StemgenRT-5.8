"""Evaluate a bass specialist with fixed drums and vocals.

The wrapper preserves the selected bass waveform and computes physical residual
Other. It is an evaluation assembly, not the finished independent-stem system.
"""
import torch
from torch import nn

from ._model.dsp import require
from .model import ModelOutput
from .replacement import ReplacementState, SOURCE_ORDER


class BassEvaluationHybrid(nn.Module):
    """Preserve parent drums and anchored vocals; replace bass without rescaling.

    This temporary evaluation assembly has no independent trained drums model.
    Its overall/Other scores do not qualify the later complete system.
    """
    hop_samples = graph_alignment_samples = 128
    flush_hops = 1

    def __init__(self, anchored, bass):
        super().__init__()
        require(anchored is not bass, "Use independent model states")
        for model in (anchored, bass):
            info = model.architecture_metadata
            require(info["hop_samples"] == info["graph_alignment_samples"] == 128
                    and info["sample_rate"] == 44100 and info["flush_hops"] == 1,
                    "Hybrid models must share physical alignment")
            require(not any(module.training for module in model.modules()), "Require frozen eval models")
        require(tuple(anchored.architecture_metadata["source_order"]) == SOURCE_ORDER
                and tuple(bass.architecture_metadata["source_order"]) == ("bass",), "Wrong hybrid source order")
        require(not ({id(p) for p in anchored.parameters()} & {id(p) for p in bass.parameters()}),
                "Hybrid child parameters must be independent")
        self.anchored, self.bass = anchored, bass
        self.register_buffer("output_source_scales", torch.ones(4))
        self.eval().requires_grad_(False)

    @property
    def architecture_metadata(self):
        return {"version": "bass-first-evaluation-hybrid-v1", "source_order": list(SOURCE_ORDER),
            "hop_samples": 128, "graph_alignment_samples": 128, "sample_rate": 44100, "flush_hops": 1,
            "anchored": self.anchored.architecture_metadata, "bass": self.bass.architecture_metadata,
            "deployment_candidate": False, "complete_independent_system": False,
            "output_policy": "Frozen parent drums, final bass EMA, fixed vocal EMA, physical residual Other"}

    def initial_state(self, batch_size, *, device=None):
        return ReplacementState(self.anchored.initial_state(batch_size, device=device),
                                self.bass.initial_state(batch_size, device=device))

    def render(self, audio, state=None):
        state = self.initial_state(audio.shape[0], device=audio.device) if state is None else state
        require(type(state) is ReplacementState, "Wrong hybrid streaming state")
        anchored = self.anchored.render(audio, state.parent)
        bass = self.bass.render(audio, state.candidate)
        require(torch.equal(anchored.delayed_mixture, bass.delayed_mixture), "Hybrid mixture coordinates differ")
        drums, vocals = anchored.deployed[:, 0], anchored.deployed[:, 2]
        target = bass.deployed[:, 0]
        other = anchored.delayed_mixture - ((drums + target) + vocals)
        estimates = torch.stack((drums, target, vocals, other), 1)
        return ModelOutput(estimates, estimates, estimates, torch.zeros_like(estimates), anchored.delayed_mixture,
                           ReplacementState(anchored.state, bass.state), estimates)

