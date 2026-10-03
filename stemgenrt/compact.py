"""Experimental causal spectral separator with explicit output sources.

The vocal variant computes one stereo output. The joint control uses the same
backbone and four independent masks. Neither variant redistributes errors
between output sources. Training support must select matching reference stems.
These are untrained research architectures, not replacements for StemgenRT58.
"""
from __future__ import annotations

from typing import NamedTuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ._model.dsp import AsymmetricSynthesis, asymmetric_windows, require
from .model import ContextOutput, ModelOutput

VERSION = "compact-causal-complex-mask-gru-v1"
SOURCE_ORDER = ("drums", "bass", "vocals", "other")


class CompactState(NamedTuple):
    audio_history: Tensor
    recurrent_hidden: Tensor
    spectral_numerator_tail: Tensor

    def detached(self):
        return type(self)(*(value.detach() for value in self))


class CompactSeparator(nn.Module):
    """1024-point causal analysis, 256-sample synthesis, 128-sample hop.

    The renderer has 128 samples of alignment; the host's 128-sample hop
    accumulation gives the same 256-sample (5.805 ms) delay as the baseline.
    Current-frame normalization includes stereo channels and bins, never time.
    A unidirectional GRU supplies past context without future audio.
    """

    state_type = CompactState
    state_names = CompactState._fields

    def __init__(self, *, sources=("vocals",), hidden_size=384, layers=3):
        super().__init__()
        require(tuple(sources) in (("vocals",), SOURCE_ORDER), "Use the vocal candidate or matched joint control")
        require(type(hidden_size) is int and 16 <= hidden_size <= 1024
                and type(layers) is int and 1 <= layers <= 4, "Invalid compact backbone geometry")
        self.sources = tuple(sources)
        self.hidden_size, self.layers = hidden_size, layers
        self.training_precision = "fp32"
        self.provenance = {"architecture_version": VERSION, "training_updates": 0,
                           "quality_measured": False, "initialization": "scratch"}
        analysis, synthesis = asymmetric_windows()
        self.register_buffer("analysis_window", analysis)
        self.synthesis = AsymmetricSynthesis(analysis, synthesis)
        self.input_projection = nn.Linear(2 * 513 * 3, hidden_size)
        self.input_norm = nn.RMSNorm(hidden_size, eps=1e-6)
        self.recurrent = nn.GRU(hidden_size, hidden_size, layers, batch_first=True)
        self.output_norm = nn.RMSNorm(hidden_size, eps=1e-6)
        # Build the large head without advancing RNG by a source-count-dependent
        # amount. All source slices start identically; the vocal slice and
        # backbone are bit-identical between matched seed initializations.
        with torch.random.fork_rng(devices=[]):
            self.mask_head = nn.Linear(hidden_size, len(self.sources) * 2 * 513 * 2)
        with torch.no_grad():
            weight = torch.randn(2 * 513 * 2, hidden_size) * (1e-3 / hidden_size ** .5)
            self.mask_head.weight.copy_(weight.repeat(len(self.sources), 1))
            bias = torch.zeros(2, 513, 2)
            bias[..., 0] = .25
            self.mask_head.bias.copy_(bias.flatten().repeat(len(self.sources)))

    @property
    def parameter_tensor_count(self):
        return sum(1 for _ in self.parameters())

    @property
    def architecture_metadata(self):
        return {"version": VERSION, "source_order": list(self.sources),
                "hidden_size": self.hidden_size, "layers": self.layers,
                "sample_rate": 44100, "feature_n_fft": 1024, "carrier_n_fft": 1024,
                "spectral_mask_bins": 513, "spectral_output_crop": [768, 1024],
                "synthesis_frame_samples": 256, "hop_samples": 128,
                "feature_history_samples": 896, "graph_alignment_samples": 128,
                "host_queue_samples": 128, "intended_total_latency_samples": 256,
                "host_queue_implemented_in_this_module": False,
                "future_callbacks_beyond_received_input": 0, "flush_hops": 1,
                "state_names": list(self.state_names),
                "output_policy": "independent complex masks; no source correction",
                "feature_normalization": "current-frame stereo/bin RMS; no temporal statistics",
                "precision": "FP32 FFT, normalization, synthesis and persistent state; optional BF16 learned operations",
                "native_host_qualified": False}

    def initial_state(self, batch_size, *, device=None):
        require(type(batch_size) is int and batch_size > 0, "Invalid batch size")
        device = self.analysis_window.device if device is None else device
        def zeros(*shape):
            return torch.zeros(*shape, device=device, dtype=torch.float32)
        return self.state_type(zeros(batch_size, 2, 896),
                               zeros(self.layers, batch_size, self.hidden_size),
                               zeros(batch_size, len(self.sources), 2, 128))

    def _validate_state(self, state, batch, device):
        require(type(state) is self.state_type, "Wrong compact streaming state type")
        shapes = ((batch, 2, 896), (self.layers, batch, self.hidden_size),
                  (batch, len(self.sources), 2, 128))
        require(all(v.shape == shape and v.dtype == torch.float32 and v.device == device
                    for v, shape in zip(state, shapes)), "Invalid compact state shape, dtype or device")

    def render(self, audio, state=None):
        require(audio.ndim == 3 and audio.shape[1] == 2 and audio.shape[-1] > 0
                and audio.shape[-1] % 128 == 0 and audio.dtype == torch.float32,
                "Require FP32 stereo audio with a positive multiple of 128 samples")
        require(audio.device == self.analysis_window.device, "Audio and model devices differ")
        state = self.initial_state(audio.shape[0]) if state is None else state
        self._validate_state(state, audio.shape[0], audio.device)
        bf16 = self.training and self.training_precision == "bf16"
        require(self.training_precision in ("fp32", "bf16") and (not bf16 or audio.is_cuda),
                "BF16 learned operations require CUDA")
        with torch.autocast(audio.device.type, enabled=False):
            joined = torch.cat((state.audio_history, audio), -1)
            spectrum = torch.fft.rfft(joined.unfold(-1, 1024, 128) * self.analysis_window)
            power = spectrum.real.square() + spectrum.imag.square()
            scale = power.mean((1, 3), keepdim=True).clamp_min(1e-8).sqrt()
            components = torch.view_as_real(spectrum / scale).permute(0, 2, 1, 3, 4).flatten(2)
            magnitude = torch.log1p((power + 1e-12).sqrt() / scale).permute(0, 2, 1, 3).flatten(2)
            features = torch.cat((components, magnitude), -1)
            with torch.autocast(audio.device.type, dtype=torch.bfloat16, enabled=bf16):
                projected = self.input_projection(features)
            embedded = self.input_norm(F.silu(projected.float()))
            with torch.autocast(audio.device.type, dtype=torch.bfloat16, enabled=bf16):
                recurrent, hidden = self.recurrent(
                    embedded.to(torch.bfloat16) if bf16 else embedded,
                    state.recurrent_hidden.to(torch.bfloat16) if bf16 else state.recurrent_hidden)
            representation = self.output_norm(embedded + recurrent.float())
            with torch.autocast(audio.device.type, dtype=torch.bfloat16, enabled=bf16):
                logits = self.mask_head(representation)
            masks = logits.float().reshape(audio.shape[0], -1, len(self.sources), 2, 513, 2)
            masks = torch.view_as_complex(masks.contiguous()).permute(0, 2, 3, 1, 4)
            # DC/Nyquist imaginary values cannot contribute to a real waveform.
            # irfft implements that endpoint convention explicitly.
            separated = spectrum[:, None] * masks
            estimates, tail = self.synthesis.spectral(separated, state.spectral_numerator_tail)
            next_state = self.state_type(joined[..., -896:].clone(), hidden.float(), tail)
            physical = torch.cat((state.audio_history[..., -128:], audio), -1)[..., :audio.shape[-1]]
            return ModelOutput(estimates, estimates, estimates, torch.zeros_like(estimates),
                               physical, next_state, estimates)

    @torch.no_grad()
    def warm_state(self, audio, state=None):
        return self.render(audio, state).state

    def forward(self, audio, state=None, *, return_raw=False):
        result = self.render(audio, state)
        return (result.raw if return_raw else result.deployed), result.state

    def forward_chunk(self, audio, state=None, *, return_raw=False):
        require(audio.shape[-1] == 128, "Literal input requires exactly 128 samples")
        return self.forward(audio, state, return_raw=return_raw)

    def flush(self, state, *, return_raw=False):
        require(type(state) is self.state_type, "Flush requires a compact state")
        return self.forward_chunk(state.audio_history.new_zeros(state.audio_history.shape[0], 2, 128),
                                  state, return_raw=return_raw)

    def render_scored_context(self, mixture, *, warmup_samples, carry_state):
        require(carry_state is True and type(warmup_samples) is int and warmup_samples > 0
                and warmup_samples % 128 == 0 and mixture.ndim == 3
                and mixture.shape[-1] > warmup_samples, "Invalid compact training context")
        state = self.warm_state(mixture[..., :warmup_samples]).detached()
        require(all(not v.requires_grad and v.grad_fn is None for v in state), "Warmup retained gradients")
        score = mixture[..., warmup_samples:]
        count, padding = score.shape[-1], (-score.shape[-1]) % 128
        result = self.render(F.pad(score, (0, padding + 128)), state)
        raw, deployed, physical = (v[..., 128:128 + count] for v in
                                   (result.raw, result.deployed, result.delayed_mixture))
        require(raw.shape == deployed.shape == (mixture.shape[0], len(self.sources), 2, count)
                and torch.equal(physical, score), "Compact training context lost physical alignment")
        return ContextOutput(raw, deployed, physical, warmup_samples, count, True, True,
                             (count + padding) // 128, 1)

    def compute_budget(self):
        """Dense learned MACs only; excludes FFT, normalization and pointwise ops."""
        width = self.hidden_size
        per_hop = 3078 * width + self.layers * 6 * width * width + width * len(self.sources) * 2052
        return {"parameters": sum(v.numel() for v in self.parameters()),
                "parameter_tensors": self.parameter_tensor_count, "dense_macs_per_hop": per_hop,
                "dense_macs_per_second": per_hop * 44100 / 128,
                "excludes": ["FFT", "normalization", "pointwise operations", "copies", "bias additions"],
                "measured_runtime": False}
