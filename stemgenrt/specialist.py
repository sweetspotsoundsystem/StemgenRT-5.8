"""Independent causal bass and drums research models with complete source identity.

Bass uses finer low-frequency bands and optional longer past-only magnitude
features. Drums adds an optional short waveform residual. Both synthesize on
the existing 128-sample alignment and 128-sample hop; neither emits other stems.
Untrained architectures and arithmetic budgets do not imply quality or native
runtime qualification. The released model and vocal checkpoint schema remain
separate.
"""
from __future__ import annotations

from typing import NamedTuple

import torch
from torch import Tensor, nn

from ._model.bands import BAND_EDGES, CausalBandBackbone, GroupedAffine
from ._model.dsp import AsymmetricSynthesis, asymmetric_windows, require
from .banded import BandSeparator, BandState
from .model import ModelOutput

VERSION = "causal-independent-stem-specialists-v1"
BASS_BAND_EDGES = (0, 1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 160, 256, 384, 513)


class WaveformBandState(NamedTuple):
    audio_history: Tensor
    local_hidden: Tensor
    global_hidden: Tensor
    spectral_numerator_tail: Tensor
    waveform_numerator_tail: Tensor

    def detached(self):
        return type(self)(*(value.detach() for value in self))


class SpecialistBackbone(CausalBandBackbone):
    def __init__(self, *, band_width, global_width, layers, edges, feature_n_fft):
        super().__init__(sources=1, band_width=band_width, global_width=global_width,
                         layers=layers, edges=edges)
        self.feature_n_fft = feature_n_fft
        self.long_edges = tuple(min(edge * 4, 2049) for edge in self.edges)
        self.long_groups = []
        self.long_encoders = nn.ModuleList()
        if feature_n_fft == 4096:
            for i, (left, right) in enumerate(zip(self.long_edges, self.long_edges[1:])):
                width = right - left
                if self.long_groups and self.long_groups[-1][2] == width:
                    first, _, _ = self.long_groups[-1]
                    self.long_groups[-1] = first, i + 1, width
                else:
                    self.long_groups.append((i, i + 1, width))
            for first, last, width in self.long_groups:
                self.long_encoders.append(GroupedAffine(
                    [nn.Linear(2 * width, band_width, bias=False) for _ in range(first, last)]))

    def forward(self, features, local_hidden, global_hidden, long_features=None):
        encoded = self.encode_features(features)
        if self.feature_n_fft == 4096:
            require(long_features is not None, "Bass long analysis features are required")
            batch, frames = features.shape[:2]
            values = []
            for module, (first, last, width) in zip(self.long_encoders, self.long_groups):
                chunk = long_features[..., self.long_edges[first]:self.long_edges[last]]
                chunk = chunk.reshape(batch, frames, 2, last - first, width)
                values.append(module(chunk.permute(0, 1, 3, 2, 4).flatten(3)))
            encoded = encoded + torch.cat(values, dim=2)
        else:
            require(long_features is None, "Short analysis cannot consume longer features")
        bands, local, glob, context = self.temporal_context(encoded, local_hidden, global_hidden)
        return self.decode_masks(bands), local, glob, context

    def dense_macs(self):
        extra = 2 * 2049 * self.width if self.feature_n_fft == 4096 else 0
        return super().dense_macs() + extra


class SpecialistSeparator(BandSeparator):
    """One selected stem with past-only context and no source redistribution.

    Optional geometry overrides support technical controls, not automatic model
    selection. Defaults are bass width 80/global 160 with 4096-point features,
    or drums width 96/global 192 with 256 learned waveform basis channels.
    """

    hop_samples = 128
    graph_alignment_samples = 128

    def __init__(self, *, source, band_width=None, global_width=None, layers=2,
                 feature_n_fft=None, waveform_basis=None):
        nn.Module.__init__(self)
        require(source in ("bass", "drums"), "Choose the bass or drums specialist")
        bass = source == "bass"
        band_width = (80 if bass else 96) if band_width is None else band_width
        global_width = (160 if bass else 192) if global_width is None else global_width
        feature_n_fft = (4096 if bass else 1024) if feature_n_fft is None else feature_n_fft
        waveform_basis = (0 if bass else 256) if waveform_basis is None else waveform_basis
        require(type(feature_n_fft) is int and feature_n_fft in (1024, 4096)
                and (bass or feature_n_fft == 1024), "Long analysis is a bass-only feature")
        require(type(waveform_basis) is int and (waveform_basis == 0 if bass else 0 <= waveform_basis <= 512),
                "Waveform residual is supported only for drums, with at most 512 basis channels")
        self.sources = (source,)
        self.band_width, self.global_width, self.layers = band_width, global_width, layers
        self.feature_n_fft, self.waveform_basis = feature_n_fft, waveform_basis
        self.history_samples = feature_n_fft - self.hop_samples
        self.state_type = WaveformBandState if waveform_basis else BandState
        self.state_names = self.state_type._fields
        self.training_precision = "fp32"
        self.provenance = {"architecture_version": VERSION, "training_updates": 0,
                           "quality_measured": False, "initialization": "scratch"}
        analysis, synthesis = asymmetric_windows()
        self.register_buffer("analysis_window", analysis)
        if feature_n_fft == 4096:
            self.register_buffer("long_analysis_window", torch.hann_window(4096, dtype=torch.float32))
        self.synthesis = AsymmetricSynthesis(analysis, synthesis)
        self.backbone = SpecialistBackbone(band_width=band_width, global_width=global_width,
            layers=layers, edges=BASS_BAND_EDGES if bass else BAND_EDGES, feature_n_fft=feature_n_fft)
        if waveform_basis:
            self.waveform_encode = nn.Linear(512, waveform_basis, bias=False)
            self.waveform_gates = nn.Linear(global_width, waveform_basis)
            self.waveform_decode = nn.Linear(waveform_basis, 512, bias=False)
            with torch.no_grad():
                self.waveform_decode.weight.mul_(.01)

    @property
    def architecture_metadata(self):
        return {"version": VERSION, "source_order": list(self.sources),
                "band_width": self.band_width, "global_width": self.global_width, "layers": self.layers,
                "band_edges": list(self.backbone.edges), "band_packing": "independent equal-width groups",
                "sample_rate": 44100, "feature_n_fft": self.feature_n_fft, "carrier_n_fft": 1024,
                "long_feature_band_edges": list(self.backbone.long_edges) if self.feature_n_fft == 4096 else [],
                "long_feature_window": "trailing periodic Hann" if self.feature_n_fft == 4096 else None,
                "waveform_basis": self.waveform_basis, "waveform_frame_samples": 256 if self.waveform_basis else 0,
                "spectral_mask_bins": 513, "spectral_output_crop": [768, 1024],
                "synthesis_frame_samples": 256, "hop_samples": 128,
                "feature_history_samples": self.history_samples, "graph_alignment_samples": 128,
                "host_queue_samples": 128, "intended_total_latency_samples": 256,
                "host_queue_implemented_in_this_module": False,
                "future_callbacks_beyond_received_input": 0, "flush_hops": 1,
                "state_names": list(self.state_names),
                "output_policy": "one independent complex mask with optional waveform residual; no source correction",
                "feature_normalization": "separate current-frame stereo/bin RMS for each FFT; no temporal statistics",
                "precision": "FP32 throughout; BF16 training is unsupported", "native_host_qualified": False}

    def initial_state(self, batch_size, *, device=None):
        require(type(batch_size) is int and batch_size > 0, "Invalid batch size")
        device = self.analysis_window.device if device is None else device
        def zeros(*shape):
            return torch.zeros(*shape, device=device, dtype=torch.float32)
        state = (zeros(batch_size, 2, self.history_samples),
                 zeros(self.layers, batch_size * self.backbone.band_count, self.band_width),
                 zeros(self.layers, batch_size, self.global_width), zeros(batch_size, 1, 2, 128))
        if self.waveform_basis:
            state += (zeros(batch_size, 1, 2, 128),)
        return self.state_type(*state)

    def _validate_state(self, state, batch, device):
        require(type(state) is self.state_type, "Wrong specialist streaming state type")
        shapes = ((batch, 2, self.history_samples),
                  (self.layers, batch * self.backbone.band_count, self.band_width),
                  (self.layers, batch, self.global_width), (batch, 1, 2, 128))
        if self.waveform_basis:
            shapes += ((batch, 1, 2, 128),)
        require(all(v.shape == shape and v.dtype == torch.float32 and v.device == device
                    for v, shape in zip(state, shapes)), "Invalid specialist state shape, dtype or device")

    def render(self, audio, state=None):
        require(audio.ndim == 3 and audio.shape[1] == 2 and audio.shape[-1] > 0
                and audio.shape[-1] % 128 == 0 and audio.dtype == torch.float32,
                "Require FP32 stereo audio with a positive multiple of 128 samples")
        require(audio.device == self.analysis_window.device, "Audio and model devices differ")
        state = self.initial_state(audio.shape[0]) if state is None else state
        self._validate_state(state, audio.shape[0], audio.device)
        require(self.training_precision == "fp32", "Specialist experiments require FP32 training")
        with torch.autocast(audio.device.type, enabled=False):
            joined = torch.cat((state.audio_history, audio), -1)
            # The short carrier ends at the same received sample as long analysis.
            frames = joined[..., self.history_samples - 896:].unfold(-1, 1024, 128)
            spectrum = torch.fft.rfft(frames * self.analysis_window)
            power = spectrum.real.square() + spectrum.imag.square()
            scale = power.mean((1, 3), keepdim=True).clamp_min(1e-8).sqrt()
            normalized = spectrum / scale
            features = torch.stack((normalized.real, normalized.imag,
                torch.log1p((power + 1e-12).sqrt() / scale)), -1).permute(0, 2, 1, 3, 4)
            long_features = None
            if self.feature_n_fft == 4096:
                long_spectrum = torch.fft.rfft(joined.unfold(-1, 4096, 128) * self.long_analysis_window)
                long_power = long_spectrum.real.square() + long_spectrum.imag.square()
                long_scale = long_power.mean((1, 3), keepdim=True).clamp_min(1e-8).sqrt()
                long_features = torch.log1p((long_power + 1e-12).sqrt() / long_scale).permute(0, 2, 1, 3)
            masks, local, glob, context = self.backbone(features, state.local_hidden, state.global_hidden,
                                                       long_features)
            masks = torch.view_as_complex(masks.contiguous()).permute(0, 2, 3, 1, 4)
            spectral, tail = self.synthesis.spectral(spectrum[:, None] * masks, state.spectral_numerator_tail)
            estimates, waveform = spectral, torch.zeros_like(spectral)
            next_values = (joined[..., -self.history_samples:].clone(), local, glob, tail)
            if self.waveform_basis:
                wave_input = frames[..., -256:].permute(0, 2, 1, 3).flatten(2)
                basis = torch.tanh(self.waveform_encode(wave_input))
                gates = torch.sigmoid(self.waveform_gates(context))
                decoded = self.waveform_decode(basis * gates).reshape(audio.shape[0], -1, 2, 256).permute(0, 2, 1, 3)
                waveform, wave_tail = self.synthesis.waveform(decoded[:, None], state.waveform_numerator_tail)
                estimates = spectral + waveform
                next_values += (wave_tail,)
            physical = torch.cat((state.audio_history[..., -128:], audio), -1)[..., :audio.shape[-1]]
            return ModelOutput(estimates, estimates, spectral, waveform, physical,
                               self.state_type(*next_values), estimates)

    def compute_budget(self):
        result = super().compute_budget()
        result["dense_macs_per_hop"] += self.waveform_basis * (1024 + self.global_width)
        result["dense_macs_per_second"] = result["dense_macs_per_hop"] * 44100 / 128
        return result
