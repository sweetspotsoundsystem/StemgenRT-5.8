"""The current eight-state StemgenRT-5.8 model and detached training context.

Constructing a model initializes untrained parameters. Checkpoints and ONNX
inference are separate APIs; no weights are downloaded or loaded on import.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import NamedTuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ._model.dsp import (
    BASIS, CHANNELS, CROP_START, EMBED, FEATURE_HISTORY, FEATURE_SAMPLES, HOP,
    KEY, MASK_BINS, PUBLIC_FUSION_SCALE, SOURCE_ORDER, SOURCES, SYNTHESIS_SAMPLES,
    VALUE, AsymmetricSynthesis, MagnitudeEncoder, asymmetric_windows,
    corrected_estimates, cross_component_correction, require,
)

__all__ = ["StemgenRT58", "StreamingState", "render_scored_context"]
# Retain the native checkpoint schema identifier for the released architecture.
VERSION = "latency58-attention-private-branch-gru500-zero-projections-v1"
PRECISION_POLICY = "latency58-asymmetric-bf16-learned-fp32-synthesis-state-v1"

# These checkpoint/export schema values describe the trained architecture.
_ARCHITECTURE = {'version': 'latency58-attention-private-branch-gru500-zero-projections-v1',
 'state_family': 'latency58-attention-private-branch-gru500-zero-projections-v1',
 'sample_rate': 44100,
 'source_order': ['drums', 'bass', 'vocals', 'other'],
 'feature_n_fft': 1024,
 'carrier_n_fft': 1024,
 'spectral_mask_bins': 513,
 'spectral_output_crop': [768, 1024],
 'synthesis_frame_samples': 256,
 'hop_samples': 128,
 'feature_history_samples': 896,
 'graph_alignment_samples': 128,
 'host_queue_samples': 128,
 'intended_total_latency_samples': 256,
 'host_queue_implemented_in_this_module': False,
 'future_callbacks_beyond_received_input': 0,
 'samplewise_latest_input_minus_output': '255-p for output callback sample p=0..127',
 'state_names': ['audio_history',
                 'fusion_hidden',
                 'spectral_numerator_tail',
                 'waveform_tail',
                 'attention_keys',
                 'attention_values',
                 'spec_memory_hidden',
                 'waveform_memory_hidden'],
 'flush_hops': 1,
 'public_fusion_state_scale': 3.814697265625e-06,
 'precision': 'float32',
 'native_host_qualified': False,
 'feature_window': 'Wang2021 K1024 M128 d0 asymmetric analysis',
 'spectral_synthesis_window': 'matched pair with Hann256 analysis-synthesis product',
 'waveform_synthesis_window': 'unchanged periodic Hann256',
 'carrier_forward_fft_reused_from_features': True,
 'spectral_denominator_rule': 'sum of the two analysis-synthesis overlap products',
 'encoder_initialization': 'FP64 effective-kernel window transfer, rounded once to FP32',
 'encoder_transfer_bf16_equivalence_claimed': False,
 'recurrent_cadence_changed_from_hann_hop128_parent': False,
 'output_policy_version': 'latency58-c204-fixed-residual-sixteenth-v1',
 'raw_output_semantics': 'native DBV plus fixed discrepancy correction; native Other',
 'component_semantics': 'spectral and waveform are before discrepancy correction',
 'fixed_residual_share': 0.0625,
 'extra_stream_state_tensors': 0,
 'extra_audio_buffering_samples': 0,
 'magnitude_features': 'log1p(sqrt(real^2+imag^2+1e-12)/sqrt(max(frame_mean_power,1e-8)))',
 'magnitude_normalization_axes': 'current-frame stereo and frequency only',
 'additional_neural_parameters': 4952632,
 'additional_audio_buffering_samples': 0,
 'additional_state_tensors': 4,
 'phase_head_rank': 64,
 'phase_head_parameters': 293632,
 'phase_correction': 'i*carrier times source-centered real coefficients at interior bins',
 'phase_correction_coordinate': 'before inherited source calibration and discrepancy correction',
 'phase_endpoint_coefficients': 'exact zero at DC and Nyquist',
 'additional_fft_transforms': 0,
 'fusion_refinement_rank': 128,
 'fusion_refinement_parameters': 256000,
 'fusion_refinement': 'fused + expand(SiLU(reduce(fused))) before branch normalization',
 'fusion_refinement_input_features': 1000,
 'temporal_attention_parameters': 384000,
 'temporal_attention_window_frames': 32,
 'temporal_attention_key_channels': 64,
 'temporal_attention_value_channels': 128,
 'temporal_attention': 'refined fused features plus causal softmax(Q K.T / sqrt(64)) V output',
 'temporal_attention_history_samples': 3968,
 'additional_state_elements_per_stream': 6952,
 'branch_memory_parameters': 3506000,
 'branch_memory_channels': 500,
 'branch_memory_layers': 1,
 'branch_memory_input': 'normalized refined branch features before attention',
 'branch_memory_output': 'zero-initialized linear residual after branch normalization',
 'branch_memory_added_state_tensors': 2,
 'branch_memory_added_state_elements_per_stream': 1000}


class StreamingState(NamedTuple):
    audio_history: torch.Tensor
    fusion_hidden: torch.Tensor
    spectral_numerator_tail: torch.Tensor
    waveform_tail: torch.Tensor
    attention_keys: torch.Tensor
    attention_values: torch.Tensor
    spec_memory_hidden: torch.Tensor
    waveform_memory_hidden: torch.Tensor

    def detached(self):
        return type(self)(*(value.detach() for value in self))

@dataclass(frozen=True)
class ModelOutput:
    raw: Tensor
    deployed: Tensor
    spectral: Tensor
    waveform: Tensor
    delayed_mixture: Tensor
    state: StreamingState
    native_raw: Tensor


@dataclass(frozen=True)
class ContextOutput:
    raw: torch.Tensor
    deployed: torch.Tensor
    physical_mixture: torch.Tensor
    warmup_samples: int
    scored_samples: int
    carried_state: bool
    initial_state_detached: bool
    data_hops: int
    flush_hops: int


class StemgenRT58(nn.Module):
    """Four-stem stereo separator with 128-sample hops and eight FP32 states.

    ``render`` accepts a whole number of hops, while ``forward_chunk`` accepts
    exactly one. Returned audio has 128 samples of graph alignment. The host's
    separate 128-sample queue gives 256 samples of total algorithmic latency.
    The model uses 32 attention frames and only received audio.
    CUDA training can set ``training_precision = "bf16"``; public states,
    synthesis, and parameters remain FP32.
    """

    sample_rate = 44100
    hop_samples = HOP
    synthesis_samples = SYNTHESIS_SAMPLES
    feature_samples = FEATURE_SAMPLES
    graph_alignment_samples = HOP
    alignment_samples = HOP
    host_queue_samples = HOP
    algorithmic_latency_samples = 2 * HOP
    host_visible_pdc_samples = 2 * HOP
    public_fusion_state_scale = PUBLIC_FUSION_SCALE
    future_context_samples = HOP
    future_callbacks_beyond_received_input = 0
    flush_required = True
    flush_hops = 1

    attention_window = 32
    parameter_tensor_count = 40
    state_type = StreamingState

    def __init__(self):
        super().__init__()
        # Keep parameter registration and initialization order checkpoint-stable.
        self.spec_encode = nn.Linear(CHANNELS * MASK_BINS * 2, EMBED)
        self.conv_encode = nn.Conv1d(CHANNELS, BASIS * 2, FEATURE_SAMPLES, stride=HOP)
        self.basis_to_embed = nn.Conv1d(BASIS, EMBED, 1)
        self.fusion_branch = nn.GRU(2 * EMBED, 2 * EMBED, num_layers=2, batch_first=True)
        self.spec_norm = nn.RMSNorm(EMBED)
        self.to_spec_masks = nn.Linear(EMBED, CHANNELS * MASK_BINS * 2 * SOURCES)
        self.waveform_norm = nn.RMSNorm(EMBED)
        self.to_waveform_masks = nn.Linear(EMBED, SOURCES * BASIS)
        self.waveform_decoder_weight = nn.Parameter(torch.zeros(BASIS, CHANNELS, SYNTHESIS_SAMPLES))
        analysis, spectral = asymmetric_windows()
        self.register_buffer("analysis_window", analysis)
        self.register_buffer("output_source_scales", torch.ones(SOURCES, dtype=torch.float32))
        self.synthesis = AsymmetricSynthesis(self.analysis_window, spectral)
        self.register_buffer("fixed_residual_share", torch.tensor(1 / 16, dtype=torch.float32))
        self.spec_encode = MagnitudeEncoder()
        self.phase_reduce = nn.Linear(EMBED, 64, bias=False)
        self.phase_expand = nn.Linear(64, CHANNELS * (MASK_BINS - 2) * SOURCES, bias=False)
        nn.init.zeros_(self.phase_expand.weight)
        self.fusion_refine_reduce = nn.Linear(2 * EMBED, 128, bias=False)
        self.fusion_refine_expand = nn.Linear(128, 2 * EMBED, bias=False)
        nn.init.zeros_(self.fusion_refine_expand.weight)
        self.temporal_query = nn.Linear(2 * EMBED, KEY, bias=False)
        self.temporal_key = nn.Linear(2 * EMBED, KEY, bias=False)
        self.temporal_value = nn.Linear(2 * EMBED, VALUE, bias=False)
        self.temporal_output = nn.Linear(VALUE, 2 * EMBED, bias=False)
        nn.init.zeros_(self.temporal_output.weight)
        self.spec_memory = nn.GRU(EMBED, EMBED, num_layers=1, batch_first=True)
        self.waveform_memory = nn.GRU(EMBED, EMBED, num_layers=1, batch_first=True)
        self.spec_memory_output = nn.Linear(EMBED, EMBED, bias=False)
        self.waveform_memory_output = nn.Linear(EMBED, EMBED, bias=False)
        nn.init.zeros_(self.spec_memory_output.weight)
        nn.init.zeros_(self.waveform_memory_output.weight)
        self.training_precision = "fp32"
        self.provenance = {"version": "cropped1024-asymmetric256-hop128-v1",
                           "initialization": "uninitialized_schema_only"}

    @property
    def architecture_metadata(self):
        return copy.deepcopy(_ARCHITECTURE)

    def initial_state(self, batch_size, *, device=None):
        require(type(batch_size) is int and batch_size > 0, "Expected positive batch size")
        device = self.output_source_scales.device if device is None else torch.device(device)
        require(device == self.output_source_scales.device
                and self.output_source_scales.dtype == torch.float32, "State uses model device and FP32")
        shapes = ((batch_size, CHANNELS, FEATURE_HISTORY), (2, batch_size, 2 * EMBED),
                  (batch_size, SOURCES, CHANNELS, HOP), (batch_size, SOURCES, CHANNELS, HOP),
                  (batch_size, self.attention_window - 1, KEY), (batch_size, self.attention_window - 1, VALUE),
                  (1, batch_size, EMBED), (1, batch_size, EMBED))
        return self.state_type(*(torch.zeros(shape, dtype=torch.float32, device=device) for shape in shapes))

    def _validate(self, audio, state):
        require(audio.ndim == 3 and audio.shape[0] > 0 and audio.shape[1] == CHANNELS
                and audio.shape[-1] >= HOP and audio.shape[-1] % HOP == 0
                and audio.dtype == torch.float32 and audio.device == self.output_source_scales.device
                and self.output_source_scales.dtype == torch.float32, "Invalid branch-memory input")
        if state is None:
            return self.initial_state(audio.shape[0])
        require(type(state) is self.state_type, "Use the distinct branch-memory state family")
        shapes = ((audio.shape[0], CHANNELS, FEATURE_HISTORY), (2, audio.shape[0], 2 * EMBED),
                  (audio.shape[0], SOURCES, CHANNELS, HOP), (audio.shape[0], SOURCES, CHANNELS, HOP),
                  (audio.shape[0], self.attention_window - 1, KEY), (audio.shape[0], self.attention_window - 1, VALUE),
                  (1, audio.shape[0], EMBED), (1, audio.shape[0], EMBED))
        require(all(value.shape == shape and value.dtype == audio.dtype and value.device == audio.device
                    for value, shape in zip(state, shapes, strict=True)), "Branch-memory state geometry changed")
        return state

    @staticmethod
    def _residual_source_softmax(logits: Tensor) -> Tensor:
        return logits.add(torch.softmax(logits, dim=-1), alpha=float(SOURCES))

    def render(self, audio: Tensor, state: StreamingState | None = None) -> ModelOutput:
        state = self._validate(audio, state)
        with torch.autocast(audio.device.type, enabled=False):
            return self._render_fp32(audio, state)

    def _render_fp32(self, audio, state):
        return self._render_impl(audio, state, tail_only=False)

    @torch.no_grad()
    def warm_state(self, audio, state=None):
        state = self._validate(audio, state)
        with torch.autocast(audio.device.type, enabled=False):
            return self._render_impl(audio, state, tail_only=True)

    def phase_coefficients(self, features):
        projected = self.phase_expand(self.phase_reduce(features))
        with torch.autocast(features.device.type, enabled=False):
            values = projected.float().reshape(features.shape[0], features.shape[1], CHANNELS, MASK_BINS - 2, SOURCES)
            # This centering is before inherited source calibration; it is not a
            # claim that the calibrated four-source correction sums to exactly zero.
            values = values - values.mean(-1, keepdim=True)
            return F.pad(values, (0, 0, 1, 1)).permute(0, 2, 1, 3, 4)

    def refinement(self, fused):
        return self.fusion_refine_expand(F.silu(self.fusion_refine_reduce(fused)))

    def attention(self, fused, past_keys, past_values, *, tail_only=False):
        window = self.attention_window
        queries = self.temporal_query(fused[:, -1:] if tail_only else fused)
        keys = torch.cat((past_keys, self.temporal_key(fused).float()), dim=1)
        values = torch.cat((past_values, self.temporal_value(fused).float()), dim=1)
        if tail_only:
            key_windows = keys[:, -window:].unsqueeze(1)
            value_windows = values[:, -window:].unsqueeze(1)
        else:
            key_windows = keys.unfold(1, window, 1).transpose(-1, -2)
            value_windows = values.unfold(1, window, 1).transpose(-1, -2)
        # Explicit FP32 attention normalization even during learned BF16 projections.
        with torch.autocast(fused.device.type, enabled=False):
            logits = (queries.float().unsqueeze(-2) * key_windows).sum(-1) * (KEY ** -.5)
            weights = torch.softmax(logits, dim=-1)
            attended = (weights.unsqueeze(-1) * value_windows).sum(-2)
        return (self.temporal_output(attended), keys[:, -(window - 1):].clone(),
                values[:, -(window - 1):].clone())

    def _render_impl(self, audio, state, *, tail_only):
        bf16 = self.training and self.training_precision == "bf16"
        require(not self.training or self.training_precision in ("fp32", "bf16"), "Unknown precision")
        require(not bf16 or audio.device.type == "cuda", "BF16 learned operations require CUDA")
        def learned():
            return torch.autocast(audio.device.type, dtype=torch.bfloat16, enabled=bf16)
        batch, _, samples = audio.shape
        joined = torch.cat((state.audio_history, audio), dim=-1)
        feature = torch.fft.rfft(joined.unfold(-1, FEATURE_SAMPLES, HOP) * self.analysis_window,
                                 n=FEATURE_SAMPLES, dim=-1)
        packed = torch.view_as_real(feature).permute(0, 2, 1, 3, 4).flatten(2)
        with learned():
            spec = self.spec_encode(packed)
            convolution = self.conv_encode(joined)
        to_relu, to_sigmoid = convolution.float().chunk(2, dim=1)
        basis = to_relu.relu() * to_sigmoid.sigmoid()
        with learned():
            waveform = self.basis_to_embed(basis).transpose(1, 2)
        spec, waveform = spec.float(), waveform.float()
        fusion_input = torch.cat((spec, waveform), dim=-1)
        physical_hidden = state.fusion_hidden / PUBLIC_FUSION_SCALE
        with learned():
            recurrent, hidden = self.fusion_branch(
                fusion_input.to(torch.bfloat16) if bf16 else fusion_input,
                physical_hidden.to(torch.bfloat16) if bf16 else physical_hidden)
        fused = fusion_input + recurrent.float()
        with learned():
            refinement = self.refinement(fused)
            refined = fused + refinement.float()
            correction, attention_keys, attention_values = self.attention(
                refined, state.attention_keys, state.attention_values, tail_only=tail_only)
        # Warmup advances both branch memories through every received frame.
        # Their inputs do not depend on attention, so warmup can compute only
        # the final attention query while retaining every recurrent update.
        private_spec, private_waveform = refined.chunk(2, dim=-1)
        private_spec = self.spec_norm(private_spec + spec)
        private_waveform = self.waveform_norm(private_waveform + waveform)
        spec_initial = state.spec_memory_hidden / PUBLIC_FUSION_SCALE
        waveform_initial = state.waveform_memory_hidden / PUBLIC_FUSION_SCALE
        with learned():
            spec_memory, spec_hidden = self.spec_memory(
                private_spec.to(torch.bfloat16) if bf16 else private_spec,
                spec_initial.to(torch.bfloat16) if bf16 else spec_initial)
            waveform_memory, waveform_hidden = self.waveform_memory(
                private_waveform.to(torch.bfloat16) if bf16 else private_waveform,
                waveform_initial.to(torch.bfloat16) if bf16 else waveform_initial)
            spec_correction = self.spec_memory_output(spec_memory)
            waveform_correction = self.waveform_memory_output(waveform_memory)
        if tail_only:
            spec_correction, waveform_correction = (value[:, -1:] for value in (spec_correction, waveform_correction))
            spec, waveform, refined = (value[:, -1:] for value in (spec, waveform, refined))
            basis, feature = basis[:, :, -1:], feature[:, :, -1:]
        frame_count = spec.shape[1]
        fused_spec, fused_waveform = (refined + correction.float()).chunk(2, dim=-1)
        spec = self.spec_norm(fused_spec + spec) + spec_correction.float()
        waveform = self.waveform_norm(fused_waveform + waveform) + waveform_correction.float()
        with learned():
            spec_logits = self.to_spec_masks(spec)
            waveform_logits = self.to_waveform_masks(waveform)
            phase = self.phase_coefficients(spec)
        spec_logits = spec_logits.float().reshape(batch, frame_count, CHANNELS, MASK_BINS, 2, SOURCES)
        masks = self._residual_source_softmax(spec_logits).permute(0, 2, 1, 3, 4, 5)
        carrier = torch.view_as_real(feature)
        masked = carrier.unsqueeze(-1) * masks + cross_component_correction(carrier, phase)
        spectrum = torch.view_as_complex(masked.permute(0, 5, 1, 2, 3, 4).contiguous())
        spectral, spec_tail = self.synthesis.spectral(spectrum, state.spectral_numerator_tail)
        waveform_logits = waveform_logits.float().reshape(batch, frame_count, SOURCES, BASIS).transpose(-1, -2)
        masks = self._residual_source_softmax(waveform_logits)
        source_basis = (basis.transpose(1, 2).unsqueeze(-1) * masks).permute(0, 3, 1, 2)
        with learned():
            decoded = F.linear(source_basis, self.waveform_decoder_weight.flatten(1).t(), bias=None)
        decoded = decoded.float().reshape(batch, SOURCES, frame_count, CHANNELS, SYNTHESIS_SAMPLES).permute(0, 1, 3, 2, 4)
        waveform_audio, wave_tail = self.synthesis.waveform(decoded, state.waveform_tail)
        next_state = self.state_type(joined[..., -FEATURE_HISTORY:].clone(),
                                    hidden.float() * PUBLIC_FUSION_SCALE, spec_tail, wave_tail,
                                    attention_keys, attention_values,
                                    spec_hidden.float() * PUBLIC_FUSION_SCALE,
                                    waveform_hidden.float() * PUBLIC_FUSION_SCALE)
        if tail_only:
            return next_state
        scales = self.output_source_scales[None, :, None, None]
        native_raw = (spectral + waveform_audio) * scales
        mixture = torch.cat((state.audio_history[..., -HOP:], audio), dim=-1)[..., :samples]
        raw, deployed = corrected_estimates(native_raw, mixture, self.fixed_residual_share)
        return ModelOutput(raw, deployed, spectral * scales, waveform_audio * scales,
                                   mixture, next_state, native_raw)

    def forward(self, audio: Tensor, state: StreamingState | None = None, *, return_raw=False):
        output = self.render(audio, state)
        return (output.raw if return_raw else output.deployed), output.state

    def forward_chunk(self, audio: Tensor, state: StreamingState | None = None, *, return_raw=False):
        require(audio.ndim == 3 and audio.shape[-1] == HOP, "Literal input requires exactly 128 samples")
        return self.forward(audio, state, return_raw=return_raw)

    def flush(self, state, *, return_raw=False):
        require(type(state) is self.state_type, "Flush requires a branch-memory state")
        return self.forward_chunk(state.audio_history.new_zeros((state.audio_history.shape[0], CHANNELS, HOP)),
                                  state, return_raw=return_raw)


def render_scored_context(model, mixture, *, warmup_samples, carry_state):
    """Warm all recurrent states without gradients, then return aligned scored audio."""
    from .compact import CompactSeparator
    from .banded import BandSeparator
    from .specialist import SpecialistSeparator
    if type(model) in (CompactSeparator, BandSeparator, SpecialistSeparator):
        return model.render_scored_context(mixture, warmup_samples=warmup_samples, carry_state=carry_state)
    require(carry_state is True and type(warmup_samples) is int and warmup_samples > 0
            and warmup_samples % 128 == 0 and mixture.ndim == 3 and mixture.shape[1] == 2
            and mixture.shape[-1] > warmup_samples and mixture.dtype == torch.float32,
            "Invalid branch_memory training context")
    require(type(model) is StemgenRT58, "Require the branch_memory model")
    state = model.warm_state(mixture[..., :warmup_samples]).detached()
    require(all(not value.requires_grad and value.grad_fn is None for value in state), "Warmup retained gradients")
    score = mixture[..., warmup_samples:]
    count, padding = score.shape[-1], (-score.shape[-1]) % 128
    result = model.render(F.pad(score, (0, padding + 128)), state)
    raw, deployed, physical = (value[..., 128:128 + count] for value in
                               (result.raw, result.deployed, result.delayed_mixture))
    require(raw.shape == deployed.shape == (mixture.shape[0], 4, 2, count)
            and torch.equal(physical, score), "Temporal-attention context lost physical alignment")
    return ContextOutput(raw, deployed, physical, warmup_samples, count, True, True,
                         (count + padding) // 128, 1)
