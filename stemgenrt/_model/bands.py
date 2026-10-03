"""Causal local/global band core; equal-width bands use independent batched maps."""
import torch
from torch import nn
from torch.nn import functional as F

from .dsp import require

BAND_EDGES = (0, 4, 8, 12, 16, 20, 24, 32, 40, 48, 64, 80, 96, 128, 160, 192,
              240, 288, 352, 416, 513)


class GroupedAffine(nn.Module):
    def __init__(self, linears):
        super().__init__()
        self.in_features, self.out_features = linears[0].in_features, linears[0].out_features
        require(all(m.in_features == self.in_features and m.out_features == self.out_features
                    for m in linears), "Grouped maps must have matching dimensions")
        self.weight = nn.Parameter(torch.stack([m.weight.detach() for m in linears]))
        self.bias = nn.Parameter(torch.stack([m.bias.detach() for m in linears]))

    def forward(self, values):
        if self.training:
            # Keep each band's weight matrix unexpanded during autograd.
            # Broadcast matmul otherwise materializes a weight-gradient matrix
            # for every batch/frame coordinate before reducing it. Folding
            # those coordinates makes that reduction part of one batched GEMM.
            grouped = values.movedim(-2, 0).flatten(1, -2)
            result = torch.bmm(grouped, self.weight.transpose(-2, -1))
            shape = (self.weight.shape[0], *values.shape[:-2], self.out_features)
            return result.reshape(shape).movedim(0, -2) + self.bias
        # Preserve the established one-hop evaluation/export graph exactly.
        return torch.matmul(values.unsqueeze(-2), self.weight.transpose(-2, -1)).squeeze(-2) + self.bias


class CausalBandBackbone(nn.Module):
    """Per-band temporal GRUs share weights and keep separate hidden states.

    Each layer exchanges information through a global causal GRU. Band edges
    partition the existing FFT bins; they do not add frequency resolution.
    Outputs are independent complex masks, with no source-axis normalization.
    """
    def __init__(self, *, sources=1, band_width=96, global_width=192, layers=2):
        super().__init__()
        require(type(sources) is int and sources in (1, 4), "Expected one or four outputs")
        require(type(band_width) is int and 16 <= band_width <= 256
                and type(global_width) is int and 16 <= global_width <= 512
                and type(layers) is int and 1 <= layers <= 4, "Invalid band backbone geometry")
        self.edges, self.band_count = BAND_EDGES, len(BAND_EDGES) - 1
        self.sources, self.width, self.global_width, self.layers = sources, band_width, global_width, layers
        encoders = [nn.Linear(6 * (r - l), band_width) for l, r in zip(self.edges, self.edges[1:])]
        self.band_identity = nn.Parameter(torch.zeros(self.band_count, band_width))
        self.input_norm = nn.RMSNorm(band_width, eps=1e-6)
        self.local = nn.ModuleList(nn.GRU(band_width, band_width, batch_first=True) for _ in range(layers))
        self.global_in = nn.ModuleList(nn.Linear(self.band_count * band_width, global_width) for _ in range(layers))
        self.global_memory = nn.ModuleList(nn.GRU(global_width, global_width, batch_first=True) for _ in range(layers))
        self.global_out = nn.ModuleList(nn.Linear(global_width, self.band_count * band_width) for _ in range(layers))
        self.local_norm = nn.ModuleList(nn.RMSNorm(band_width, eps=1e-6) for _ in range(layers))
        self.global_norm = nn.ModuleList(nn.RMSNorm(global_width, eps=1e-6) for _ in range(layers))
        hidden = [nn.Linear(band_width, band_width) for _ in range(self.band_count)]
        heads = []
        for left, right in zip(self.edges, self.edges[1:]):
            bins = right - left
            # Preserve the same RNG consumption for either number of outputs.
            with torch.random.fork_rng(devices=[]):
                head = nn.Linear(band_width, sources * 4 * bins)
            with torch.no_grad():
                weight = torch.randn(4 * bins, band_width) * (1e-3 / band_width ** .5)
                head.weight.copy_(weight.repeat(sources, 1))
                bias = torch.zeros(2, bins, 2)
                bias[..., 0] = .25
                head.bias.copy_(bias.flatten().repeat(sources))
            heads.append(head)
        self.groups = []
        for i, (left, right) in enumerate(zip(self.edges, self.edges[1:])):
            width = right - left
            if self.groups and self.groups[-1][2] == width:
                first, _, _ = self.groups[-1]
                self.groups[-1] = first, i + 1, width
            else:
                self.groups.append((i, i + 1, width))
        self.encoders = nn.ModuleList(GroupedAffine(encoders[first:last]) for first, last, _ in self.groups)
        self.mask_hidden = GroupedAffine(hidden)
        self.mask_heads = nn.ModuleList(GroupedAffine(heads[first:last]) for first, last, _ in self.groups)

    def forward(self, features, local_hidden, global_hidden):
        # [batch, frames, stereo, frequency bins, real/imag/log magnitude]
        batch, frames = features.shape[:2]
        values = []
        for module, (first, last, width) in zip(self.encoders, self.groups):
            chunk = features[..., self.edges[first]:self.edges[last], :]
            chunk = chunk.reshape(batch, frames, 2, last - first, width, 3)
            values.append(module(chunk.permute(0, 1, 3, 2, 4, 5).flatten(3)))
        bands = self.input_norm(F.silu(torch.cat(values, dim=2)) + self.band_identity)
        locals_out, globals_out = [], []
        for index in range(self.layers):
            values = bands.permute(0, 2, 1, 3).reshape(batch * self.band_count, frames, self.width)
            temporal, local_state = self.local[index](values, local_hidden[index:index + 1])
            temporal = temporal.reshape(batch, self.band_count, frames, self.width).permute(0, 2, 1, 3)
            bands = self.local_norm[index](bands + temporal)
            projected = self.global_in[index](bands.flatten(2))
            shared, global_state = self.global_memory[index](projected, global_hidden[index:index + 1])
            shared = self.global_norm[index](projected + shared)
            bands = bands + self.global_out[index](shared).reshape(batch, frames, self.band_count, self.width)
            locals_out.append(local_state)
            globals_out.append(global_state)
        hidden = F.silu(self.mask_hidden(bands))
        masks = []
        for module, (first, last, width) in zip(self.mask_heads, self.groups):
            value = module(hidden[:, :, first:last])
            value = value.reshape(batch, frames, last - first, self.sources, 2, width, 2)
            masks.append(value.permute(0, 1, 3, 4, 2, 5, 6).flatten(4, 5))
        return torch.cat(masks, dim=4), torch.cat(locals_out), torch.cat(globals_out)

    def dense_macs(self):
        band, glob, count, layers = self.width, self.global_width, self.band_count, self.layers
        return (6 * 513 * band + layers * (count * 6 * band * band
                + 2 * count * band * glob + 6 * glob * glob)
                + count * band * band + self.sources * 4 * 513 * band)
