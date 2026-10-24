"""Small ONNX lowering helpers shared by current-model exporters."""
from __future__ import annotations
import hashlib
import logging
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
logger = logging.getLogger(__name__)

def require(condition, message):
    if not condition:
        raise ValueError(message)

def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def state_sha256(state):
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str((tuple(tensor.shape), tensor.dtype)).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()

def model_state_sha256(model):
    return state_sha256(model.state_dict())

class RMSNormForONNX(nn.Module):
    """Small ONNX-friendly equivalent of ``torch.nn.RMSNorm``."""

    def __init__(self, original: nn.RMSNorm):
        super().__init__()
        if not bool(original.elementwise_affine) or original.weight is None:
            raise ValueError("ONNX RMSNorm replacement requires affine weights")
        eps = original.eps
        if eps is None:
            eps = torch.finfo(original.weight.dtype).eps
        self.eps = float(eps)
        self.register_buffer("weight", original.weight.detach().clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + self.eps)
        return x * rms * self.weight

class OneFrameGRUForONNX(nn.Module):
    """Exact, exportable GRU recurrence for the one-frame one-hop streaming graph."""

    def __init__(self, original: nn.GRU, *, batch_gates: bool = False):
        super().__init__()
        require(type(batch_gates) is bool, "batch_gates must be bool")
        self.batch_gates = batch_gates
        if not original.batch_first:
            raise ValueError("one-hop streaming export requires a batch-first GRU")
        if original.bidirectional:
            raise ValueError("one-hop streaming export requires a unidirectional GRU")
        if not original.bias:
            raise ValueError("one-hop streaming export requires GRU bias tensors")
        if float(original.dropout) != 0.0:
            raise ValueError("one-hop streaming export does not support GRU dropout")

        self.input_size = int(original.input_size)
        self.hidden_size = int(original.hidden_size)
        self.num_layers = int(original.num_layers)
        for layer in range(self.num_layers):
            for prefix in ("weight_ih", "weight_hh", "bias_ih", "bias_hh"):
                value = getattr(original, f"{prefix}_l{layer}")
                self.register_buffer(f"{prefix}_l{layer}", value.detach().clone())

    def forward(
        self, x: torch.Tensor, hidden: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        layer_input = x[:, 0]
        next_hidden: list[torch.Tensor] = []
        for layer in range(self.num_layers):
            previous = hidden[layer]
            input_gates = F.linear(
                layer_input,
                getattr(self, f"weight_ih_l{layer}"),
                getattr(self, f"bias_ih_l{layer}"),
            )
            hidden_gates = F.linear(
                previous,
                getattr(self, f"weight_hh_l{layer}"),
                getattr(self, f"bias_hh_l{layer}"),
            )
            if self.batch_gates:
                # Keep both affine operations and every scalar expression in
                # their original order. Only the independent reset/update
                # Add and Sigmoid dispatches share one tensor operation.
                sizes = (2 * self.hidden_size, self.hidden_size)
                input_pair, input_new = input_gates.split(sizes, dim=-1)
                hidden_pair, hidden_new = hidden_gates.split(sizes, dim=-1)
                reset, update = torch.sigmoid(input_pair + hidden_pair).chunk(2, dim=-1)
            else:
                input_reset, input_update, input_new = input_gates.chunk(3, dim=-1)
                hidden_reset, hidden_update, hidden_new = hidden_gates.chunk(3, dim=-1)
                reset = torch.sigmoid(input_reset + hidden_reset)
                update = torch.sigmoid(input_update + hidden_update)
            candidate = torch.tanh(input_new + reset * hidden_new)
            current = candidate + update * (previous - candidate)
            next_hidden.append(current)
            layer_input = current

        stacked_hidden = torch.stack(next_hidden, dim=0)
        return layer_input.unsqueeze(1), stacked_hidden

def replace_rmsnorm_layers(module: nn.Module) -> nn.Module:
    for name, child in list(module.named_children()):
        if isinstance(child, nn.RMSNorm):
            setattr(module, name, RMSNormForONNX(child))
            logger.debug("Replaced RMSNorm layer %s", name)
            continue
        replace_rmsnorm_layers(child)
    return module
