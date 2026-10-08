"""Experimental dynamic U8/S8 export for the causal band separator.

Each complete matrix initializer has one symmetric reduced-range S8 scale.
In particular, a packed band group uses one scalar scale, avoiding unsupported
per-channel zero-point shapes in ORT's batched integer MatMul kernel. This is
a separate numerical variant: FP32 checkpoint quality does not qualify it.
"""
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from ._model.bands import GroupedAffine
from ._model.dsp import require
from .band_export import interface, make_export_copy
from ._export.helpers import model_state_sha256, sha

VERSION = "experimental-band-dynamic-u8s8-precise-v1"


def _digest(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def source_matrices(model):
    matrices = {}
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, GroupedAffine)):
            matrices[name] = module.weight.detach().cpu().numpy().swapaxes(-2, -1).copy()
        elif isinstance(module, nn.GRU):
            require(module.num_layers == 1, "Expected interleaved one-layer band memories")
            for kind in ("weight_ih_l0", "weight_hh_l0"):
                matrices[name + "." + kind] = getattr(module, kind).detach().cpu().numpy().T.copy()
    require(len(matrices) == 15 + 6 * model.layers, "Band projection inventory changed")
    return matrices


def export_int8(model, fp32_path, destination, *, expected_fp32_sha256):
    """Convert all learned dense matrices; preserve the full audio/state ABI."""
    import onnx
    from onnx import TensorProto as T, helper, numpy_helper as nh
    from onnxruntime.quantization.onnx_model import ONNXModel
    from onnxruntime.quantization.quant_utils import quantize_data

    contract = interface(model)
    destination = Path(destination)
    require(not destination.exists(), "Use a new path for the experimental graph")
    require(isinstance(expected_fp32_sha256, str) and len(expected_fp32_sha256) == 64
            and sha(fp32_path) == expected_fp32_sha256, "FP32 graph digest differs")
    before = model_state_sha256(model)
    graph = onnx.load(str(fp32_path))
    metadata = json.loads({p.key: p.value for p in graph.metadata_props}["stemgenrt.experimental"])
    require(metadata["precision"] == "fp32" and metadata["weight_sha256"] == before
            and metadata["architecture"] == model.architecture_metadata
            and metadata["interface"] == contract, "FP32 graph/source model identity differs")
    helper_model = ONNXModel(graph)
    helper_model.replace_gemm_with_matmul()
    graph = helper_model.model
    weights = {w.name: nh.to_array(w) for w in graph.graph.initializer}
    expected = source_matrices(model)
    removed, used, added, nodes, proof = set(), set(), [], [], []
    for node in graph.graph.node:
        if node.op_type != "MatMul":
            nodes.append(node)
            continue
        require(node.input[1] in weights, "Unexpected dynamic matrix multiplication")
        name, weight = node.input[1], weights[node.input[1]]
        matches = [key for key, value in expected.items() if np.array_equal(value, weight)]
        require(len(matches) == 1 and matches[0] not in used and name not in removed
                and sum(name in n.input for n in graph.graph.node) == 1,
                "Unmapped, duplicated or shared band matrix")
        module = matches[0]
        zero, scale, quantized = quantize_data(weight, T.INT8, symmetric=True, reduce_range=True)
        zero = np.asarray(zero, np.int8).reshape(())
        scale = np.asarray(scale, np.float32).reshape(())
        quantized = np.asarray(quantized, np.int8)
        require(zero == 0 and np.isfinite(scale) and scale > 0
                and np.abs(quantized.astype(np.int16)).max() <= 64
                and weight.shape[-2] * 255 * 64 < 2**31, "Invalid reduced-range integer matrix")
        for suffix, value in (("_quantized", quantized), ("_scale", scale), ("_zero_point", zero)):
            added.append(nh.from_array(value, name + suffix))
        prefix = node.name + "/u8s8"
        q, xs, xz, integer, floating, scales = (prefix + suffix for suffix in
            ("/input", "/input_scale", "/input_zero", "/integer", "/float", "/scales"))
        nodes.extend([
            helper.make_node("DynamicQuantizeLinear", [node.input[0]], [q, xs, xz], name=prefix + "/quantize"),
            helper.make_node("MatMulInteger", [q, name + "_quantized", xz, name + "_zero_point"],
                             [integer], name=prefix + "/matmul"),
            helper.make_node("Cast", [integer], [floating], name=prefix + "/cast", to=T.FLOAT),
            helper.make_node("Mul", [xs, name + "_scale"], [scales], name=prefix + "/scale"),
            helper.make_node("Mul", [floating, scales], list(node.output), name=prefix + "/dequantize")])
        proof.append({"module": module, "initializer": name, "shape": list(weight.shape),
            "source_matrix_sha256": _digest(weight), "quantized_matrix_sha256": _digest(quantized),
            "scale_sha256": _digest(scale), "scale_shape": [], "zero_point_shape": [],
            "worst_centered_dot_absolute_sum": weight.shape[-2] * 255 * 64,
            "maximum_unsigned_signed_pair_absolute_sum": 2 * 255 * 64})
        removed.add(name)
        used.add(module)
    require(used == set(expected), "Not every learned matrix was converted")
    preserved = {v.name: v.SerializeToString() for v in graph.graph.initializer if v.name not in removed}
    kept = [v for v in graph.graph.initializer if v.name not in removed]
    del graph.graph.node[:]
    graph.graph.node.extend(nodes)
    del graph.graph.initializer[:]
    graph.graph.initializer.extend([*kept, *added])
    del graph.graph.value_info[:]
    onnx.checker.check_model(graph, full_check=True)
    from ._export.band_precision import rewrite
    graph, precision_report = rewrite(graph)
    metadata.update(precision="dynamic U8/S8 products; FP64 quantizer ancestors and analysis; FP32 decoding and public states",
                    runtime_variant=VERSION, source_fp32_sha256=expected_fp32_sha256,
                    quality_measured=False, native_host_qualified=False)
    helper.set_model_props(graph, {"stemgenrt.experimental": json.dumps(metadata, sort_keys=True)})
    require(all(v.SerializeToString() == preserved[v.name] for v in graph.graph.initializer if v.name in preserved)
            and model_state_sha256(model) == before, "Unconverted weights or source model changed")
    destination.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(graph, str(destination))
    return {**metadata, "onnx_sha256": sha(destination), "bytes": destination.stat().st_size,
            "converted_matrices": proof, "unconverted_initializers_byte_exact": True,
            "precision_boundaries": precision_report,
            "activation_scale_scope": "all elements of each one-hop matrix input, including packed bands"}


class _IntegerProduct(nn.Module):
    """Independent NumPy ranges and int32 products; no ORT execution oracle."""
    def __init__(self, weight, stored, name):
        super().__init__()
        raw = np.asarray(weight, np.float32)
        scale = np.float32(float(np.max(np.abs(raw))) / 64.)
        if scale < np.finfo(np.float32).tiny:
            scale = np.float32(1.)
        quantized = np.clip(np.rint(raw / scale), -64, 64).astype(np.int8)
        require(np.array_equal(quantized, stored[name + "_quantized"])
                and np.array_equal(scale, stored[name + "_scale"])
                and stored[name + "_zero_point"].shape == () and stored[name + "_zero_point"] == 0,
                "Independent weight reconstruction differs: " + name)
        self.weight, self.scale = quantized.astype(np.int32), scale

    def forward(self, values):
        require(values.device.type == "cpu" and values.dtype in (torch.float32, torch.float64),
                "Integer reference requires CPU floating input")
        x = values.detach().float().numpy()
        minimum = np.minimum(x.min(), np.float32(0))
        maximum = np.maximum(x.max(), np.float32(0))
        scale = np.float32((maximum - minimum) / np.float32(255.)) if maximum != minimum else np.float32(1.)
        require(np.isfinite(scale) and scale > 0, "Invalid reference activation scale")
        zero = np.clip(np.rint(-minimum / scale), 0, 255).astype(np.int32)
        quantized = np.clip(np.rint(x / scale) + zero, 0, 255).astype(np.int32)
        integer = (quantized - zero) @ self.weight
        return torch.from_numpy(integer.astype(np.float32) * np.float32(scale * self.scale))


class _IntegerAffine(nn.Module):
    def __init__(self, module, product):
        super().__init__()
        self.product, self.grouped = product, isinstance(module, GroupedAffine)
        self.register_buffer("bias", module.bias.detach().clone())

    def forward(self, values):
        result = self.product(values.unsqueeze(-2) if self.grouped else values)
        return (result.squeeze(-2) if self.grouped else result) + self.bias


class _IntegerGRU(nn.Module):
    def __init__(self, module, ih, hh):
        super().__init__()
        self.ih, self.hh = ih, hh
        self.register_buffer("bias_ih", module.bias_ih_l0.detach().clone())
        self.register_buffer("bias_hh", module.bias_hh_l0.detach().clone())

    def forward(self, values, hidden):
        input_gates = self.ih(values[:, 0]) + self.bias_ih
        hidden_gates = self.hh(hidden[0]) + self.bias_hh
        ir, iz, inn = input_gates.chunk(3, -1)
        hr, hz, hn = hidden_gates.chunk(3, -1)
        reset, update = torch.sigmoid(ir + hr), torch.sigmoid(iz + hz)
        candidate = torch.tanh(inn + reset * hn)
        current = candidate + update * (hidden[0] - candidate)
        return current.unsqueeze(1), current.unsqueeze(0)


class _PreciseStreaming(nn.Module):
    """Independent explicit arithmetic matching the graph's precision boundary."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, audio, history, local_hidden, global_hidden, tail):
        model, core = self.model, self.model.backbone
        joined = torch.cat((history.double(), audio.double()), -1)
        spectrum = torch.fft.rfft(joined * model.analysis_window)
        power = spectrum.real.square() + spectrum.imag.square()
        scale = power.mean((1, 2), keepdim=True).clamp_min(float(np.float32(1e-8))).sqrt()
        normalized = spectrum / scale
        magnitude = torch.log(1 + (power + float(np.float32(1e-12))).sqrt() / scale)
        features = torch.stack((normalized.real, normalized.imag, magnitude), -1).unsqueeze(1)
        values = []
        for module, (first, last, width) in zip(core.encoders, core.groups):
            chunk = features[..., core.edges[first]:core.edges[last], :]
            chunk = chunk.reshape(1, 1, 2, last - first, width, 3)
            values.append(module(chunk.permute(0, 1, 3, 2, 4, 5).flatten(3)))
        embedded = torch.cat(values, dim=2)
        bands = core.input_norm(embedded / (1 + torch.exp(-embedded)) + core.band_identity)
        locals_out, globals_out = [], []
        for index in range(core.layers):
            values = bands.permute(0, 2, 1, 3).reshape(core.band_count, 1, core.width)
            temporal, local = core.local[index](values, local_hidden[index:index + 1].double())
            bands = core.local_norm[index](bands + temporal.reshape(1, core.band_count, 1, core.width).permute(0, 2, 1, 3))
            projected = core.global_in[index](bands.flatten(2))
            shared, glob = core.global_memory[index](projected, global_hidden[index:index + 1].double())
            shared = core.global_norm[index](projected + shared)
            bands = bands + core.global_out[index](shared).reshape(1, 1, core.band_count, core.width)
            locals_out.append(local.float())
            globals_out.append(glob.float())
        hidden = core.mask_hidden(bands)
        hidden = hidden / (1 + torch.exp(-hidden))
        masks = []
        for module, (first, last, width) in zip(core.mask_heads, core.groups):
            value = module(hidden[:, :, first:last])
            value = value.reshape(1, 1, last - first, core.sources, 2, width, 2)
            masks.append(value.permute(0, 1, 3, 4, 2, 5, 6).flatten(4, 5))
        masks = torch.cat(masks, dim=4)[:, 0]
        carrier = torch.view_as_real(spectrum).float()[:, None]
        real = carrier[..., 0] * masks[..., 0] - carrier[..., 1] * masks[..., 1]
        imag = carrier[..., 0] * masks[..., 1] + carrier[..., 1] * masks[..., 0]
        frames = torch.fft.irfft(torch.complex(real, imag), n=1024)[..., 768:] * model.synthesis.spectral_window.float()
        audio = (frames[..., :128] + tail) / model.synthesis.spectral_denominator.float()
        return audio, joined[..., -896:].float(), torch.cat(locals_out), torch.cat(globals_out), frames[..., 128:]


def make_integer_reference(model, graph_path, report):
    """Reconstruct every quantized weight independently, then render one hop."""
    import onnx
    from onnx import numpy_helper as nh
    require(sha(graph_path) == report["onnx_sha256"] and report["weight_sha256"] == model_state_sha256(model),
            "Integer graph/source identity differs")
    stored = {v.name: nh.to_array(v) for v in onnx.load(str(graph_path)).graph.initializer}
    source = source_matrices(model)
    products = {row["module"]: _IntegerProduct(source[row["module"]], stored, row["initializer"])
                for row in report["converted_matrices"]}
    require(set(products) == set(source), "Reference matrix inventory differs")
    require(report["runtime_variant"] == VERSION, "Unsupported integer reference variant")
    wrapper = make_export_copy(model).double()
    from ._export.helpers import RMSNormForONNX
    for module in wrapper.modules():
        if isinstance(module, RMSNormForONNX):
            module.eps = float(np.float32(module.eps))
    for name, module in list(model.named_modules()):
        if isinstance(module, (nn.Linear, GroupedAffine)):
            original = copy.deepcopy(module).cpu()
            if not name.startswith("backbone.mask_heads."):
                original.double()
            replacement = _IntegerAffine(original, products[name])
        elif isinstance(module, nn.GRU):
            replacement = _IntegerGRU(copy.deepcopy(module).cpu().double(), products[name + ".weight_ih_l0"],
                                       products[name + ".weight_hh_l0"])
        else:
            continue
        parent, _, leaf = name.rpartition(".")
        target = wrapper.model.get_submodule(parent) if parent else wrapper.model
        setattr(target, leaf, replacement)
    return _PreciseStreaming(wrapper.model).eval().requires_grad_(False)
