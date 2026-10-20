"""Experimental execution variants of independent-stem ONNX exports.

These transformations do not update training weights or replace the release.
Fused FP32 arithmetic still requires numerical and hardware qualification.
Selective INT8 is a distinct numerical candidate, requiring separation-quality
and absence/preservation evaluation before any deployment recommendation.
"""
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import helper as h, numpy_helper as nh, TensorProto as T

from ._export.helpers import sha

def require(value, message):
    if not value:
        raise ValueError(message)


def _cell_signature(prefix, nodes, constants):
    """Identify the exact exported recurrence, including every gate and bias."""
    group = [n for n in nodes.values() if n.name.startswith(prefix + "/") and n.op_type != "Constant"]
    gi, gh = nodes[prefix + "/Gemm"], nodes[prefix + "/Gemm_1"]
    ingress = {nodes[prefix + "/Gather"].input[0]: "sequence",
               nodes[prefix + "/Gather_1"].input[0]: "initial_h",
               gi.input[1]: "weight_ih", gh.input[1]: "weight_hh",
               gi.input[2]: "bias_ih", gh.input[2]: "bias_hh"}
    def encode(name):
        if name in ingress:
            return ["input", ingress[name]]
        if name in constants:
            value = constants[name]
            return ["constant", str(value.dtype), list(value.shape), value.tolist()]
        require(name.startswith(prefix + "/"), "Unexpected recurrence input: " + name)
        return ["intermediate", name[len(prefix) + 1:]]
    schema = [{"name": n.name[len(prefix) + 1:], "op": n.op_type, "domain": n.domain,
               "inputs": [encode(x) for x in n.input],
               "outputs": [x[len(prefix) + 1:] for x in n.output],
               "attributes": {a.name: h.get_attribute_value(a) for a in n.attribute}}
              for n in group]
    return hashlib.sha256(json.dumps(schema, sort_keys=True).encode()).hexdigest()


def prune(graph):
    needed = {x.name for x in graph.graph.output}
    kept = []
    for node in reversed(graph.graph.node):
        if needed.intersection(node.output):
            kept.append(node)
            needed.update(x for x in node.input if x)
    del graph.graph.node[:]
    graph.graph.node.extend(reversed(kept))
    initializers = [x for x in graph.graph.initializer if x.name in needed]
    del graph.graph.initializer[:]
    graph.graph.initializer.extend(initializers)
    del graph.graph.value_info[:]
    return graph


def fuse_grus(source):
    graph = copy.deepcopy(source)
    before_inputs = [x.SerializeToString() for x in graph.graph.input]
    before_outputs = [x.SerializeToString() for x in graph.graph.output]
    original = list(graph.graph.node)
    weights = {x.name: nh.to_array(x) for x in graph.graph.initializer}
    nodes = {x.name: x for x in original}
    constants = {n.output[0]: nh.to_array(next(a.t for a in n.attribute if a.name == "value"))
                 for n in original if n.op_type == "Constant"}
    removed, replacements, proof = set(), {}, []
    for stem in ("drums", "bass", "vocals"):
        for kind in ("local", "global_memory"):
            for layer in range(2):
                prefix = f"/{stem}/backbone/{kind}.{layer}"
                group = [x for x in original if x.name.startswith(prefix + "/")]
                require(len(group) == 53, f"Unrecognized cell node count: {prefix}: {len(group)}")
                require(_cell_signature(prefix, nodes, constants) ==
                        "f4275bd22c026c590a1d1038f94df815bfe60384da39ecfed47d702f3e1bff44",
                        "Unrecognized recurrent gate arithmetic: " + prefix)
                real = [x for x in group if x.op_type != "Constant"]
                outputs = {o for n in real for o in n.output}
                external = {i for n in original if n not in group for i in n.input if i in outputs}
                expected = {prefix + "/Concat_output_0", prefix + "/Unsqueeze_1_output_0"}
                require(external == expected, f"Unexpected cell boundaries: {prefix}: {external}")
                gi, gh = nodes[prefix + "/Gemm"], nodes[prefix + "/Gemm_1"]
                for node in (gi, gh):
                    attrs = {a.name: h.get_attribute_value(a) for a in node.attribute}
                    require(attrs == {"alpha": 1., "beta": 1., "transB": 1}, "Unexpected Gemm")
                wi, wh, bi, bh = (weights[x] for x in (gi.input[1], gh.input[1], gi.input[2], gh.input[2]))
                hidden = wh.shape[1]
                require(wi.shape == wh.shape == (3 * hidden, hidden)
                        and bi.shape == bh.shape == (3 * hidden,), "Unsupported GRU geometry")
                # PyTorch r,z,n -> ONNX z,r,h; retain both biases separately.
                order = np.r_[hidden:2 * hidden, 0:hidden, 2 * hidden:3 * hidden]
                names = [prefix + "/fused/" + x for x in ("W", "R", "B")]
                values = (wi[order][None], wh[order][None], np.concatenate((bi[order], bh[order]))[None])
                for name, value in zip(names, values):
                    graph.graph.initializer.append(nh.from_array(np.ascontiguousarray(value), name))
                sequence = nodes[prefix + "/Gather"].input[0]
                hidden_input = nodes[prefix + "/Gather_1"].input[0]
                transposed = prefix + "/fused/X"
                replacements[real[0].name] = [
                    h.make_node("Transpose", [sequence], [transposed], name=prefix + "/fused/transpose_input", perm=[1, 0, 2]),
                    h.make_node("GRU", [transposed, *names, "", hidden_input],
                                ["", prefix + "/Concat_output_0"], name=prefix + "/fused/GRU",
                                hidden_size=hidden, direction="forward", linear_before_reset=1),
                    h.make_node("Transpose", [prefix + "/Concat_output_0"],
                                [prefix + "/Unsqueeze_1_output_0"], name=prefix + "/fused/transpose_output", perm=[1, 0, 2])]
                removed.update(x.name for x in real)
                proof.append({"cell": prefix, "hidden": hidden, "input": sequence, "initial_h": hidden_input,
                              "source_weights": [gi.input[1], gh.input[1], gi.input[2], gh.input[2]],
                              "gate_order_source": "r,z,n", "gate_order_candidate": "z,r,h",
                              "linear_before_reset": 1, "removed_runtime_nodes": len(real)})
    new_nodes = []
    for node in original:
        if node.name in replacements:
            new_nodes.extend(replacements[node.name])
        if node.name not in removed:
            new_nodes.append(node)
    del graph.graph.node[:]
    graph.graph.node.extend(new_nodes)
    prune(graph)
    require(before_inputs == [x.SerializeToString() for x in graph.graph.input]
            and before_outputs == [x.SerializeToString() for x in graph.graph.output], "Public ABI changed")
    onnx.checker.check_model(graph, full_check=True)
    return graph, proof


def quantize_global(source):
    graph = copy.deepcopy(source)
    weights = {x.name: nh.to_array(x) for x in graph.graph.initializer}
    changed, new_nodes = [], []
    for node in graph.graph.node:
        selected = node.op_type == "MatMul" and any(f"/backbone/{kind}." in node.name for kind in ("global_in", "global_out"))
        if not selected:
            new_nodes.append(node)
            continue
        weight = weights[node.input[1]]
        require(weight.ndim == 2 and weight.dtype == np.float32, "Unexpected projection")
        scale = np.float32(np.max(np.abs(weight)) / 64.)
        require(np.isfinite(scale) and scale > 0, "Invalid scale")
        quantized = np.clip(np.rint(weight / scale), -64, 64).astype(np.int8)
        require(weight.shape[0] * 255 * 64 < 2 ** 31, "Integer accumulation overflow")
        prefix = node.name + "/u8s8"
        qw, ws, wz = [prefix + suffix for suffix in ("/weight", "/weight_scale", "/weight_zero")]
        for name, value in ((qw, quantized), (ws, scale), (wz, np.int8(0))):
            graph.graph.initializer.append(nh.from_array(np.asarray(value), name))
        qx, xs, xz, acc, fp, scales = [prefix + suffix for suffix in ("/input", "/input_scale", "/input_zero", "/integer", "/float", "/scales")]
        new_nodes.extend([
            h.make_node("DynamicQuantizeLinear", [node.input[0]], [qx, xs, xz], name=prefix + "/quantize"),
            h.make_node("MatMulInteger", [qx, qw, xz, wz], [acc], name=prefix + "/matmul"),
            h.make_node("Cast", [acc], [fp], name=prefix + "/cast", to=T.FLOAT),
            h.make_node("Mul", [xs, ws], [scales], name=prefix + "/scale"),
            h.make_node("Mul", [fp, scales], list(node.output), name=prefix + "/dequantize")])
        changed.append({"node": node.name, "source_initializer": node.input[1],
                        "shape": list(weight.shape), "scale": float(scale), "range": [-64, 64],
                        "max_pair_absolute_sum": 2 * 255 * 64,
                        "max_dot_absolute_sum": weight.shape[0] * 255 * 64})
    require(len(changed) == 12, "Expected twelve global projections")
    del graph.graph.node[:]
    graph.graph.node.extend(new_nodes)
    prune(graph)
    onnx.checker.check_model(graph, full_check=True)
    return graph, changed


def specialize_layouts(source):
    """Collapse reshape-only paths under the already fixed public tensor ABI.

Transpose becomes a reshape only when the order of all non-singleton axes is
unchanged. No floating arithmetic, matrix values, or true permutations change.
The output shape is proved by ONNX shape inference, not observed from audio.
"""
    graph = onnx.shape_inference.infer_shapes(copy.deepcopy(source), strict_mode=True, data_prop=True)
    types = {v.name: v for v in [*graph.graph.input, *graph.graph.value_info, *graph.graph.output]}
    shapes = {}
    for name, value in types.items():
        dims = value.type.tensor_type.shape.dim
        require(all(d.HasField("dim_value") for d in dims), "Unresolved intermediate shape: " + name)
        shapes[name] = [d.dim_value for d in dims]
    aliases, layout_origins, nodes, proof = {}, {}, [], []
    public_outputs = {v.name for v in graph.graph.output}

    def resolve(name):
        while name in aliases:
            name = aliases[name]
        return name

    for original in graph.graph.node:
        node = copy.deepcopy(original)
        node.input[:] = [resolve(x) for x in node.input]
        is_layout = node.op_type in ("Reshape", "Squeeze", "Unsqueeze")
        if node.op_type == "Transpose":
            shape = shapes[original.input[0]]
            permutation = next((list(a.ints) for a in node.attribute if a.name == "perm"), list(reversed(range(len(shape)))))
            is_layout = [axis for axis in permutation if shape[axis] != 1] == [axis for axis, size in enumerate(shape) if size != 1]
        if not is_layout:
            nodes.append(node)
            continue
        output = node.output[0]
        origin = node.input[0]
        if origin in layout_origins:
            origin = layout_origins[origin]
        origin = resolve(origin)
        wanted = shapes[output]
        require(np.prod(shapes[origin], dtype=np.int64) == np.prod(wanted, dtype=np.int64), "Layout changes element count")
        layout_origins[output] = origin
        proof.append({"node": node.name, "operator": node.op_type, "original_input": original.input[0],
                      "origin": origin, "input_shape": shapes[origin], "output_shape": wanted})
        if shapes[origin] == wanted and output not in public_outputs:
            aliases[output] = origin
            continue
        if shapes[origin] == wanted:
            nodes.append(h.make_node("Identity", [origin], [output], name=node.name + "/fixed_identity"))
        else:
            shape_name = node.name + "/fixed_shape"
            graph.graph.initializer.append(nh.from_array(np.asarray(wanted, dtype=np.int64), shape_name))
            nodes.append(h.make_node("Reshape", [origin, shape_name], [output], name=node.name + "/fixed_layout"))
    del graph.graph.node[:]
    graph.graph.node.extend(nodes)
    prune(graph)
    onnx.checker.check_model(graph, full_check=True)
    return graph, proof


def specialize_original_layouts(source):
    """Prove gate-slice shapes, then specialize layouts without changing GRUs.

The exported integer Shape/Div expressions hide the otherwise fixed gate-slice
sizes from ONNX shape inference. The exact validated recurrence and public
hidden-state dimensions establish these shapes without evaluating music.
"""
    _, cells = fuse_grus(source)  # Validate the complete known recurrence first.
    graph = copy.deepcopy(source)
    shapes = {x.name: [d.dim_value for d in x.type.tensor_type.shape.dim] for x in graph.graph.input}
    annotations = []
    for cell in cells:
        prefix = cell["cell"]
        stem = prefix.split("/")[1]
        kind = "local" if "/local." in prefix else "global"
        hidden_shape = shapes[stem + "_" + kind + "_hidden"]
        require(len(hidden_shape) == 3 and hidden_shape[0] == 2
                and hidden_shape[-1] == cell["hidden"], "Unsupported hidden-state geometry")
        shape = hidden_shape[1:]
        slices = [n for n in graph.graph.node if n.name.startswith(prefix + "/") and n.op_type == "Slice"]
        require(len(slices) == 6, "Expected six validated recurrent gate slices")
        for node in slices:
            annotations.append({"tensor": node.output[0], "shape": shape})
    replaced = {a["tensor"] for a in annotations}
    existing = [v for v in graph.graph.value_info if v.name not in replaced]
    del graph.graph.value_info[:]
    graph.graph.value_info.extend(existing)
    graph.graph.value_info.extend(h.make_tensor_value_info(a["tensor"], T.FLOAT, a["shape"]) for a in annotations)
    result, layout = specialize_layouts(graph)
    return result, {"recurrence_shape_annotations": annotations, "layout": layout,
                    "recurrent_arithmetic_changed": False}


VARIANTS = ("layout-fp32", "fused-fp32", "fused-layout-fp32", "fused-layout-global-int8")


def optimize(source, destination, *, expected_sha256, variant="layout-fp32"):
    """Write a new candidate from an authenticated complete FP32 export.

Only the fixed-shape, two-layer independent DBV export is supported. Every
cell must match the known one-hop recurrence before fusion. The public tensor
interface is retained byte for byte. The source file is never rewritten.
"""
    source, destination = Path(source), Path(destination)
    require(variant in VARIANTS, "Unknown optimization variant")
    require(not destination.exists(), "Use a new candidate destination")
    require(isinstance(expected_sha256, str) and len(expected_sha256) == 64
            and sha(source) == expected_sha256, "Source graph identity differs")
    graph = onnx.load(source)
    onnx.checker.check_model(graph, full_check=True)
    metadata = {v.key: v.value for v in graph.metadata_props}
    info = json.loads(metadata.get("stemgenrt.experimental", "{}"))
    require(info.get("schema") == "stemgenrt-experimental-independent-onnx-v1"
            and info.get("precision") == "fp32" and "runtime_variant" not in info,
            "Require an original independent FP32 export")
    require(len(graph.graph.input) == len(graph.graph.output) == 14
            and list(info["architecture"]["source_order"]) == ["drums", "bass", "vocals", "other"],
            "Independent graph interface differs")
    for tensor in graph.graph.input:
        require(tensor.type.tensor_type.elem_type == T.FLOAT and
                all(d.HasField("dim_value") and d.dim_value > 0 for d in tensor.type.tensor_type.shape.dim),
                "Require fixed positive FP32 input shapes")
    original_interface = ([x.SerializeToString() for x in graph.graph.input],
                          [x.SerializeToString() for x in graph.graph.output])
    if variant == "layout-fp32":
        result, proof = specialize_original_layouts(graph)
    else:
        result, fusion = fuse_grus(graph)
        proof = {"fusion": fusion}
        if variant != "fused-fp32":
            result, proof["layout"] = specialize_layouts(result)
        if variant.endswith("int8"):
            result, proof["quantization"] = quantize_global(result)
    require(original_interface == ([x.SerializeToString() for x in result.graph.input],
                                    [x.SerializeToString() for x in result.graph.output]), "Public tensor interface changed")
    info.update(runtime_variant=variant, source_fp32_sha256=expected_sha256,
                precision="fp32" if variant.endswith("fp32") else
                    "12 reduced-range U8/S8 global products; all other computation FP32",
                quality_measured=False, native_host_qualified=False)
    metadata["stemgenrt.experimental"] = json.dumps(info, sort_keys=True)
    h.set_model_props(result, metadata)
    onnx.checker.check_model(result, full_check=True)
    require(sha(source) == expected_sha256, "Source graph changed during conversion")
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation prevents silently replacing another candidate.
    with destination.open("xb") as stream:
        stream.write(result.SerializeToString())
    return {"schema": "independent-deployment-candidate-v1", "variant": variant,
            "source_fp32_sha256": expected_sha256, "onnx_sha256": sha(destination),
            "bytes": destination.stat().st_size, "training_weights_changed": False,
            "public_tensor_interface_unchanged": True, "quality_measured": False,
            "native_host_qualified": False, "transformation_proof": proof,
            "nodes": dict(Counter(n.op_type for n in result.graph.node))}


