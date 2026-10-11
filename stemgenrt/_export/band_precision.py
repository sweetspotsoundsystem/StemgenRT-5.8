"""FP64 quantizer ancestors for the experimental band integer graph.

Integer products and their dequantization stay FP32. Public audio and states,
complex masking and inverse synthesis remain FP32. Initializers are retained
byte-for-byte; explicit casts define every precision boundary.
"""
import copy

import numpy as np

from .helpers import require


def rewrite(graph):
    import onnx
    from onnx import TensorProto as T, helper, numpy_helper as nh

    graph = copy.deepcopy(graph)
    producers = {v: n for n in graph.graph.node for v in n.output}
    replacements, removed = {}, set()
    for product in graph.graph.node:
        if product.op_type != "Mul":
            continue
        for value in product.input:
            sigmoid = producers.get(value)
            if sigmoid is None or sigmoid.op_type != "Sigmoid" or sigmoid.input[0] not in product.input:
                continue
            require(sum(value in n.input for n in graph.graph.node) == 1, "Shared SiLU sigmoid")
            prefix = product.name + "/portable_silu"
            replacements[sigmoid.name] = [
                helper.make_node("Neg", list(sigmoid.input), [prefix + "/negative"], name=prefix + "/neg"),
                helper.make_node("Exp", [prefix + "/negative"], [prefix + "/exponential"], name=prefix + "/exp"),
                helper.make_node("Constant", [], [prefix + "/one"], name=prefix + "/constant",
                                 value=nh.from_array(np.asarray(1., np.float32))),
                helper.make_node("Add", [prefix + "/exponential", prefix + "/one"], [prefix + "/denominator"], name=prefix + "/add"),
                helper.make_node("Div", [sigmoid.input[0], prefix + "/denominator"], list(product.output), name=prefix + "/divide")]
            removed.add(product.name)
    require(len(replacements) == 2, "Expected the input and mask-hidden SiLU operations")
    nodes = [r for n in graph.graph.node if n.name not in removed for r in replacements.get(n.name, [n])]
    del graph.graph.node[:]
    graph.graph.node.extend(nodes)
    original = onnx.shape_inference.infer_shapes(graph, strict_mode=True, check_type=True)
    types = {v.name: v.type.tensor_type.elem_type for v in
             [*original.graph.input, *original.graph.output, *original.graph.value_info]}
    types.update({v.name: v.data_type for v in original.graph.initializer})
    producers = {v: n for n in original.graph.node for v in n.output}
    consumers = {}
    for node in original.graph.node:
        for value in node.input:
            consumers.setdefault(value, []).append(node)
    def single(value, operation):
        found = consumers[value]
        require(len(found) == 1 and found[0].op_type == operation, "Integer projection topology changed")
        return found[0]
    islands, quantizers = set(), []
    for integer in (n for n in original.graph.node if n.op_type == "MatMulInteger"):
        quantizer = producers[integer.input[0]]
        cast = single(integer.output[0], "Cast")
        scaled = single(cast.output[0], "Mul")
        scales = producers[next(v for v in scaled.input if v != cast.output[0])]
        require(quantizer.op_type == "DynamicQuantizeLinear" and scales.op_type == "Mul"
                and quantizer.output[1] in scales.input, "Unexpected integer projection")
        islands.update(n.name for n in (quantizer, integer, cast, scaled, scales))
        quantizers.append(quantizer)
    precise = set()
    def visit(value):
        node = producers.get(value)
        if node is None or node.name in islands or node.name in precise:
            return
        precise.add(node.name)
        for parent in node.input:
            visit(parent)
    for quantizer in quantizers:
        visit(quantizer.input[0])
    require(sum(n.op_type == "DFT" and n.name in precise for n in original.graph.node) == 1,
            "Only the analysis FFT should need FP64")
    available = {v.name: v.type.tensor_type.elem_type for v in original.graph.input}
    available.update({v.name: v.data_type for v in original.graph.initializer})
    aliases = {v.name: v.name + "__internal" for v in original.graph.output}
    nodes, casts = [], {}
    def input_as(name, precision):
        actual_name = aliases.get(name, name)
        actual_type = available[name]
        if actual_type not in (T.FLOAT, T.DOUBLE) or actual_type == precision:
            return actual_name
        key = name, precision
        if key not in casts:
            value = name + ("__to64" if precision == T.DOUBLE else "__to32")
            nodes.append(helper.make_node("Cast", [actual_name], [value],
                         name="/band_precision/cast_" + str(len(casts)), to=precision))
            casts[key] = value
        return casts[key]
    for source_node in original.graph.node:
        node = copy.deepcopy(source_node)
        precision = T.DOUBLE if node.name in precise else T.FLOAT
        for index, value in enumerate(node.input):
            if value:
                node.input[index] = input_as(value, precision)
        for attr in node.attribute:
            if attr.type == onnx.AttributeProto.TENSOR and attr.t.data_type == T.FLOAT and precision == T.DOUBLE:
                attr.t.CopyFrom(nh.from_array(nh.to_array(attr.t).astype(np.float64), attr.t.name))
            if node.op_type == "Cast" and attr.name == "to" and attr.i == T.FLOAT:
                attr.i = precision
        for index, value in enumerate(node.output):
            available[value] = precision if types[value] == T.FLOAT else types[value]
            node.output[index] = aliases.get(value, value)
        nodes.append(node)
    for output in original.graph.output:
        nodes.append(helper.make_node("Identity", [input_as(output.name, T.FLOAT)], [output.name],
                                     name="/band_precision/output/" + output.name))
    del graph.graph.node[:]
    graph.graph.node.extend(nodes)
    del graph.graph.value_info[:]
    onnx.checker.check_model(graph, full_check=True)
    return graph, {"fp64_ancestor_nodes": len(precise), "fp32_integer_blocks": len(quantizers),
                   "fp64_analysis_fft": True, "fp32_synthesis_fft": True}
