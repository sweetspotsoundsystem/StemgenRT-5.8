"""Partition an existing independent FP32 graph without changing its operations.

Each child owns its original state inputs/outputs. The host reconstructs Other
from the previous physical input and the three completed stem outputs. This is
an experimental deployment layout; the matching model and training code remain
unchanged. It is not an export of new weights or a timing qualification.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

SOURCES = ('drums', 'bass', 'vocals')
COUNTS = (5, 4, 4)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _validate_host_residual(model, terminal):
    """Prove the residual arithmetic and history update that the host replaces.

    Accept the explicit FP32 export layout. A graph with another residual or
    alignment must keep its parent execution until it has a matching host.
    No graph operations are rewritten by this validation.
    """
    import numpy as np
    from onnx import helper, numpy_helper, TensorProto

    nodes = {output: node for node in model.graph.node for output in node.output}
    constants = {tensor.name: tensor for tensor in model.graph.initializer}

    def require(condition, detail):
        if not condition:
            raise ValueError('Unsupported parent residual: ' + detail)

    def node(name, operation, inputs):
        result = nodes.get(name)
        require(result is not None and result.op_type == operation
                and result.domain in ('', 'ai.onnx') and len(result.input) == inputs,
                'expected ' + operation)
        return result

    def attribute(value, name, default=None):
        return next((helper.get_attribute_value(a) for a in value.attribute
                     if a.name == name), default)

    def integer(name):
        tensor = constants.get(name)
        if tensor is None:
            tensor = attribute(node(name, 'Constant', 0), 'value')
        require(isinstance(tensor, TensorProto) and tensor.data_type == TensorProto.INT64,
                'expected a constant int64 index')
        return numpy_helper.to_array(tensor)

    def equal_index(name, expected):
        return np.array_equal(integer(name), np.asarray(expected, dtype=np.int64))

    def shape(name):
        value = next((v for v in model.graph.input if v.name == name), None)
        require(value is not None and value.type.tensor_type.elem_type == TensorProto.FLOAT,
                'expected a float32 input')
        return [d.dim_value for d in value.type.tensor_type.shape.dim]

    def tail(name, source, source_length, length):
        sliced = node(name, 'Slice', 5)
        require(sliced.input[0] == source
                and (equal_index(sliced.input[1], [-length])
                     or equal_index(sliced.input[1], [source_length - length]))
                and (equal_index(sliced.input[2], [np.iinfo(np.int64).max])
                     or equal_index(sliced.input[2], [source_length]))
                and (equal_index(sliced.input[3], [2]) or equal_index(sliced.input[3], [-1]))
                and equal_index(sliced.input[4], [1]), 'expected the literal trailing samples')

    require(shape('audio_chunk') == [1, 2, 128], 'expected stereo 128-sample input')
    history = 'vocals_audio_history'
    dimensions = shape(history)
    require(len(dimensions) == 3 and dimensions[:2] == [1, 2] and dimensions[2] >= 128,
            'expected a vocal audio history of at least one hop')
    history_length = dimensions[2]
    expanded = node(terminal.input[3], 'Unsqueeze', 2)
    require(equal_index(expanded.input[1], [1]), 'expected the Other source axis')
    subtraction = node(expanded.input[0], 'Sub', 2)
    tail(subtraction.input[0], history, history_length, 128)
    total = node(subtraction.input[1], 'Add', 2)
    drums_bass = node(total.input[0], 'Add', 2)
    for selected, branch in zip([*drums_bass.input, total.input[1]], terminal.input[:3]):
        gathered = node(selected, 'Gather', 2)
        require(gathered.input[0] == branch and attribute(gathered, 'axis', 0) == 1
                and equal_index(gathered.input[1], 0), 'expected ((drums + bass) + vocals)')

    # A literal previous-input residual also requires the carried history to
    # end with the current audio chunk on every successful step.
    next_history = nodes.get('next_' + history)
    if history_length == 128 and next_history is not None and next_history.op_type == 'Identity':
        require(list(node('next_' + history, 'Identity', 1).input) == ['audio_chunk'],
                'history must retain the current input')
    else:
        sliced = node('next_' + history, 'Slice', 5)
        joined = node(sliced.input[0], 'Concat', 2)
        require(list(joined.input) == [history, 'audio_chunk']
                and attribute(joined, 'axis') in (-1, 2), 'history must append the current input')
        tail('next_' + history, sliced.input[0], history_length + 128, history_length)


def partition_model(source, destination):
    import onnx
    from onnx import helper, TensorProto
    from onnx.utils import Extractor
    source, destination = Path(source), Path(destination)
    if destination.exists():
        raise ValueError('Preserve existing partitions')
    # Deserialize exactly the bytes whose identity is recorded, including if
    # another process replaces the source during conversion.
    source_bytes = source.read_bytes()
    source_hash = hashlib.sha256(source_bytes).hexdigest()
    model = onnx.load_model_from_string(source_bytes)
    onnx.checker.check_model(model)
    if any(t.data_location == TensorProto.EXTERNAL for t in model.graph.initializer):
        raise ValueError('Require a self-contained parent')
    inputs, outputs = list(model.graph.input), list(model.graph.output)
    names = [v.name for v in inputs]
    out_names = [v.name for v in outputs]
    if (len(inputs) != 14 or len(outputs) != 14 or names[0] != 'audio_chunk'
            or out_names != ['separated_chunk', *('next_' + n for n in names[1:])]):
        raise ValueError('Require the thirteen-state independent parent interface')
    terminal = [n for n in model.graph.node if 'separated_chunk' in n.output]
    if (len(terminal) != 1 or terminal[0].op_type != 'Concat' or len(terminal[0].input) != 4
            or {a.name: helper.get_attribute_value(a) for a in terminal[0].attribute}.get('axis') != 1):
        raise ValueError('Require final DBV/Other concatenation along source axis')
    _validate_host_residual(model, terminal[0])
    # Expose already-existing branch tensors; never re-export/reassociate math.
    for tensor in terminal[0].input[:3]:
        if not any(v.name == tensor for v in model.graph.value_info):
            model.graph.value_info.append(helper.make_tensor_value_info(tensor, TensorProto.FLOAT, [1, 1, 2, 128]))
    prepared, records = [], []
    offset = 0
    for index, (stem, count) in enumerate(zip(SOURCES, COUNTS)):
        state_names = names[1 + offset:1 + offset + count]
        if not all(name.startswith(stem + '_') for name in state_names):
            raise ValueError('Parent state groups differ')
        branch_output = terminal[0].input[index]
        selected_inputs = ['audio_chunk', *state_names]
        selected_outputs = [branch_output, *('next_' + n for n in state_names)]
        child = Extractor(model).extract_model(selected_inputs, selected_outputs)
        # Check dependency closure independently: no other recurrent state may
        # feed this branch. Extractor preserves selected inputs even if unused.
        available = set(selected_inputs) | {t.name for t in child.graph.initializer}
        for node in child.graph.node:
            if any(name and name not in available for name in node.input):
                raise ValueError('Branch depends on another stem or missing input')
            available.update(node.output)
        child.graph.node.append(helper.make_node('Identity', [branch_output], ['separated_chunk'],
                                                  name='partition_output_identity'))
        child.graph.output[0].CopyFrom(helper.make_tensor_value_info('separated_chunk', TensorProto.FLOAT, [1, 1, 2, 128]))
        metadata = dict(schema='stemgenrt-independent-partition-v1', source=stem,
                        parent_sha256=source_hash, state_offset=offset, state_count=count,
                        operation_policy='Original parent nodes and initializer bytes; one output Identity added',
                        source_order=list(SOURCES), host_residual='previous_input - ((drums + bass) + vocals)')
        del child.metadata_props[:]
        prop = child.metadata_props.add(); prop.key = 'stemgenrt_partition'; prop.value = json.dumps(metadata, sort_keys=True, separators=(',', ':'))
        child.producer_name = 'StemgenRT independent graph partitioner'
        onnx.checker.check_model(child)
        prepared.append((stem, child))
        records.append(dict(**metadata, path=stem+'.onnx',
                            input_names=[v.name for v in child.graph.input],
                            output_names=[v.name for v in child.graph.output]))
        offset += count
    if sha(source) != source_hash:
        raise ValueError('Parent changed during partitioning')
    destination.mkdir(parents=True)
    for (stem, child), record in zip(prepared, records):
        path = destination/(stem+'.onnx'); onnx.save(child, path)
        record.update(sha256=sha(path), bytes=path.stat().st_size, nodes=len(child.graph.node))
    manifest = dict(schema='stemgenrt-independent-partitions-v1', parent_sha256=source_hash,
                    branches=records, weights_and_training_unchanged=True, timing_qualified=False)
    (destination/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    return manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source',type=Path);parser.add_argument('destination',type=Path)
    args=parser.parse_args();print(json.dumps(partition_model(args.source,args.destination),indent=2))

if __name__=='__main__':main()
