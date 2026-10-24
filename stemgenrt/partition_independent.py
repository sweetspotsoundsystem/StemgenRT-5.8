"""Partition an existing independent FP32 graph without changing its operations.

Each child owns its original state inputs/outputs. The host reconstructs Other
from the previous physical input and the three completed stem outputs. This is
an experimental deployment layout; the matching model and training code remain
unchanged. It is not an export of new weights or a timing qualification.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
from pathlib import Path

SOURCES = ('drums', 'bass', 'vocals')
COUNTS = (5, 4, 4)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def partition_model(source, destination):
    import onnx
    from onnx import helper, TensorProto
    from onnx.utils import Extractor
    source, destination = Path(source), Path(destination)
    if destination.exists():
        raise ValueError('Preserve existing partitions')
    model = onnx.load(source, load_external_data=False)
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
    # Expose already-existing branch tensors; never re-export/reassociate math.
    for tensor in terminal[0].input[:3]:
        if not any(v.name == tensor for v in model.graph.value_info):
            model.graph.value_info.append(helper.make_tensor_value_info(tensor, TensorProto.FLOAT, [1, 1, 2, 128]))
    source_hash = sha(source)
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
