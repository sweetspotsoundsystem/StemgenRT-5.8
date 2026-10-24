"""Graph partitioning retains original math and forbids cross-stem state dependencies."""
import json
import numpy as np
import pytest
onnx=pytest.importorskip('onnx')
ort=pytest.importorskip('onnxruntime')
from onnx import helper as h, TensorProto as T, numpy_helper
from stemgenrt.partition_independent import partition_model, SOURCES, COUNTS


def parent(path, cross_state=False, axis=1):
    names=[f'{stem}_s{i}' for stem,count in zip(SOURCES,COUNTS) for i in range(count)]
    inputs=[h.make_tensor_value_info(name,T.FLOAT,[1,2,128]) for name in ['audio_chunk',*names]]
    outputs=[h.make_tensor_value_info('separated_chunk',T.FLOAT,[1,4,2,128]),
             *[h.make_tensor_value_info('next_'+name,T.FLOAT,[1,2,128]) for name in names]]
    nodes=[];initializers=[numpy_helper.from_array(np.array([1],np.int64),'axis')]
    wave=[]
    for i,stem in enumerate(SOURCES):
        initializers.append(numpy_helper.from_array(np.array(float(i+1),np.float32),stem+'_weight'))
        nodes.append(h.make_node('Mul',['audio_chunk',stem+'_weight'],[stem+'_audio']))
        current=stem+'_audio'
        if cross_state and stem=='drums':
            nodes.append(h.make_node('Add',[current,'bass_s0'],['cross_stem_audio']));current='cross_stem_audio'
        nodes.append(h.make_node('Unsqueeze',[current,'axis'],[stem+'_wave']))
        wave.append(stem+'_wave')
    for name in names:nodes.append(h.make_node('Add',['audio_chunk',name],['next_'+name]))
    nodes.append(h.make_node('Concat',[*wave,wave[0]],['separated_chunk'],axis=axis))
    m=h.make_model(h.make_graph(nodes,'synthetic-independent',inputs,outputs,initializers),opset_imports=[h.make_opsetid('',17)])
    m.ir_version=8;onnx.save(m,path);return m


def test_original_nodes_and_initializer_bytes_are_preserved(tmp_path):
    source=tmp_path/'parent.onnx';original=parent(source)
    result=partition_model(source,tmp_path/'parts')
    old_nodes={tuple(n.output):n.SerializeToString() for n in original.graph.node}
    old_weights={n.name:n.SerializeToString() for n in original.graph.initializer}
    for branch in result['branches']:
        child=onnx.load(tmp_path/'parts'/branch['path'])
        assert all(n.SerializeToString()==old_nodes[tuple(n.output)] for n in child.graph.node[:-1])
        assert all(n.SerializeToString()==old_weights[n.name] for n in child.graph.initializer)
        assert child.graph.node[-1].op_type=='Identity'
        assert len(child.graph.input)==branch['state_count']+1
        assert json.loads(child.metadata_props[0].value)['parent_sha256']==result['parent_sha256']


def test_separate_state_trajectories_match_parent(tmp_path):
    source=tmp_path/'parent.onnx';parent(source);result=partition_model(source,tmp_path/'parts')
    opts=ort.SessionOptions();opts.intra_op_num_threads=1;opts.inter_op_num_threads=1
    whole=ort.InferenceSession(str(source),opts,providers=['CPUExecutionProvider'])
    parts=[ort.InferenceSession(str(tmp_path/'parts'/b['path']),opts,providers=['CPUExecutionProvider']) for b in result['branches']]
    rng=np.random.default_rng(39)
    states=[rng.normal(size=(1,2,128)).astype('float32') for _ in range(13)]
    independent=[[s.copy() for s in states[b['state_offset']:b['state_offset']+b['state_count']]] for b in result['branches']]
    for _ in range(4):
        audio=rng.normal(size=(1,2,128)).astype('float32')
        expected=whole.run(None,dict(zip([x.name for x in whole.get_inputs()],[audio,*states])))
        for i,(session,b) in enumerate(zip(parts,result['branches'])):
            actual=session.run(None,dict(zip([x.name for x in session.get_inputs()],[audio,*independent[i]])))
            assert actual[0].tobytes()==expected[0][:,i:i+1].tobytes()
            for a,e in zip(actual[1:],expected[1+b['state_offset']:1+b['state_offset']+b['state_count']]): assert a.tobytes()==e.tobytes()
            independent[i]=actual[1:]
        states=expected[1:]


def test_cross_stem_state_dependency_is_rejected(tmp_path):
    source=tmp_path/'parent.onnx';parent(source,cross_state=True)
    with pytest.raises(ValueError,match='depends'):partition_model(source,tmp_path/'parts')
    assert not (tmp_path/'parts').exists()


def test_wrong_output_axis_is_rejected(tmp_path):
    source=tmp_path/'parent.onnx';parent(source,axis=2)
    with pytest.raises(ValueError,match='concatenation'):partition_model(source,tmp_path/'parts')


def test_existing_partition_directory_is_preserved(tmp_path):
    source=tmp_path/'parent.onnx';parent(source);parts=tmp_path/'parts';parts.mkdir();(parts/'keep').write_text('keep')
    with pytest.raises(ValueError,match='Preserve'):partition_model(source,parts)
    assert (parts/'keep').read_text()=='keep'


def test_partition_bytes_are_deterministic(tmp_path):
    source=tmp_path/'parent.onnx';parent(source)
    a=partition_model(source,tmp_path/'a');b=partition_model(source,tmp_path/'b')
    assert a==b
