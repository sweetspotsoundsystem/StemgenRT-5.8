"""Graph partitioning retains original math and forbids cross-stem state dependencies."""
import json
import numpy as np
import pytest
onnx=pytest.importorskip('onnx')
ort=pytest.importorskip('onnxruntime')
from onnx import helper as h, TensorProto as T, numpy_helper
from stemgenrt.partition_independent import partition_model, SOURCES, COUNTS


def parent(path, cross_state=False, axis=1):
    names=[f'{stem}_audio_history' if i == 0 else f'{stem}_s{i}'
           for stem,count in zip(SOURCES,COUNTS) for i in range(count)]
    inputs=[h.make_tensor_value_info(name,T.FLOAT,[1,2,128]) for name in ['audio_chunk',*names]]
    outputs=[h.make_tensor_value_info('separated_chunk',T.FLOAT,[1,4,2,128]),
             *[h.make_tensor_value_info('next_'+name,T.FLOAT,[1,2,128]) for name in names]]
    nodes=[];initializers=[numpy_helper.from_array(np.array(value,np.int64),name)
        for name,value in [('axis',[1]),('zero',0),('start',[-128]),
                           ('end',[np.iinfo(np.int64).max]),('time_axis',[2]),('step',[1])]]
    wave=[]
    for i,stem in enumerate(SOURCES):
        initializers.append(numpy_helper.from_array(np.array(float(i+1),np.float32),stem+'_weight'))
        nodes.append(h.make_node('Mul',['audio_chunk',stem+'_weight'],[stem+'_audio']))
        current=stem+'_audio'
        if cross_state and stem=='drums':
            nodes.append(h.make_node('Add',[current,'bass_audio_history'],['cross_stem_audio']));current='cross_stem_audio'
        nodes.append(h.make_node('Unsqueeze',[current,'axis'],[stem+'_wave']))
        wave.append(stem+'_wave')
    for name in names:
        if name.endswith('_audio_history'):
            nodes.append(h.make_node('Identity',['audio_chunk'],['next_'+name]))
        else:
            nodes.append(h.make_node('Add',['audio_chunk',name],['next_'+name]))
    for stem in SOURCES:
        nodes.append(h.make_node('Gather',[stem+'_wave','zero'],[stem+'_selected'],axis=1))
    nodes.extend([
        h.make_node('Slice',['vocals_audio_history','start','end','time_axis','step'],['previous_audio']),
        h.make_node('Add',['drums_selected','bass_selected'],['drums_bass']),
        h.make_node('Add',['drums_bass','vocals_selected'],['sum_stems']),
        h.make_node('Sub',['previous_audio','sum_stems'],['other_audio']),
        h.make_node('Unsqueeze',['other_audio','axis'],['other_wave']),
        h.make_node('Concat',[*wave,'other_wave'],['separated_chunk'],axis=axis)])
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
        waves=[]
        for i,(session,b) in enumerate(zip(parts,result['branches'])):
            actual=session.run(None,dict(zip([x.name for x in session.get_inputs()],[audio,*independent[i]])))
            waves.append(actual[0][:,0])
            assert actual[0].tobytes()==expected[0][:,i:i+1].tobytes()
            for a,e in zip(actual[1:],expected[1+b['state_offset']:1+b['state_offset']+b['state_count']]): assert a.tobytes()==e.tobytes()
            independent[i]=actual[1:]
        other=states[9]-((waves[0]+waves[1])+waves[2])
        assert other.tobytes()==expected[0][:,3].tobytes()
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


@pytest.mark.parametrize('change', ['wrong_other', 'current_input', 'addition_order',
                                  'history_update', 'wrong_slice', 'wrong_source'])
def test_incompatible_host_residual_is_rejected_before_writing(tmp_path, change):
    source=tmp_path/'parent.onnx'; model=parent(source)
    nodes={output:node for node in model.graph.node for output in node.output}
    if change=='wrong_other': nodes['separated_chunk'].input[3]='drums_wave'
    elif change=='current_input': nodes['other_audio'].input[0]='audio_chunk'
    elif change=='addition_order':
        nodes['drums_bass'].input[:]=['bass_selected','vocals_selected']
        nodes['sum_stems'].input[1]='drums_selected'
    elif change=='history_update': nodes['next_vocals_audio_history'].input[0]='vocals_audio_history'
    elif change=='wrong_slice': nodes['previous_audio'].input[1]='end'
    elif change=='wrong_source': nodes['vocals_selected'].input[0]='bass_wave'
    onnx.save(model,source)
    with pytest.raises(ValueError,match='parent residual'):
        partition_model(source,tmp_path/'parts')
    assert not (tmp_path/'parts').exists()


def test_source_replaced_during_load_is_not_authenticated_as_original(tmp_path, monkeypatch):
    source=tmp_path/'parent.onnx'; parent(source)
    load=onnx.load_model_from_string
    def replace_after_load(data):
        model=load(data)
        replacement=onnx.ModelProto(); replacement.CopyFrom(model)
        replacement.producer_name='a different parent'
        onnx.save(replacement,source)
        return model
    monkeypatch.setattr(onnx,'load_model_from_string',replace_after_load)
    with pytest.raises(ValueError,match='Parent changed'):
        partition_model(source,tmp_path/'parts')
    assert not (tmp_path/'parts').exists()
