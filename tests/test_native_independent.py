"""Synthetic result records verify analyzer rejection paths, not hardware timing."""
import array
import copy
import csv
import gzip
import io
import json
import struct
import tarfile
from pathlib import Path
import sys

import pytest
from examples.native_independent.analyze import analyze,compare_trace,sha,stats
from examples.native_independent.prepare import verify_training_archive


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value))


def report(protocol,warmup,measured,check):
    return dict(status='completed',runtime_platform='macos',paced=True,condition='qos',graphs=1,
        warmup_hops=warmup,measured_hops=measured,parity_trace=check,onnxruntime_version='1.26.0',
        intra_op_threads=1,inter_op_threads=1,spinning=False,kleidiai=False,provider='CPU',execution='sequential',
        preallocated_tensors=True,initial_default_policy=True,final_default_policy=True,
        quality_measured=False,plugin_deadlines_qualified=False,physical_residual_other=False,
        graph_alignment_samples=128,host_queue_measured=False)


@pytest.fixture
def synthetic(tmp_path):
    package=tmp_path/'toy-package'
    output=tmp_path/'toy-results'
    package.mkdir();output.mkdir()
    (package/'sdk/lib').mkdir(parents=True)
    (package/'sdk/lib/toy.dylib').write_bytes(b'unit test placeholder; not an executable')
    golden=array.array('f',[0.]*25)
    (package/'reference.f32').write_bytes(golden.tobytes())
    manifest={'files':{name:sha(package/name) for name in ('sdk/lib/toy.dylib','reference.f32')}}
    write(package/'manifest.json',manifest)
    protocol={'allowed_chips':['Apple M4'],'cases':{'toy':['toy-model']},
        'models':{'toy-model':{'reference':'reference.f32','interface':{
            'output_shapes':[[1]]*5,'output_names':['audio','history','local','global','tail']}}},
        'reference_tolerances':{'waveform_atol':1e-5,'state_atol':5e-5,'rtol':2e-5},
        'preflight_warmup_hops':2,'preflight_measured_hops':3,
        'warmup_hops':2,'measured_hops_per_trial':3,'measured_hops_per_case':6,
        'round_order':[['toy'],['toy']],'onnxruntime_version':'1.26.0',
        'interpretation':'Synthetic unit test records, not measurements','three_model_case':'unit test'}
    write(output/'plan.json',dict(protocol=protocol,chip='Apple M4',machine='arm64',package_manifest=manifest))
    pre=report(protocol,2,3,True)
    write(output/'preflight-toy.json',pre)
    with gzip.open(output/'preflight-toy.bin.trace.gz','wb') as stream:
        stream.write(golden.tobytes())
    write(output/'preflight-toy-parity.json',compare_trace(output/'preflight-toy.bin.trace.gz',protocol,'toy',package))
    columns={'run':'run_wall_ms','loop':'loop_wall_ms','thread_cpu':'thread_cpu_ms',
        'start_lateness':'start_lateness_ms','completion':'completion_from_arrival_ms'}
    rows=[]
    for index,loop in enumerate((.2,.3,3.1)):
        rows.append(dict(hop=index,run_wall_ms=loop-.1,loop_wall_ms=loop,thread_cpu_ms=loop-.05,
            start_lateness_ms=.1,completion_from_arrival_ms=loop+.1,hop_budget_ms=128000/44100))
    for index in (1,2):
        name=f'round{index}-toy'
        current=report(protocol,2,3,False)
        current.update(deadline_misses=1,compute_over_budget=1,
            loaded_runtime_sha256=manifest['files']['sdk/lib/toy.dylib'])
        current.update({key:stats([r[column] for r in rows]) for key,column in columns.items()})
        write(output/(name+'.json'),current)
        (output/(name+'.bin')).write_bytes(b'fixed synthetic endpoint')
        with gzip.open(output/(name+'.csv.gz'),'wt',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
            writer.writeheader();writer.writerows(rows)
    return package,output


def test_valid_synthetic_records_reaggregate_exact_exposure(synthetic):
    package,output=synthetic
    result=analyze(output,package)
    assert result['cases']['toy']['measured_hops']==6
    assert result['cases']['toy']['deadline_misses']==2
    assert result['cases']['toy']['loop']['mean_ms']==pytest.approx(1.2)
    assert result['quality_measured'] is result['plugin_deadlines_qualified'] is False


@pytest.mark.parametrize('mutation',['portable','wrong_count','missing_preflight','corrupt_trace','changed_endpoint','wrong_statistics','timing_coordinates'])
def test_synthetic_invalid_evidence_is_rejected(synthetic,mutation):
    package,output=synthetic
    path=output/'round2-toy.json'
    report=json.loads(path.read_text())
    if mutation=='portable':
        report['runtime_platform']='portable'
    elif mutation=='wrong_count':
        report['deadline_misses']=0
    elif mutation=='missing_preflight':
        (output/'preflight-toy.json').unlink()
    elif mutation=='corrupt_trace':
        bad=array.array('f',[0.]*25)
        bad[-1]=.1
        with gzip.open(output/'preflight-toy.bin.trace.gz','wb') as stream:
            stream.write(bad.tobytes())
    elif mutation=='changed_endpoint':
        (output/'round2-toy.bin').write_bytes(b'different synthetic endpoint')
    elif mutation=='wrong_statistics':
        report['loop']['p99_ms']+=1
    elif mutation=='timing_coordinates':
        with gzip.open(output/'round2-toy.csv.gz','rt') as stream:
            rows=list(csv.DictReader(stream))
        rows[0]['completion_from_arrival_ms']='0'
        with gzip.open(output/'round2-toy.csv.gz','wt',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    write(path,report)
    with pytest.raises((ValueError,FileNotFoundError)):
        analyze(output,package)


@pytest.fixture
def combined_trace(tmp_path):
    def f32(value):
        return struct.unpack('<f',struct.pack('<f',value))[0]
    case='combined'
    sources=('drums','bass','vocals')
    models={source:{'source_order':[source],'reference':source+'.f32','interface':{
        'output_shapes':[[1,1,2,128],[1]],'output_names':['audio','state']}} for source in sources}
    protocol={'cases':{case:list(sources)},'case_assembly':{case:True},'models':models,
        'preflight_warmup_hops':1,'preflight_measured_hops':2,
        'reference_tolerances':{'waveform_atol':1e-5,'state_atol':5e-5,'rtol':2e-5},
        'input_file':'input.f32','assembly_reference':'assembly.f32'}
    audio=array.array('f',[value for value in (.3,.4,.5) for _ in range(256)])
    (tmp_path/'input.f32').write_bytes(audio.tobytes())
    reference={source:array.array('f') for source in sources}
    trace,assembly=array.array('f'),array.array('f')
    for hop in range(3):
        values=[f32(.02*(hop+1)),f32(-.03*(hop+1)),f32(.12+.01*hop)]
        for source,value in zip(sources,values):
            tensors=array.array('f',[value]*256+[.25])
            reference[source].extend(tensors)
            trace.extend(tensors)
        physical=audio[(hop-1)*256] if hop else 0.
        other=f32(physical-f32(f32(values[0]+values[1])+values[2]))
        extra=array.array('f',[other]*256)+audio[hop*256:(hop+1)*256]
        assembly.extend(extra);trace.extend(extra)
    for source,values in reference.items():
        (tmp_path/(source+'.f32')).write_bytes(values.tobytes())
    (tmp_path/'assembly.f32').write_bytes(assembly.tobytes())
    path=tmp_path/'trace.f32';path.write_bytes(trace.tobytes())
    return tmp_path,protocol,case,path


def test_combined_trace_checks_delayed_other_and_exact_physical_history(combined_trace):
    package,protocol,case,path=combined_trace
    result=compare_trace(path,protocol,case,package)
    assert result['status']=='pass' and result['compared_hops']==3
    assert result['maximum_absolute_errors']['assembly']['other']==0.
    assert result['maximum_absolute_errors']['assembly']['physical_history']==0.


@pytest.mark.parametrize('mutation',['other_below_tolerance','history_below_tolerance','source_order','truncated'])
def test_combined_assembly_corruption_is_rejected(combined_trace,mutation):
    package,protocol,case,path=combined_trace
    data=array.array('f');data.frombytes(path.read_bytes())
    if mutation=='other_below_tolerance':data[3*257]+=1e-7
    if mutation=='history_below_tolerance':data[3*257+256]+=1e-7
    if mutation=='source_order':protocol['cases'][case].reverse()
    if mutation=='truncated':data.pop()
    path.write_bytes(data.tobytes())
    with pytest.raises(ValueError):
        compare_trace(path,protocol,case,package)


@pytest.mark.parametrize('mutation',[None,'missing_trainer','changed_trainer','wrong_archive_sha'])
def test_matching_training_archive_is_required(tmp_path,mutation):
    root=tmp_path/'source'
    files={'stemgenrt/trainer.py':b'actual trainer\n','stemgenrt/specialist.py':b'actual model\n',
        'stemgenrt/metadata.json':b'{}\n','train_streaming.py':b'entrypoint\n','pyproject.toml':b'project\n',
        'LICENSE':b'license\n','configs/bass-specialist.json':b'{}\n','configs/drums-specialist.json':b'{}\n'}
    for relative,data in files.items():
        path=root/relative;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
    path=tmp_path/'source.tar.gz'
    with tarfile.open(path,'w:gz') as archive:
        for relative,data in files.items():
            if relative=='stemgenrt/trainer.py':
                if mutation=='missing_trainer':continue
                if mutation=='changed_trainer':data=b'wrong! trainer\n'
            member=tarfile.TarInfo('stemgenrt-test/'+relative);member.size=len(data)
            archive.addfile(member,io.BytesIO(data))
    digest='0'*64 if mutation=='wrong_archive_sha' else sha(path)
    if mutation is None:
        result=verify_training_archive(path,digest,root=root)
        assert set(result)==set(files)
    else:
        with pytest.raises(ValueError,match='Training archive'):
            verify_training_archive(path,digest,root=root)
