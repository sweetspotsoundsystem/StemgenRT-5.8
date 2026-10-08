"""Validate complete independent-stem native traces and paced worker timings."""
import array
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics
import struct
import sys


def sha(path):
    result=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):
            result.update(block)
    return result.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def require(condition,message):
    if not condition:
        raise ValueError(message)


def check_package(here):
    manifest=read(here/'manifest.json')
    for relative,digest in manifest['files'].items():
        original=here/relative
        path=original.resolve(strict=True)
        require(path.is_relative_to(here.resolve()) and not original.is_symlink() and path.is_file()
                and sha(path)==digest,'Package identity differs: '+relative)
    return manifest


def stats(values):
    require(values and all(math.isfinite(v) for v in values),'Invalid numeric samples')
    ordered=sorted(values)
    def percentile(fraction):
        index=(len(ordered)-1)*fraction
        a=int(index)
        b=min(a+1,len(ordered)-1)
        return ordered[a]+(index-a)*(ordered[b]-ordered[a])
    mean=statistics.fmean(values)
    return dict(mean_ms=mean,p50_ms=percentile(.5),p95_ms=percentile(.95),p99_ms=percentile(.99),
                p999_ms=percentile(.999),maximum_ms=ordered[-1],mean_fraction_of_hop=mean/(128000/44100))


def floats(path):
    require(sys.byteorder=='little','Require little-endian host')
    data=array.array('f')
    path=Path(path)
    if path.suffix=='.gz':
        with gzip.open(path,'rb') as stream:
            data.frombytes(stream.read())
    else:
        data.frombytes(path.read_bytes())
    return data


def compare_trace(path,protocol,case,here):
    actual=floats(path)
    keys=protocol['cases'][case]
    models=[protocol['models'][key] for key in keys]
    shapes=[[math.prod(shape) for shape in model['interface']['output_shapes']] for model in models]
    totals=[sum(sizes) for sizes in shapes]
    count=protocol['preflight_warmup_hops']+protocol['preflight_measured_hops']
    combined=protocol.get('case_assembly',{}).get(case,False)
    require(len(actual)==count*(sum(totals)+(512 if combined else 0)),'Native trace size differs')
    references=[floats(here/model['reference']) for model in models]
    require(all(len(values)==count*size for values,size in zip(references,totals)),'Reference trace size differs')
    limits=protocol['reference_tolerances']
    errors={key:{name:0. for name in model['interface']['output_names']} for key,model in zip(keys,models)}
    assembly=floats(here/protocol['assembly_reference']) if combined else None
    input_audio=floats(here/protocol['input_file']) if combined else None
    if combined:
        require(len(assembly)==count*512 and len(input_audio)>=count*256,'Assembly reference size differs')
        require([model['source_order'] for model in models]==[['drums'],['bass'],['vocals']],
                'Assembly source order differs')
        errors['assembly']={'other':0.,'physical_history':0.,'mixture_closure':0.}
    def f32(value):
        return struct.unpack('<f',struct.pack('<f',value))[0]
    cursor=0
    for hop in range(count):
        native_sources=[]
        for key,model,sizes,total,reference in zip(keys,models,shapes,totals,references):
            ref_cursor=hop*total
            if combined: native_sources.append(actual[cursor:cursor+256])
            for tensor,(name,size) in enumerate(zip(model['interface']['output_names'],sizes)):
                atol=limits['waveform_atol'] if tensor==0 else limits['state_atol']
                maximum=0.
                for i in range(size):
                    found,expected=actual[cursor+i],reference[ref_cursor+i]
                    error=abs(found-expected)
                    require(math.isfinite(found) and math.isfinite(expected)
                            and error<=atol+limits['rtol']*abs(expected),
                            f'Reference mismatch: {key}, {name}, hop {hop}, element {i}: {found} versus {expected}')
                    maximum=max(maximum,error)
                errors[key][name]=max(errors[key][name],maximum)
                cursor+=size
                ref_cursor+=size
        if combined:
            for i in range(256):
                other,expected=actual[cursor+i],assembly[hop*512+i]
                require(math.isfinite(other) and abs(other-expected)<=limits['waveform_atol']+limits['rtol']*abs(expected),
                        f'Residual Other differs at hop {hop}, element {i}')
                physical=input_audio[(hop-1)*256+i] if hop else 0.
                native_sum=f32(f32(native_sources[0][i]+native_sources[1][i])+native_sources[2][i])
                require(other==f32(physical-native_sum),'Other was not calculated from the delayed physical input and native DBV')
                history=actual[cursor+256+i]
                require(history==input_audio[hop*256+i] and history==assembly[hop*512+256+i],
                        'Physical input history copy changed')
                closure=abs(float(native_sum)+other-physical)
                require(closure<=1e-6,'Native physical mixture reconstruction failed')
                errors['assembly']['other']=max(errors['assembly']['other'],abs(other-expected))
                errors['assembly']['mixture_closure']=max(errors['assembly']['mixture_closure'],closure)
            cursor+=512
    return dict(status='pass',compared_hops=count,maximum_absolute_errors=errors,
                reference_scope='same numerical variant, each output and state, every preflight hop')


def validate_report(report,protocol,case,warmup,measured,check,*,expected_platform="macos"):
    require(report['runtime_platform']==expected_platform and report['status']=='completed' and report['paced'] and report['condition']=='qos'
        and report['graphs']==len(protocol['cases'][case]) and report['warmup_hops']==warmup
        and report['measured_hops']==measured and report['parity_trace'] is check
        and report['onnxruntime_version']==protocol['onnxruntime_version']
        and report['intra_op_threads']==report['inter_op_threads']==1
        and not report['spinning'] and not report['kleidiai']
        and report['provider']=='CPU' and report['execution']=='sequential'
        and report['preallocated_tensors'] and report['initial_default_policy'] and report['final_default_policy']
        and report['quality_measured'] is False and report['plugin_deadlines_qualified'] is False
        and report['physical_residual_other'] is protocol.get('case_assembly',{}).get(case,False)
        and report['graph_alignment_samples']==128 and report['host_queue_measured'] is False,
        'Native invocation, provider, policy or scope differs')


def analyze(output,package=None):
    output=Path(output)
    plan=read(output/'plan.json')
    protocol=plan['protocol']
    package=Path(__file__).resolve().parent if package is None else Path(package)
    require(check_package(package)==plan['package_manifest'],'Result package identity differs')
    for case in protocol['cases']:
        name='preflight-'+case
        preflight=read(output/(name+'.json'))
        validate_report(preflight,protocol,case,protocol['preflight_warmup_hops'],protocol['preflight_measured_hops'],True)
        calculated=compare_trace(output/(name+'.bin.trace.gz'),protocol,case,package)
        require(calculated==read(output/(name+'-parity.json')),'Preflight parity record differs')
    require(plan['chip'] in protocol['allowed_chips'] and plan['machine']=='arm64','Native Mac identity missing')
    rows_by_case={key:[] for key in protocol['cases']}
    rounds_by_case={key:[] for key in protocol['cases']}
    snapshots={key:set() for key in protocol['cases']}
    runtime_digests=set()
    mapping={'run':'run_wall_ms','loop':'loop_wall_ms','thread_cpu':'thread_cpu_ms',
             'start_lateness':'start_lateness_ms','completion':'completion_from_arrival_ms'}
    for index,order in enumerate(protocol['round_order'],1):
        require(set(order)==set(protocol['cases']) and len(order)==len(protocol['cases']),'Unbalanced round')
        for case in order:
            name=f'round{index}-{case}'
            report=read(output/(name+'.json'))
            validate_report(report,protocol,case,protocol['warmup_hops'],protocol['measured_hops_per_trial'],False)
            with gzip.open(output/(name+'.csv.gz'),'rt',newline='') as stream:
                rows=[{k:float(v) for k,v in row.items()} for row in csv.DictReader(stream)]
            require(len(rows)==protocol['measured_hops_per_trial'] and all(row['hop']==i for i,row in enumerate(rows)),
                    'Missing or repeated measured hop')
            for row in rows:
                require(all(math.isfinite(v) for v in row.values()),'Nonfinite timing sample')
                require(all(row[key]>=0 for key in mapping.values())
                        and abs(row['hop_budget_ms']-128000/44100)<1e-5
                        and row['run_wall_ms']<=row['loop_wall_ms']+1e-7
                        and abs(row['completion_from_arrival_ms']-(row['start_lateness_ms']+row['loop_wall_ms']))<1e-6,
                        'Timing coordinate inconsistency')
            misses=sum(r['completion_from_arrival_ms']>r['hop_budget_ms'] for r in rows)
            over=sum(r['loop_wall_ms']>r['hop_budget_ms'] for r in rows)
            require(misses==report['deadline_misses'] and over==report['compute_over_budget'],'Deadline accounting differs')
            for name_key,column in mapping.items():
                calculated=stats([r[column] for r in rows])
                require(all(math.isclose(value,report[name_key][key],rel_tol=1e-9,abs_tol=1e-7)
                            for key,value in calculated.items()),'Reported percentiles differ from raw samples')
            snapshots[case].add(sha(output/(name+'.bin')))
            runtime_digests.add(report['loaded_runtime_sha256'])
            rows_by_case[case].extend(rows)
            rounds_by_case[case].append(dict(round=index,deadline_misses=misses,compute_over_budget=over,
                loop=stats([r['loop_wall_ms'] for r in rows]),completion=stats([r['completion_from_arrival_ms'] for r in rows])))
    require(len(runtime_digests)==1 and all(len(v)==1 for v in snapshots.values()),'Runtime or repeat endpoint differs')
    expected_runtime={v for k,v in plan['package_manifest']['files'].items() if k.startswith('sdk/lib/')}
    require(runtime_digests<=expected_runtime,'Loaded runtime is outside pinned package')
    cases={}
    for case,rows in rows_by_case.items():
        require(len(rows)==protocol['measured_hops_per_case'],'Case exposure differs')
        cases[case]={key:stats([r[column] for r in rows]) for key,column in mapping.items()}
        cases[case].update(measured_hops=len(rows),deadline_misses=sum(r['completion_from_arrival_ms']>r['hop_budget_ms'] for r in rows),
            compute_over_budget=sum(r['loop_wall_ms']>r['hop_budget_ms'] for r in rows),rounds=rounds_by_case[case])
    return dict(schema='independent-stem-m4-worker-cost-results-v1',chip=plan['chip'],cases=cases,
        hop_budget_ms=128000/44100,quality_measured=False,plugin_deadlines_qualified=False,
        all_three_stems_trained=protocol.get('all_three_stems_trained',False),
        interpretation=protocol['interpretation'],three_model_case=protocol['three_model_case'])


def markdown(result):
    lines=[f"# Independent stem architecture worker cost on {result['chip']}",'',
        '| Case | Mean loop (ms) | p99 loop (ms) | p99 completion (ms) | Deadline misses |',
        '| --- | ---: | ---: | ---: | ---: |']
    for name,row in result['cases'].items():
        lines.append(f"| {name} | {row['loop']['mean_ms']:.4f} | {row['loop']['p99_ms']:.4f} | {row['completion']['p99_ms']:.4f} | {row['deadline_misses']} / {row['measured_hops']} |")
    return '\n'.join(lines+['',result['interpretation'],'',result['three_model_case'],
        '', 'Synthetic worker timing. Separation quality and plugin/DAW deadlines remain unqualified.',''])


if __name__=='__main__':
    result=analyze(Path(sys.argv[1]))
    print(markdown(result))
