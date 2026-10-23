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
    path=Path(path)
    if path.suffix=='.gz':
        with gzip.open(path,'rt') as stream:
            return json.load(stream)
    return json.loads(path.read_text())


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


def validate_report(report,protocol,case,warmup,measured,check,*,expected_platform="macos",repeat_input=False,profiled=False):
    scheduling=protocol.get('scheduling',{}).get(case,{'mode':'qos','computation_ns':2000000})
    mode=scheduling['mode']
    require(mode in ('qos','realtime','workgroup'),'Unknown scheduling condition')
    require(report.get('profiled',False) is profiled
            and bool(report.get('operator_profiles',[])) is profiled,
            'Instrumented profiling must be separate from timing evidence')
    require(report['runtime_platform']==expected_platform and report['status']=='completed' and report['paced'] and report['condition']==mode
        and report['graphs']==len(protocol['cases'][case]) and report['warmup_hops']==warmup
        and report['measured_hops']==measured and report['parity_trace'] is check
        and report['onnxruntime_version']==protocol['onnxruntime_version']
        and report['intra_op_threads']==report['inter_op_threads']==1
        and not report['spinning'] and not report['kleidiai']
        and report['provider']=='CPU' and report['execution']=='sequential'
        and report['preallocated_tensors'] and report['initial_default_policy'] is (mode=='qos')
        and report['final_default_policy'] is (mode=='qos')
        and report['quality_measured'] is False and report['plugin_deadlines_qualified'] is False
        and report['physical_residual_other'] is protocol.get('case_assembly',{}).get(case,False)
        and report.get('graph_embedded_residual_other',False) is protocol.get('embedded_residual_cases',{}).get(case,False)
        and report.get('input_repeated',False) is repeat_input
        and (not repeat_input or report.get('input_period_hops')==protocol['input_period_hops'])
        and report['graph_alignment_samples']==128 and report['host_queue_measured'] is False,
        'Native invocation, provider, policy or scope differs')
    if 'scheduling' in protocol:
        require(set(protocol['scheduling'])==set(protocol['cases']), 'Incomplete scheduling protocol')
        budget=scheduling['computation_ns']
        require(type(budget) is int and 0<budget<128000000000//44100
                and report['requested_computation_ns']==budget,'CPU estimate differs')
        numer,denom=report['mach_timebase_numer'],report['mach_timebase_denom']
        require(type(numer) is int and type(denom) is int and numer>0 and denom>0,'Invalid Mach timebase')
        for name in ('initial_policy','final_policy'):
            policy=report[name]
            if mode!='qos':
                period=(128000000000//44100)*denom//numer
                require(policy==dict(period_ticks=period,computation_ticks=budget*denom//numer,
                    constraint_ticks=period,preemptible=True),'Real-time policy readback differs')
        intervals=warmup+measured if mode=='workgroup' else 0
        require(report['workgroup_interval_starts']==report['workgroup_interval_finishes']==intervals,
                'Incomplete or unexpected audio-workgroup intervals')


def summarize_profile(path,expected_runs):
    """Operator attribution only: instrumentation invalidates deadline timings."""
    events=read(path)
    require(isinstance(events,list),'Invalid ORT profile')
    runs=[event for event in events if event.get('cat')=='Session' and event.get('name')=='model_run']
    require(len(runs)==expected_runs,'Operator profile has incomplete graph exposure')
    totals={}
    nodes={}
    for event in events:
        if event.get('cat')!='Node' or not event.get('name','').endswith('_kernel_time'):
            continue
        args=event.get('args',{})
        require(args.get('provider')=='CPUExecutionProvider','Profile provider differs')
        duration=event.get('dur')
        require(isinstance(duration,(int,float)) and math.isfinite(duration) and duration>=0,
                'Invalid operator duration')
        op=args.get('op_name')
        require(isinstance(op,str) and op,'Missing operator identity')
        for mapping,key in ((totals,op),(nodes,event['name'])):
            row=mapping.setdefault(key,dict(calls=0,total_us=0.))
            row['calls']+=1
            row['total_us']+=duration
    require(totals,'Empty operator profile')
    total=sum(row['total_us'] for row in totals.values())
    require(total>0,'Empty operator durations')
    for mapping in (totals,nodes):
        for row in mapping.values():
            row.update(calls_per_run=row['calls']/expected_runs,mean_us_per_run=row['total_us']/expected_runs,
                       fraction_of_instrumented_node_time=row['total_us']/total)
    return dict(scope='Instrumented operator attribution; no deadline or plugin qualification',
        graph_runs=expected_runs,includes_warmup=True,operators=totals,
        top_nodes=dict(sorted(nodes.items(),key=lambda item:item[1]['total_us'],reverse=True)[:40]))


TIMING_COLUMNS={'run':'run_wall_ms','loop':'loop_wall_ms','thread_cpu':'thread_cpu_ms',
                'start_lateness':'start_lateness_ms','completion':'completion_from_arrival_ms'}


def validate_samples(output,name,report,measured):
    mapping=TIMING_COLUMNS
    with gzip.open(output/(name+'.csv.gz'),'rt',newline='') as stream:
        rows=[{k:float(v) for k,v in row.items()} for row in csv.DictReader(stream)]
    require(len(rows)==measured and all(row['hop']==i for i,row in enumerate(rows)),
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
    return rows,misses,over


def validate_endpoint(path,protocol,case):
    expected_size=4*sum(math.prod(shape) for key in protocol['cases'][case]
                        for shape in protocol['models'][key]['interface']['output_shapes'])
    if protocol.get('case_assembly',{}).get(case,False):expected_size+=512*4
    require(path.stat().st_size==expected_size,'Endpoint size differs')
    require(all(math.isfinite(x) for x in floats(path)),'Nonfinite endpoint')
    return sha(path)


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
    mapping=TIMING_COLUMNS
    for index,order in enumerate(protocol['round_order'],1):
        require(set(order)==set(protocol['cases']) and len(order)==len(protocol['cases']),'Unbalanced round')
        for case in order:
            name=f'round{index}-{case}'
            report=read(output/(name+'.json'))
            validate_report(report,protocol,case,protocol['warmup_hops'],protocol['measured_hops_per_trial'],False)
            rows,misses,over=validate_samples(output,name,report,protocol['measured_hops_per_trial'])
            snapshots[case].add(validate_endpoint(output/(name+'.bin'),protocol,case))
            runtime_digests.add(report['loaded_runtime_sha256'])
            rows_by_case[case].extend(rows)
            rounds_by_case[case].append(dict(round=index,deadline_misses=misses,compute_over_budget=over,
                loop=stats([r['loop_wall_ms'] for r in rows]),completion=stats([r['completion_from_arrival_ms'] for r in rows])))
    endurance={}
    if 'endurance_order' in protocol:
        order=protocol['endurance_order']
        require(set(order)==set(protocol['cases']) and len(order)==len(protocol['cases']),
                'Unbalanced endurance exposure')
        period=protocol['input_period_hops']
        require(isinstance(period,int) and period>0
                and (package/protocol['input_file']).stat().st_size==period*256*4,
                'Endurance input period differs')
        for case in order:
            name='endurance-'+case
            report=read(output/(name+'.json'))
            validate_report(report,protocol,case,protocol['endurance_warmup_hops'],
                            protocol['endurance_measured_hops'],False,repeat_input=True)
            rows,misses,over=validate_samples(output,name,report,protocol['endurance_measured_hops'])
            snapshot=output/(name+'.bin')
            endpoint_digest=validate_endpoint(snapshot,protocol,case)
            runtime_digests.add(report['loaded_runtime_sha256'])
            endurance[case]={key:stats([r[column] for r in rows]) for key,column in mapping.items()}
            endurance[case].update(measured_hops=len(rows),deadline_misses=misses,compute_over_budget=over,
                input_period_hops=period,continuous_state=True,endpoint_sha256=endpoint_digest,
                nominal_audio_seconds=len(rows)*128/44100)
    require(len(runtime_digests)==1 and all(len(v)==1 for v in snapshots.values()),'Runtime or repeat endpoint differs')
    expected_runtime={v for k,v in plan['package_manifest']['files'].items() if k.startswith('sdk/lib/')}
    require(runtime_digests<=expected_runtime,'Loaded runtime is outside pinned package')
    if protocol.get('identical_graph_across_conditions',False):
        require(len({tuple(keys) for keys in protocol['cases'].values()})==1,'Scheduling graphs differ')
        require(len(set.union(*snapshots.values()))==1,'Scheduling conditions changed final tensors')
        if endurance:
            require(len({row['endpoint_sha256'] for row in endurance.values()})==1,
                    'Scheduling conditions changed endurance final tensors')
    profiles={}
    for case,counts in protocol.get('profiles',{}).items():
        name='profile-'+case
        report=read(output/(name+'.json'))
        validate_report(report,protocol,case,counts['warmup_hops'],counts['measured_hops'],False,profiled=True)
        require(report['loaded_runtime_sha256'] in runtime_digests,'Profile runtime differs')
        require(len(report['operator_profiles'])==1,'Expected one operator profile')
        calculated=summarize_profile(output/(name+'-operators.json.gz'),counts['warmup_hops']+counts['measured_hops'])
        require(calculated==read(output/(name+'-summary.json')),'Operator profile summary differs')
        profiles[case]=calculated
    cases={}
    for case,rows in rows_by_case.items():
        require(len(rows)==protocol['measured_hops_per_case'],'Case exposure differs')
        cases[case]={key:stats([r[column] for r in rows]) for key,column in mapping.items()}
        cases[case].update(measured_hops=len(rows),deadline_misses=sum(r['completion_from_arrival_ms']>r['hop_budget_ms'] for r in rows),
            compute_over_budget=sum(r['loop_wall_ms']>r['hop_budget_ms'] for r in rows),rounds=rounds_by_case[case])
    return dict(schema='independent-stem-m4-worker-cost-results-v1',chip=plan['chip'],cases=cases,endurance=endurance,
        operator_profiles=profiles,
        hop_budget_ms=128000/44100,quality_measured=False,plugin_deadlines_qualified=False,
        all_three_stems_trained=protocol.get('all_three_stems_trained',False),
        interpretation=protocol['interpretation'],three_model_case=protocol['three_model_case'])


def markdown(result):
    lines=[f"# Independent stem architecture worker cost on {result['chip']}",'',
        '| Case | Mean loop (ms) | p99 loop (ms) | p99 completion (ms) | Deadline misses |',
        '| --- | ---: | ---: | ---: | ---: |']
    for name,row in result['cases'].items():
        lines.append(f"| {name} | {row['loop']['mean_ms']:.4f} | {row['loop']['p99_ms']:.4f} | {row['completion']['p99_ms']:.4f} | {row['deadline_misses']} / {row['measured_hops']} |")
    if result.get('endurance'):
        lines+=['','Continuous endurance trials (input repeats; model state continues):','',
            '| Case | Seconds | Mean loop (ms) | p99 completion (ms) | Deadline misses |',
            '| --- | ---: | ---: | ---: | ---: |']
        for name,row in result['endurance'].items():
            lines.append(f"| {name} | {row['nominal_audio_seconds']:.1f} | {row['loop']['mean_ms']:.4f} | {row['completion']['p99_ms']:.4f} | {row['deadline_misses']} / {row['measured_hops']} |")
    return '\n'.join(lines+['',result['interpretation'],'',result['three_model_case'],
        '', 'Synthetic worker timing. Separation quality and plugin/DAW deadlines remain unqualified.',''])


if __name__=='__main__':
    result=analyze(Path(sys.argv[1]))
    print(markdown(result))
