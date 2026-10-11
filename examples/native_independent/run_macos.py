#!/usr/bin/env python3
"""Run the complete independent-stem architecture probe on a physical M4 or M4 Pro."""
import argparse
from datetime import datetime,timezone
import gzip
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import traceback
import zipfile

from analyze import analyze,check_package,compare_trace,markdown,read,require,sha,validate_report
HERE=Path(__file__).resolve().parent


def utc():
    return datetime.now(timezone.utc).isoformat()


def write(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def capture(args,optional=False):
    result=subprocess.run(args,capture_output=True,text=True,timeout=30)
    if result.returncode and not optional:
        result.check_returncode()
    return result.stdout.strip() if result.returncode==0 else 'unavailable: '+result.stderr.strip()


def compress(path):
    with path.open('rb') as source,gzip.open(str(path)+'.gz','wb') as target:
        shutil.copyfileobj(source,target)
    path.unlink()


def package_results(output):
    files=sorted(p for p in output.iterdir() if p.is_file() and p.name not in ('benchmark','result-files.json'))
    write(output/'result-files.json',{p.name:sha(p) for p in files})
    archive=output.with_suffix('.zip')
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as target:
        for path in sorted(output.iterdir()):
            if path.is_file() and path.name!='benchmark':
                target.write(path,output.name+'/'+path.name)
    return archive


def run(args,output,manifest):
    protocol=read(HERE/'protocol.json')
    chip=capture(['sysctl','-n','machdep.cpu.brand_string'])
    require(chip in protocol['allowed_chips'],'This probe requires Apple M4 or Apple M4 Pro')
    host={key:capture(['sysctl','-n',key],optional=True) for key in
          ('hw.model','hw.memsize','hw.physicalcpu','hw.logicalcpu','hw.perflevel0.physicalcpu','hw.perflevel1.physicalcpu')}
    write(output/'plan.json',dict(status='frozen_before_measurement',frozen_utc=utc(),chip=chip,host=host,
        machine=platform.machine(),macos=capture(['sw_vers']),compiler=capture(['xcrun','clang++','--version']),
        power_notes=args.power_notes,protocol=protocol,package_manifest=manifest,
        package_manifest_sha256=sha(HERE/'manifest.json'),python=sys.version))
    write(output/'power-before.json',dict(settings=capture(['pmset','-g','custom'],optional=True),
        source=capture(['pmset','-g','batt'],optional=True),thermal=capture(['pmset','-g','therm'],optional=True)))
    binary=output/'benchmark'
    command=['xcrun','clang++','-std=c++20','-O3','-DNDEBUG','-fblocks','-mmacosx-version-min=14.0',
        '-Wall','-Wextra','-Werror',str(HERE/'benchmark.cpp'),'-I'+str(HERE/'sdk/include'),
        '-L'+str(HERE/'sdk/lib'),'-Wl,-rpath,'+str(HERE/'sdk/lib'),'-lonnxruntime',
        '-framework','AudioToolbox','-pthread','-o',str(binary)]
    write(output/'compile-command.json',command)
    result=subprocess.run(command,capture_output=True,text=True,timeout=180)
    (output/'compile.log').write_text(result.stdout+result.stderr)
    result.check_returncode()
    write(output/'binary.json',dict(sha256=sha(binary),dependencies=capture(['otool','-L',str(binary)])))

    def trial(name,case,warmup,measured,check):
        print(f'Running {name}: {measured:,} measured hops',flush=True)
        samples,snapshot=output/(name+'.csv'),output/(name+'.bin')
        models=[HERE/protocol['models'][key]['path'] for key in protocol['cases'][case]]
        action=('check' if check else 'measure')+('_combined' if protocol.get('case_assembly',{}).get(case,False) else '')
        command=[str(binary),str(HERE/protocol['input_file']),str(warmup),str(measured),
                 action,str(samples),str(snapshot),*[str(p) for p in models]]
        started=utc()
        result=subprocess.run(command,capture_output=True,text=True,timeout=60+3*(warmup+measured)*128/44100)
        (output/(name+'.stdout')).write_text(result.stdout)
        (output/(name+'.stderr')).write_text(result.stderr)
        write(output/(name+'-execution.json'),dict(command=command,started_utc=started,ended_utc=utc(),returncode=result.returncode))
        result.check_returncode()
        report=json.loads(result.stdout)
        validate_report(report,protocol,case,warmup,measured,check)
        library=Path(report['loaded_runtime']).resolve(strict=True)
        require(library.is_relative_to(HERE/'sdk/lib'),'Loaded runtime is outside pinned SDK')
        relative=str(library.relative_to(HERE))
        require(relative in manifest['files'] and sha(library)==manifest['files'][relative],'Loaded runtime identity differs')
        report['loaded_runtime_sha256']=sha(library)
        write(output/(name+'.json'),report)
        if check:
            trace=Path(str(snapshot)+'.trace')
            parity=compare_trace(trace,protocol,case,HERE)
            write(output/(name+'-parity.json'),parity)
            compress(trace)
        compress(samples)

    for case in protocol['cases']:
        trial('preflight-'+case,case,protocol['preflight_warmup_hops'],protocol['preflight_measured_hops'],True)
    print('All native preflights match their waveform and state references.',flush=True)
    for index,order in enumerate(protocol['round_order'],1):
        for case in order:
            trial(f'round{index}-{case}',case,protocol['warmup_hops'],protocol['measured_hops_per_trial'],False)
    require(check_package(HERE)==manifest,'Package changed during measurement')
    summary=analyze(output)
    write(output/'summary.json',summary)
    (output/'SUMMARY.md').write_text(markdown(summary))
    write(output/'power-after.json',dict(source=capture(['pmset','-g','batt'],optional=True),
        thermal=capture(['pmset','-g','therm'],optional=True)))
    write(output/'completion.json',dict(status='cost_probe_completed',completed_utc=utc(),
        quality_measured=False,plugin_deadlines_qualified=False))
    print(markdown(summary),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True,help='New result directory, without a file extension')
    parser.add_argument('--power-notes',required=True,help='Actual power source/mode and other active workloads')
    args=parser.parse_args()
    if sys.platform!='darwin' or platform.machine()!='arm64':
        parser.error('Run with native arm64 Python on the target physical Mac')
    manifest=check_package(HERE)
    output=args.output.resolve()
    if output.suffix or output.exists() or output.with_suffix('.zip').exists():
        parser.error('Choose a new directory name without an extension; existing results are preserved')
    output.mkdir(parents=True,exist_ok=False)
    caffeine=None
    success=False
    try:
        caffeine=subprocess.Popen(['caffeinate','-i','-w',str(os.getpid())])
        run(args,output,manifest)
        success=True
    except (Exception,KeyboardInterrupt):
        (output/'failure.txt').write_text(traceback.format_exc())
        print('Stopped; diagnostics saved in '+str(output/'failure.txt'),file=sys.stderr)
    finally:
        if caffeine is not None:
            caffeine.terminate()
            caffeine.wait(timeout=10)
        archive=package_results(output)
        print(('Results' if success else 'Diagnostic')+' ZIP: '+str(archive),flush=True)
    return 0 if success else 1


if __name__=='__main__':
    raise SystemExit(main())
