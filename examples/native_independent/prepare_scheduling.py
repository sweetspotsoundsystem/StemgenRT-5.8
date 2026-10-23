"""Freeze a same-graph scheduling experiment from an authenticated native packet."""
import argparse
from datetime import datetime, timezone
import itertools
from pathlib import Path
import shutil

if __package__:
    from .analyze import check_package, read, require, sha
    from .prepare import HERE, ROOT, verify_training_archive, write
else:
    from analyze import check_package, read, require, sha
    from prepare import HERE, ROOT, verify_training_archive, write


def prepare(args):
    source=args.source_packet.resolve(strict=True)
    manifest=check_package(source)
    require(sha(source/'manifest.json')==args.source_manifest_sha256,'Source manifest identity differs')
    original=read(source/'protocol.json')
    model=original['models'][args.model_key]
    require(model['sha256']==args.model_sha256 and sha(source/model['path'])==args.model_sha256,
            'Graph identity differs')
    require(model['trained'] and model['source_order']==['drums','bass','vocals','other']
            and len(model['interface']['input_names'])==14 and len(model['interface']['output_names'])==14,
            'Require the trained independent thirteen-state graph')
    require(model['reference_kind'].startswith('Native IndependentStems FP32')
            and sha(source/model['reference'])==model['reference_sha256'],
            'Independent PyTorch reference differs')
    require(original['onnxruntime_version']=='1.26.0' and original['hop_samples']==128
            and original['sample_rate']==44100 and original['synthetic_audio_only'], 'Probe contract differs')
    require(sha(source/original['input_file'])==original['input_sha256'],'Synthetic input identity differs')
    for path in (source/'sdk').rglob('*'):
        require(not path.is_symlink(),'Source SDK contains aliases')
        if path.is_file():
            require(str(path.relative_to(source)) in manifest['files'],'Source SDK has unauthenticated files')
    training=verify_training_archive(args.training_source,args.training_source_sha256)
    output=args.output.resolve()
    require(not output.exists(),'Use a new packet directory')
    output.mkdir(parents=True)
    for name in ('benchmark.cpp','scheduling.h','run_macos.py','analyze.py'):
        shutil.copyfile(HERE/name,output/name)
    shutil.copytree(source/'sdk',output/'sdk')
    # The official dylib uses this versioned install name. Keep regular files
    # for every alias, so package verification authenticates the loaded bytes.
    library=output/'sdk/lib/libonnxruntime.1.26.0.dylib'
    for name in ('libonnxruntime.1.dylib','libonnxruntime.dylib'):
        shutil.copyfile(library,library.parent/name)
    for relative in (model['path'],model['reference'],original['input_file']):
        require(relative in manifest['files'],'Input absent from source manifest')
        target=output/relative
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source/relative,target)
    (output/'training-source').mkdir()
    shutil.copyfile(args.training_source,output/'training-source'/args.training_source.name)
    shutil.copyfile(ROOT/'LICENSE',output/'LICENSE')
    names=['qos','realtime','workgroup']
    protocol={key:original[key] for key in ('onnxruntime_version','allowed_chips','sample_rate','hop_samples',
        'graph_alignment_samples','algorithmic_delay_samples','input_file','input_sha256',
        'synthetic_audio_only','preflight_warmup_hops','preflight_measured_hops','reference_tolerances')}
    protocol.update(schema='independent-stem-scheduling-m4-v1',prepared_utc=datetime.now(timezone.utc).isoformat(),
        cases={name:[args.model_key] for name in names},models={args.model_key:model},
        embedded_residual_cases={name:True for name in names},identical_graph_across_conditions=True,
        scheduling={name:dict(mode=name,computation_ns=2700000) for name in names},
        round_order=[list(order) for order in itertools.permutations(names)],
        warmup_hops=1024,measured_hops_per_trial=10000,measured_hops_per_case=60000,
        endurance_order=names,endurance_warmup_hops=1024,endurance_measured_hops=105000,
        input_period_hops=(output/original['input_file']).stat().st_size//1024,
        profiles={'qos':dict(warmup_hops=16,measured_hops=64)},all_three_stems_trained=True,
        quality_measured=False,plugin_deadlines_qualified=False,
        interpretation='Six order-balanced short trials per condition, then separate five-minute continuous-state trials. '
            'The unchanged graph uses one worker, no spinning and KleidiAI disabled. '
            'The 2.7 ms Mach computation estimate is frozen before measurement; every deadline stays on the original clock. '
            'The standalone audio workgroup is a scheduling experiment, not a host audio-unit workgroup. '
            'Operator profiling runs last and is excluded from deadline statistics. Endurance order is sequential and may confound thermal/time effects.',
        three_model_case='Identical released independent Drums/Bass/Vocals graph, thirteen states and physical residual Other in all three conditions.')
    require(protocol['input_period_hops']>=protocol['warmup_hops']+protocol['measured_hops_per_trial'],
            'Insufficient synthetic input')
    write(output/'protocol.json',protocol)
    write(output/'provenance.json',dict(source_manifest_sha256=args.source_manifest_sha256,
        source_protocol_sha256=sha(source/'protocol.json'),model_sha256=args.model_sha256,
        training_source_archive_sha256=sha(args.training_source),training_source_files_sha256=training,
        preparation_source_sha256=sha(__file__),graph_reexported=False,no_dataset_audio_opened=True))
    (output/'README.md').write_text('''# Same-graph M4 scheduling experiment

Use native arm64 Python 3.9+ and Xcode command-line tools on an Apple M4 or M4 Pro.
Keep the Mac on AC power, disable Low Power Mode, and record other workloads.
Do not change the packet. Allow roughly 30 minutes for the full run.

```sh
python3 run_macos.py --output /path/to/new-results --power-notes "AC; Low Power Mode off; actual workloads"
```

Return the result ZIP. The runner authenticates the model, input, SDK and
independent PyTorch traces, then tests QoS, Mach real-time and a standalone audio
workgroup with the same released model. It stops on policy or parity failure.
Every condition receives all six short trial orders and five minutes of
continuous recurrent state. A separate operator profile runs after timing.

These are worker measurements. Plugin callback and DAW qualification require
the separate plugin harness; no measured mean or scheduling policy implies a pass.
The matching model/training/export source archive is in training-source/.
''')
    require(all(not p.is_symlink() for p in output.rglob('*')),'Packet must contain only regular files')
    write(output/'manifest.json',dict(schema='independent-native-probe-package-v1',
        files={str(p.relative_to(output)):sha(p) for p in sorted(output.rglob('*')) if p.is_file()}))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('source-packet','training-source','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    for name in ('source-manifest-sha256','model-key','model-sha256','training-source-sha256'):
        parser.add_argument('--'+name,required=True)
    prepare(parser.parse_args())


if __name__=='__main__':
    main()
