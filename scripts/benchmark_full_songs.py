#!/usr/bin/env python3
"""Render and score complete 44.1 kHz stereo songs with an authenticated ONNX graph."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import soundfile as sf
from stemgenrt.fullsong import SOURCES, aggregate, json_metrics, score


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def load(path):
    audio, rate = sf.read(path, dtype='float64', always_2d=True)
    if rate != 44100 or audio.shape[1] != 2 or not len(audio) or not np.isfinite(audio).all():
        raise ValueError('Require nonempty finite 44.1 kHz stereo audio: ' + str(path))
    return audio


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-root', type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--onnx', type=Path)
    mode.add_argument('--estimate-root', type=Path)
    parser.add_argument('--sha256', help='Required model identity for --onnx')
    parser.add_argument('--whole-song-only', action='store_true')
    parser.add_argument('--output', type=Path, required=True, help='New evidence directory')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('--output must not already exist')
    if args.onnx and not args.sha256:
        parser.error('--onnx requires --sha256')
    if not args.onnx and args.sha256:
        parser.error('--sha256 applies only to --onnx')
    songs = sorted(p for p in args.reference_root.iterdir() if p.is_dir())
    if not songs:
        parser.error('No song directories in --reference-root')
    separator = None
    if args.onnx:
        from stemgenrt.streaming import StreamingSeparator
        separator = StreamingSeparator(args.onnx, expected_sha256=args.sha256)
    versions = {name: importlib.metadata.version(name) for name in
                (['numpy', 'soundfile'] + ([] if args.whole_song_only else ['museval'])
                 + (['onnxruntime'] if args.onnx else []))}
    args.output.mkdir(parents=True)
    protocol = {'sample_rate': 44100, 'source_order': list(SOURCES),
                'complete_songs': True, 'model_sha256': args.sha256,
                'graph_alignment_samples': 128 if separator else 'already aligned estimates required',
                'partial_last_hop': 'zero pad once', 'graph_flush_hops': 1,
                'gain_fitting': False, 'delay_search': False, 'source_permutation': False,
                'development_activity_gate': False, 'versions': versions,
                'whole_song_energy_delta': 1e-7, 'museval': not args.whole_song_only,
                'museval_window_hop': 44100, 'museval_filter_length': 512,
                'museval_framewise_filters': False, 'includes_plugin_callback_queue': False,
                'declared_songs': [p.name for p in songs]}
    (args.output / 'protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
    records = []
    for index, folder in enumerate(songs):
        references = [load(folder / (source + '.wav')) for source in SOURCES]
        if len({audio.shape for audio in references}) != 1:
            raise ValueError('Reference lengths differ: ' + folder.name)
        reference = np.stack(references)
        inputs = {source: digest(folder / (source + '.wav')) for source in SOURCES}
        if separator:
            mixture = load(folder / 'mixture.wav')
            if mixture.shape != references[0].shape:
                raise ValueError('Mixture length differs: ' + folder.name)
            inputs['mixture'] = digest(folder / 'mixture.wav')
            estimate = separator.separate(np.ascontiguousarray(mixture.T, dtype=np.float32))
            estimate = estimate.transpose(0, 2, 1)
            outputs = {'model_sha256': args.sha256}
        else:
            estimate = np.stack([load(args.estimate_root / folder.name / (s + '.wav'))
                                 for s in SOURCES])
            outputs = {s: digest(args.estimate_root / folder.name / (s + '.wav'))
                       for s in SOURCES}
        record = {'track': folder.name, 'frames': reference.shape[1],
                  'input_sha256': inputs, 'estimate_identity': outputs,
                  'scores': json_metrics(score(reference, estimate,
                                              compute_museval=not args.whole_song_only))}
        (args.output / f'{index:03d}.json').write_text(
            json.dumps(record, indent=2, allow_nan=False) + '\n')
        records.append(record)
        print(json.dumps({'completed': len(records), 'total': len(songs), 'track': folder.name}), flush=True)
    result = {'protocol': protocol, 'aggregate': aggregate(records), 'records': records}
    (args.output / 'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
