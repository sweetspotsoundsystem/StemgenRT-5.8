#!/usr/bin/env python3
"""Prepare lawfully obtained MoisesDB audio using its official decoder."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stemgenrt.moises import prepare_track


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-path', type=Path, required=True)
    parser.add_argument('--selection', type=Path, required=True,
                        help='JSON list of {provider, id}; explicit training selection only')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a new output directory')
    selected = json.loads(args.selection.read_text())
    keys = [(row['provider'], row['id']) for row in selected]
    if not keys or len(keys) != len(set(keys)):
        parser.error('Require a nonempty selection with unique provider/id pairs')
    for key in keys:
        for value in key:
            if not isinstance(value, str) or not value or value in ('.', '..') or '/' in value or '\\' in value:
                parser.error('Provider and id must be single path components')
    from moisesdb.track import MoisesDBTrack
    versions = {name: importlib.metadata.version(name) for name in
                ('moisesdb', 'numpy', 'librosa', 'soundfile')}
    args.output.mkdir(parents=True)
    report = {'schema': 'stemgenrt-moises-reference-v1', 'versions': versions,
              'historical_byte_equivalence': False, 'selection': selected, 'tracks': []}
    for provider, track_id in sorted(keys):
        track = MoisesDBTrack(provider, track_id, str(args.data_path), sample_rate=44100)
        report['tracks'].append(prepare_track(track, args.output / provider / track_id))
        print(provider + '/' + track_id, flush=True)
    (args.output / 'complete.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
