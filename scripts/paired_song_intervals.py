#!/usr/bin/env python3
"""Paired percentile intervals for two frozen full-song summary endpoints."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def compare(previous, current, *, seed=20261025, draws=10000):
    sources = ('drums', 'bass', 'vocals', 'other')
    def index(rows):
        result = {row['track']: row for row in rows}
        if len(result) != len(rows):
            raise ValueError('Duplicate song')
        return result
    old, new = index(previous), index(current)
    if not old or old.keys() != new.keys():
        raise ValueError('Require identical nonempty song inventories')
    songs = sorted(old)
    delta = np.array([[new[s]['global_sdr_db'][k] - old[s]['global_sdr_db'][k]
                       for k in sources] for s in songs])
    if not np.isfinite(delta).all():
        raise ValueError('Whole-song scores must be finite')
    delta = np.column_stack([delta, delta.mean(axis=1)])
    sampled = np.random.default_rng(seed).integers(len(songs), size=(draws, len(songs)))
    lower, upper = np.quantile(delta[sampled].mean(axis=1), [.025, .975], axis=0)
    entries = {key: {'mean_change_db': float(delta[:, i].mean()),
                     'median_change_db': float(np.median(delta[:, i])),
                     'ci95_db': [float(lower[i]), float(upper[i])],
                     'improved_songs': int((delta[:, i] > 0).sum())}
               for i, key in enumerate((*sources, 'overall'))}
    return {'method': 'paired percentile bootstrap over songs', 'draws': draws,
            'seed': seed, 'song_count': len(songs), 'sources': entries,
            'simultaneous_intervals': False, 'training_seed_uncertainty': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--previous', type=Path, required=True)
    parser.add_argument('--current', type=Path, required=True)
    parser.add_argument('--previous-artifact', default='int8')
    parser.add_argument('--current-artifact', default='independent_fp32')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    old = json.loads(args.previous.read_text())['artifacts'][args.previous_artifact]['songs']
    new = json.loads(args.current.read_text())['artifacts'][args.current_artifact]['songs']
    result = compare(old, new)
    result['input_sha256'] = {label: hashlib.sha256(path.read_bytes()).hexdigest()
                              for label, path in [('previous', args.previous), ('current', args.current)]}
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False); stream.write('\n')


if __name__ == '__main__':
    main()
