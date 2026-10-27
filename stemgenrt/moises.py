"""Deterministic four-stem preparation from MoisesDB's published taxonomy.

This is a new reference pipeline, not a recovered historical conversion script.
It deliberately uses floating WAV output to avoid undocumented clipping.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import soundfile as sf

SOURCES = ('drums', 'bass', 'vocals', 'other')
TAXONOMY = frozenset((*SOURCES, 'guitar', 'other_plucked', 'percussion', 'piano',
                      'other_keys', 'bowed_strings', 'wind'))


def four_stems(stems):
    """Map taxonomy stems, zero-padding tails and duplicating mono to stereo."""
    if not stems or set(stems) - TAXONOMY:
        raise ValueError('Empty or unknown MoisesDB taxonomy')
    arrays = {}
    for name, data in stems.items():
        audio = np.asarray(data, dtype=np.float64)
        if (audio.ndim != 2 or audio.shape[0] not in (1, 2) or not audio.shape[1]
                or not np.isfinite(audio).all()):
            raise ValueError('Require finite mono/stereo [channels, samples] audio')
        arrays[name] = np.repeat(audio, 2, axis=0) if audio.shape[0] == 1 else audio
    length = max(value.shape[1] for value in arrays.values())
    result = {source: np.zeros((2, length), dtype=np.float64) for source in SOURCES}
    for name in sorted(arrays):
        target = name if name in SOURCES else 'other'
        result[target][:, :arrays[name].shape[1]] += arrays[name]
    return result


def prepare_track(track, output):
    """Consume a MoisesDBTrack at 44.1 kHz; never overwrite an existing song."""
    if track.sr != 44100:
        raise ValueError('Construct MoisesDBTrack with sample_rate=44100')
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    if {row['stemName'] for row in track.json_data['stems']} - TAXONOMY:
        raise ValueError('Unknown raw MoisesDB taxonomy')
    stems = four_stems(track.stems)
    stems['mixture'] = sum(stems.values())
    output.mkdir(parents=True)
    files = {}
    for name, audio in stems.items():
        path = output / (name + '.wav')
        sf.write(path, audio.T, 44100, subtype='FLOAT')
        with path.open('rb') as stream:
            files[path.name] = hashlib.file_digest(stream, 'sha256').hexdigest()
    input_hashes = {}
    paths = [Path(track.path) / 'data.json']
    paths.extend(Path(path) for group in track.sources.values()
                 for tracks in group.values() for path in tracks)
    for path in sorted(set(paths)):
        with path.open('rb') as stream:
            input_hashes[path.relative_to(track.path).as_posix()] = hashlib.file_digest(stream, 'sha256').hexdigest()
    receipt = {'provider': track.provider, 'id': track.id, 'artist': track.artist,
               'song': track.name, 'sample_rate': 44100, 'channels': 2,
               'frames': stems['mixture'].shape[1], 'subtype': 'FLOAT',
               'gain_normalization': False, 'tail_policy': 'zero pad',
               'historical_byte_equivalence': False, 'input_sha256': input_hashes, 'sha256': files}
    (output / 'preparation.json').write_text(json.dumps(receipt, indent=2) + '\n')
    return receipt
