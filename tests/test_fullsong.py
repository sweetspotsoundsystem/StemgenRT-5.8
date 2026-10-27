import numpy as np
import pytest

from stemgenrt.fullsong import SOURCES, aggregate, json_metrics, score
from stemgenrt.moises import four_stems


def test_whole_song_includes_silent_samples_and_does_not_fit_gain():
    ref = np.ones((4, 12, 2))
    estimate = ref * 0.5
    ref[:, 6:] = 0
    values = score(ref, estimate, compute_museval=False)
    # 12 units of reference energy and 6 of error, including the silent half.
    expected = 10 * np.log10((12 + 1e-7) / (6 + 1e-7))
    assert list(values['global_sdr_db'].values()) == pytest.approx([expected] * 4)
    for broken in (estimate[:, :-1], np.full_like(estimate, np.nan)):
        with pytest.raises(ValueError):
            score(ref, broken, compute_museval=False)


def test_aggregate_weights_songs_equally_and_retains_undefined_frames():
    rows = []
    for name, whole, frames in [('short', 0, [0., 10., None]), ('long', 20, [20.] * 100)]:
        rows.append({'track': name, 'scores': {
            'global_sdr_db': dict.fromkeys(SOURCES, whole),
            'museval': {'SDR': dict.fromkeys(SOURCES, frames)}}})
    result = aggregate(rows)
    assert result['overall'] == {'global_mean_db': 10., 'museval_median_db': 12.5}
    assert result['sources']['vocals']['museval_defined_frames'] == 102
    assert result['sources']['vocals']['museval_total_frames'] == 103
    assert json_metrics([float('nan'), float('inf'), -float('inf')]) == [None, 'Infinity', '-Infinity']
    with pytest.raises(ValueError, match='Duplicate'):
        aggregate(rows + rows)


def test_moises_percussion_maps_to_other_and_missing_stems_are_silent():
    mapped = four_stems({'vocals': np.ones((1, 3)), 'percussion': np.ones((2, 5)) * .25,
                         'guitar': np.ones((2, 4)) * .5})
    np.testing.assert_array_equal(mapped['drums'], 0)
    np.testing.assert_array_equal(mapped['bass'], 0)
    np.testing.assert_array_equal(mapped['vocals'], [[1, 1, 1, 0, 0]] * 2)
    np.testing.assert_array_equal(mapped['other'], [[.75, .75, .75, .75, .25]] * 2)
    with pytest.raises(ValueError, match='unknown'):
        four_stems({'misspelled': np.ones((2, 3))})
