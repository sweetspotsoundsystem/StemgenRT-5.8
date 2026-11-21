from copy import deepcopy

import numpy as np
import pytest

from stemgenrt.bass_sampling_eval import compare_sampling_runs


def song(name, bass, other, solo, quiet=None):
    return {"ordinary":{"name":name,"excerpts":[{"reference_start":0,"reference_end":44100}],
                        "per_stem":{"bass":{"full_sdr_db":bass,"band_sdr_db":{"low_20_250":bass-.1},
                                            "sir_db":bass+.5,"absent_fp_dbfs":quiet},"other":{"full_sdr_db":other}}},
            "controlled":{"scores":{"target_only":{"bass_scores":{"full_sdr_db":solo}},
                                    "target_absent":{"bass_output_levels":{"output_to_input_db":-22.}}}}}


def test_paired_sampling_scores_align_songs_and_retain_solo_gain():
    fixed = [song("a",0.,2.,10.),song("b",3.,2.,0.,-50.)]
    hard = [song("b",1.,2.,2.,-49.),song("a",2.,2.,11.)]
    trial = [song("a",4.,0.,10.5),song("b",4.,3.,1.,-48.)]
    result = compare_sampling_runs(fixed,hard,trial,seed=7,draws=500)
    expected = np.array([0.,2.])
    joint = result["primary_joint_db"]
    assert joint["by_song"] == {"a":0.,"b":2.} and joint["mean_db"] == joint["median_db"] == 1.
    assert joint["positive_songs"] == 1 and joint["eligible_songs"] == 2
    means = np.random.default_rng(7).choice(expected,size=(500,2)).mean(1)
    np.testing.assert_array_equal(joint["paired_song_ci95_db"],np.quantile(means,[.025,.975]))
    assert result["candidate_minus_fixed"]["ordinary_absent_dbfs"]["by_song"] == {"b":2.}
    assert result["solo_preservation"] == {"alternating_minus_fixed_db":1.5,"quarter_rate_minus_fixed_db":.75,
                                          "retained_fraction":.5}
    identical = compare_sampling_runs(fixed,fixed,fixed,draws=5)
    assert identical["solo_preservation"]["retained_fraction"] is None


def test_paired_sampling_scores_check_coverage_and_eligibility():
    base = [song("a",0.,0.,2.)]
    out = compare_sampling_runs(base,base,base,draws=3)
    assert out["candidate_minus_fixed"]["ordinary_absent_dbfs"]["paired_song_ci95_db"] is None
    with pytest.raises(ValueError,match="inventories"):
        compare_sampling_runs(base,[],base)
    with pytest.raises(ValueError,match="Duplicate"):
        compare_sampling_runs(base,base,base*2)
    trial = deepcopy(base)
    trial[0]["ordinary"]["excerpts"] = []
    with pytest.raises(ValueError,match="excerpts"):
        compare_sampling_runs(base,base,trial)
    trial = [song("a",0.,0.,2.,-40.)]
    with pytest.raises(ValueError,match="eligibility"):
        compare_sampling_runs(base,base,trial)
    with pytest.raises(ValueError,match="finite"):
        compare_sampling_runs(base,base,[song("a",float("nan"),0.,2.)])
    with pytest.raises(ValueError,match="draws"):
        compare_sampling_runs(base,base,base,draws=False)
