"""Paired development scores for fixed, alternating and reduced-frequency Bass-only sampling."""
import math
from statistics import mean, median

import numpy as np


def _metrics(row):
    stems = row["ordinary"]["per_stem"]
    bass = stems["bass"]
    views = row["controlled"]["scores"]
    return {"bass_sdr":bass["full_sdr_db"], "other_sdr":stems["other"]["full_sdr_db"], "bass_sir":bass["sir_db"],
            "bass_low_sdr":bass["band_sdr_db"]["low_20_250"], "ordinary_absent_dbfs":bass["absent_fp_dbfs"],
            "target_absent_to_input_db":views["target_absent"]["bass_output_levels"]["output_to_input_db"],
            "target_only_sdr":views["target_only"]["bass_scores"]["full_sdr_db"]}


def _index(rows):
    result = {}
    for row in rows:
        name = row["ordinary"]["name"]
        if name in result:
            raise ValueError("Duplicate song in the sampling comparison")
        result[name] = row
    return result


def _summary(deltas, seed, draws):
    if not deltas:
        return {"mean_db":None, "median_db":None, "positive_songs":0, "eligible_songs":0,
                "paired_song_ci95_db":None, "by_song":{}}
    values = np.asarray(list(deltas.values()), dtype=np.float64)
    indexes = np.random.default_rng(seed).integers(len(values), size=(draws, len(values)))
    return {"mean_db":mean(deltas.values()), "median_db":median(deltas.values()),
            "positive_songs":int(np.count_nonzero(values>0)), "eligible_songs":len(values),
            "paired_song_ci95_db":np.quantile(values[indexes].mean(1), [.025,.975]).tolist(), "by_song":deltas}


def compare_sampling_runs(fixed, alternating, candidate, *, seed=20261121, draws=20000):
    """Compare matched per-song ordinary and controlled scores from three frozen Bass runs.

    Each iterable uses the per-song format returned by score_bass_interpolation_track.
    Keep the candidate's song order for bootstrap draws. A missing measurement is valid
    when all three runs have the same missing value. The primary joint score weights Bass
    and Other equally within each song. The solo gain fraction is defined when alternating
    sampling improves mean isolated Bass SDR relative to fixed sampling.
    """
    if type(draws) is not int or draws <= 0:
        raise ValueError("Supply a positive number of bootstrap draws")
    fixed, alternating, candidate = (_index(x) for x in (fixed,alternating,candidate))
    if not candidate or not fixed.keys() == alternating.keys() == candidate.keys():
        raise ValueError("Sampling runs require identical nonempty song inventories")
    comparisons = {"candidate_minus_fixed":{}, "candidate_minus_matched_control":{}}
    solo_control = []
    for name,record in candidate.items():
        controls = (fixed[name],alternating[name])
        if any(r["ordinary"]["excerpts"] != record["ordinary"]["excerpts"] for r in controls):
            raise ValueError("Sampling runs require identical excerpts for each song")
        fv,av,cv = (_metrics(r) for r in (*controls,record))
        for key,c in cv.items():
            a,b = fv[key],av[key]
            if (a is None) != (c is None) or (b is None) != (c is None):
                raise ValueError("Metric eligibility differs between sampling runs")
            for diffs in comparisons.values():
                diffs.setdefault(key,{})
            if c is None:
                continue
            if not all(math.isfinite(v) for v in (a,b,c)):
                raise ValueError("Sampling scores must be finite")
            comparisons["candidate_minus_fixed"][key][name] = c-a
            comparisons["candidate_minus_matched_control"][key][name] = c-b
            if key == "target_only_sdr":
                solo_control.append(b-a)
    for diffs in comparisons.values():
        bass,other = diffs["bass_sdr"],diffs["other_sdr"]
        diffs["joint_sdr"] = {s:(bass[s]+other[s])/2 for s in candidate if s in bass and s in other}
    summaries = {label:{k:_summary(v,seed,draws) for k,v in diffs.items()} for label,diffs in comparisons.items()}
    parent_gain = mean(solo_control) if solo_control else None
    candidate_gain = summaries["candidate_minus_fixed"]["target_only_sdr"]["mean_db"]
    return {**summaries, "primary_joint_db":summaries["candidate_minus_matched_control"]["joint_sdr"],
            "solo_preservation":{"alternating_minus_fixed_db":parent_gain, "quarter_rate_minus_fixed_db":candidate_gain,
                                 "retained_fraction":candidate_gain/parent_gain if parent_gain is not None and parent_gain>0 else None}}
