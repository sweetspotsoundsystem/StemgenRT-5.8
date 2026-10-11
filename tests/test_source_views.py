"""Counterfactual inputs preserve their source subset, prefix history and EOF."""
import hashlib

import numpy as np
import pytest
import soundfile as sf
import torch

from stemgenrt.banded import BandSeparator
from stemgenrt.evaluation import EvaluationTrack, MetricConfig
from stemgenrt.independent import IndependentStems
from stemgenrt.specialist import SpecialistSeparator
from stemgenrt.source_views import combine_sources, source_subsets, stream_source_views, score_source_views


@pytest.mark.parametrize("source", ["bass", "drums", "vocals"])
def test_source_view_streaming_matches_continuous_prefixes_and_group_independent_hashes(tmp_path, source):
    model = IndependentStems(
        drums=SpecialistSeparator(source="drums", band_width=16, global_width=16, layers=1, waveform_basis=16).eval(),
        bass=SpecialistSeparator(source="bass", band_width=16, global_width=16, layers=1).eval(),
        vocals=BandSeparator(band_width=16, global_width=16, layers=1).eval())
    sources = np.random.default_rng(61).normal(0, .03, (4, 2, 6397)).astype(np.float32)
    paths = tuple(tmp_path / (name + ".wav") for name in ("drums", "bass", "vocals", "other"))
    for path, values in zip(paths, sources):
        sf.write(path, values.T, 44100, subtype="FLOAT")
    mixture_path = tmp_path / "mixture.wav"
    sf.write(mixture_path, sources.sum(0).T, 44100, subtype="FLOAT")
    track = EvaluationTrack("synthetic", mixture_path, paths, ((257, 1987), (4303, 6397)))
    refs, outputs, mixtures, metadata = stream_source_views(model, track, target_source=source, unroll_hops=3)
    _, other_outputs, _, other_metadata = stream_source_views(model, track, target_source=source, unroll_hops=17)
    assert metadata["input_stream_sha256"] == other_metadata["input_stream_sha256"]
    assert metadata["source_subset_fixed_from_track_origin"] and metadata["coverage_complete"]
    for view, included in source_subsets(source).items():
        mixed = combine_sources(sources, included)
        padded = np.pad(mixed, ((0, 0), (0, metadata["received_samples"] - mixed.shape[-1])))
        assert hashlib.sha256(np.ascontiguousarray(padded.T).tobytes()).hexdigest() == metadata["input_stream_sha256"][view]
        with torch.inference_mode():
            expected = model.render(torch.from_numpy(padded)[None]).deployed[0].numpy()
        for i, (start, stop) in enumerate(track.intervals):
            np.testing.assert_array_equal(refs[i], sources[..., start:stop])
            np.testing.assert_array_equal(mixtures[view][i], mixed[..., start:stop])
            np.testing.assert_allclose(outputs[view][i], expected[..., start + 128:stop + 128], rtol=2e-5, atol=2e-7)
            np.testing.assert_allclose(outputs[view][i], other_outputs[view][i], rtol=2e-5, atol=2e-7)


def test_silent_target_output_is_counted_in_active_input_leakage():
    references = [np.random.default_rng(9).normal(0, .03, (4, 2, 44100)).astype(np.float32)]
    intervals = [{"reference_start": 0, "reference_end": 44100, "estimate_start": 128, "estimate_end": 44228}]
    mixtures, outputs = {}, {}
    for view, included in source_subsets("bass").items():
        mixture = combine_sources(references[0], included)
        estimates = np.zeros_like(references[0])
        estimates[3] = mixture
        mixtures[view], outputs[view] = [mixture], [estimates]
    scores = score_source_views("synthetic", intervals, references, outputs, mixtures, MetricConfig(), target_source="bass")
    leakage = scores["target_absent"]["native_output_levels"]["bass"]
    assert leakage["input_active_windows"] == 1
    assert leakage["output_to_input_db"] == -60.
    assert leakage["output_rms_dbfs"] == -120.
    assert scores["target_only"]["native_output_levels"]["bass"]["signed_desired_projection_gain"] == 0.


def test_source_views_reject_unknown_target():
    with pytest.raises(ValueError, match="named source"):
        source_subsets("guitar")
