from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from stemgenrt import losses
from stemgenrt._losses import solo_sampling as sampling
from stemgenrt._losses.release_margin import specification as margin_specification
from stemgenrt.checkpoint import _validate_training_provenance
from stemgenrt.specialist import SpecialistSeparator
from stemgenrt.trainer import TrainingConfig


WINDOW = sampling.METRICS.window_samples


def test_screen_chooses_lowest_active_window_mean_and_ignores_trailing_audio():
    reference = torch.full((16, 2, 2*WINDOW+10), .25)
    estimate = reference.clone()
    estimate[0].zero_()
    reference[0, :, WINDOW:2*WINDOW] = 0.  # Only one active second.
    estimate[4] *= .99
    estimate[8, :, :WINDOW] *= .98
    estimate[8, :, WINDOW:2*WINDOW] *= .9999
    estimate[1, :, 2*WINDOW:] = 4.  # Incomplete final window.
    result = sampling.choose_solo(reference, estimate)
    assert result["selected_index"] == 4 and result["active_windows"][0] == 1
    assert result["release_solo_sdr_db"][0] is None
    assert result["release_solo_sdr_db"][4] == pytest.approx(40., abs=.001)
    assert result["release_solo_sdr_db"][8] == pytest.approx((33.9794+60)/2, abs=.001)
    assert result["qualifying_crops"] == 2
    estimate[15] = estimate[4]
    assert sampling.choose_solo(reference, estimate)["selected_index"] == 15
    estimate[15] = reference[15]
    estimate[3] = estimate[4]
    assert sampling.choose_solo(reference, estimate)["selected_index"] == 3
    assert sampling.choose_solo(reference, reference)["selected_index"] == 15
    with pytest.raises(ValueError):
        sampling.choose_solo(reference.requires_grad_(), estimate)


def test_alternating_screen_uses_fresh_full_context_per_microbatch(monkeypatch):
    reference = torch.ones(16, 2, 128 + 2*WINDOW)
    reference[5] *= .5
    model = SpecialistSeparator(source="bass", band_width=16, global_width=16, layers=1).eval().requires_grad_(False)
    calls = []
    def render(released, audio, *, warmup_samples, carry_state):
        assert released is model and carry_state and warmup_samples == 128 and not torch.is_grad_enabled()
        assert audio.shape[-1] == 128 + 2*WINDOW
        calls.append(len(audio))
        wave = audio[..., warmup_samples:].clone()
        wave[wave == .5] *= .98
        return SimpleNamespace(deployed=wave[:, None])
    monkeypatch.setattr(sampling, "render_scored_context", render)
    plain = sampling.select_for_update(model, reference, step=1, warmup_samples=128, microbatch=4)
    assert plain == {"selected_index":15, "screened":False} and not calls
    screened = sampling.select_for_update(model, reference, step=2, warmup_samples=128, microbatch=4)
    assert calls == [4]*4 and screened["selected_index"] == 5
    assert not any(p.grad is not None for p in model.parameters())


def test_chosen_solo_only_replaces_the_second_view():
    stems = torch.randn(16, 4, 2, 384)
    mixture = stems.sum(1)
    ordinary = losses.source_views(mixture, stems, target_source="bass")
    assert all(torch.equal(a,b) for a,b in zip(ordinary, losses.source_views(mixture, stems, target_source="bass",solo_index=15), strict=True))
    controlled, targets = losses.source_views(mixture, stems, target_source="bass", solo_index=14)
    assert torch.equal(controlled[0], ordinary[0][0]) and torch.equal(targets[0], ordinary[1][0])
    assert torch.equal(controlled[1], stems[14,1]) and torch.equal(targets[1,1], stems[14,1])
    assert torch.count_nonzero(targets[1,[0,2,3]]) == 0
    assert losses.policy(target_source="bass", solo_index=14)["view_indices"] == [14,14]
    for source,index in (("vocals",1),("bass",16),("bass",True)):
        with pytest.raises(ValueError):
            losses.source_views(mixture,stems,target_source=source,solo_index=index)


def test_hard_solo_training_config_and_recovery_match_the_reference():
    digest="c"*64
    config=TrainingConfig(model_family="specialist",target_source="bass",precision="fp32",release_margin_weight=.05,
                          bass_release_sha256=digest,bass_solo_sampling=sampling.VERSION)
    assert config.validate() == config
    for wrong in (replace(config,bass_solo_sampling="unknown"),replace(config,release_margin_weight=0.,bass_release_sha256=None)):
        with pytest.raises(ValueError):
            wrong.validate()
    compact={"model_family":"specialist","target_source":"bass","release_margin_weight":.05,
             "bass_release_sha256":digest,"bass_solo_sampling":sampling.VERSION}
    provenance={"current_stage_bass_release_margin":margin_specification(.05,digest),
                "current_stage_bass_solo_sampling":{**sampling.specification(),"reference_sha256":digest}}
    _validate_training_provenance(compact,provenance)
    with pytest.raises(ValueError,match="solo sampling"):
        _validate_training_provenance({k:v for k,v in compact.items() if k!="bass_solo_sampling"},provenance)
    with pytest.raises(ValueError,match="solo sampling"):
        _validate_training_provenance(compact,{**provenance,"current_stage_bass_solo_sampling":sampling.specification()})
