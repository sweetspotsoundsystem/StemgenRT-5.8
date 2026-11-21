import copy
import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from stemgenrt import losses
from stemgenrt._losses import residual_other
from stemgenrt.banded import BandSeparator
from stemgenrt._evaluation.metrics import MetricConfig, windowed_sdr
from stemgenrt.checkpoint import ParameterEMA, _validate_training_provenance, state_sha256
from stemgenrt.model import render_scored_context
from stemgenrt.specialist import SpecialistSeparator
from stemgenrt.trainer import TrainingConfig


def test_residual_uses_aligned_mixture_and_recorded_other_with_per_crop_scaling():
    torch.manual_seed(393)
    mix, ref, drums, vocals, bass = torch.randn(5, 3, 2, 384).unbind()
    mix[0].zero_()
    mix[1].mul_(.02)
    bass = bass.clone().requires_grad_()
    result = residual_other.normalized_other_error(bass, drums, vocals, mix, ref)
    residual = mix - ((drums + bass) + vocals) - ref
    weights = mix.square().mean((1, 2)).clamp_min(1e-6)
    torch.testing.assert_close(result, (residual.square().mean((1, 2)) / weights).mean())
    grad, = torch.autograd.grad(result, bass)
    torch.testing.assert_close(grad, -2 * residual / weights[:, None, None] / bass.numel())
    with pytest.raises(ValueError, match="aligned stereo"):
        residual_other.normalized_other_error(bass, drums.requires_grad_(), vocals, mix, ref)


def test_frozen_peer_render_and_ordinary_group_replay(monkeypatch):
    torch.manual_seed(31)
    bass = SpecialistSeparator(source="bass", band_width=16, global_width=16, layers=1)
    peers = {"drums": SpecialistSeparator(source="drums", band_width=16, global_width=16, layers=1,
                                          waveform_basis=16),
             "vocals": BandSeparator(sources=("vocals",), band_width=16, global_width=16, layers=1)}
    # Nonzero decoder weights make omitting a peer or misaligning its output observable.
    with torch.no_grad():
        for model in peers.values():
            for parameter in model.parameters():
                if not parameter.count_nonzero():
                    parameter.normal_(0, .01)
    hashes = {"drums": "d" * 64, "vocals": "a" * 64}
    def load(name, *, expected_sha256, role, device):
        assert expected_sha256 == hashes[name] and role == "ema" and device == "cpu"
        return peers[name]
    monkeypatch.setattr(residual_other, "load_model", load)
    supervision = residual_other.ResidualOtherLoss(.25, hashes, drums_checkpoint="drums",
                                                  vocals_checkpoint="vocals", device="cpu")
    targets = .03 * torch.randn(16, 4, 2, 128 + 44160)
    mixture = targets.sum(1) + .002
    before = {name: state_sha256(model.state_dict()) for name, model in peers.items()}
    with torch.no_grad():
        def render(model):
            return torch.cat([render_scored_context(model, m, warmup_samples=128, carry_state=True).deployed[:, 0]
                              for m in mixture.split(4)])
        deployed = render(bass)
        expected = mixture[..., 128:] - ((render(peers["drums"]) + deployed) + render(peers["vocals"]))
        mse = (expected - targets[:, 3, :, 128:]).square().mean((1, 2))
        expected_term = (mse / mixture[..., 128:].square().mean((1, 2)).clamp_min(1e-6)).mean()
        dbs = [windowed_sdr(ref.numpy(), est.numpy(), MetricConfig())["db"]
               for ref, est in zip(targets[:, 3, :, 128:], expected, strict=True)]
    initial = copy.deepcopy(bass)
    row = losses.grouped_update(bass, torch.optim.Adam(bass.parameters(), lr=3e-4, foreach=False),
                                ParameterEMA(bass), mixture, targets, step=1, warmup_samples=128,
                                ordinary_microbatch=4, target_source="bass", residual_other=supervision)
    assert row["accumulation_policy"]["version"] == "bass-residual-other-v1"
    assert row["groups"]["ordinary"]["residual_other_loss_unweighted"] == pytest.approx(float(expected_term))
    assert "residual_other_weight" not in row["groups"]["auxiliary"]
    assert row["ema_updates"] == 1
    mixed = residual_other.ResidualOtherLoss(.25, hashes, drums_checkpoint="drums", vocals_checkpoint="vocals",
                                             device="cpu", sdr_weight=.04)
    peer_forwards = []
    def count_render(model, *args, **kwargs):
        peer_forwards.append(model)
        return render_scored_context(model, *args, **kwargs)
    monkeypatch.setattr(residual_other, "render_scored_context", count_render)
    trial = losses.grouped_update(initial, torch.optim.Adam(initial.parameters(), lr=3e-4, foreach=False),
                                  ParameterEMA(initial), mixture, targets, step=1, warmup_samples=128,
                                  ordinary_microbatch=4, target_source="bass", residual_other=mixed)
    ordinary = trial["groups"]["ordinary"]
    assert trial["accumulation_policy"]["residual_other_version"] == residual_other.WINDOW_SDR_VERSION
    assert ordinary["residual_other_sdr_loss_unweighted"] == pytest.approx(-sum(dbs) / len(dbs), abs=1e-6)
    assert ordinary["residual_other_sdr_counts"] == {"eligible_crops": 16, "active_windows": 16}
    assert ordinary["weighted_loss"] == pytest.approx(row["groups"]["ordinary"]["weighted_loss"]
                                                      - .04 * sum(dbs) / len(dbs), abs=2e-6)
    assert trial["groups"]["auxiliary"] == row["groups"]["auxiliary"]
    assert state_sha256(initial.state_dict()) != state_sha256(bass.state_dict())
    for name, model in peers.items():
        assert peer_forwards.count(model) == 4
        assert state_sha256(model.state_dict()) == before[name]
        assert all(p.grad is None and not p.requires_grad for p in model.parameters())


def test_residual_recipe_binds_the_two_peers_and_bass_training():
    config = TrainingConfig(model_family="specialist", target_source="bass", precision="fp32",
                            residual_other_weight=.25, residual_peer_sha256={"drums": "a"*64, "vocals": "b"*64})
    assert config.validate() == config
    for changes in ({"residual_other_weight": float("nan")}, {"residual_other_weight": -1},
                    {"residual_peer_sha256": {"bass": "a"*64}}, {"residual_other_weight": 0},
                    {"target_source": "drums"}, {"residual_peer_sha256": None}):
        with pytest.raises(ValueError):
            replace(config, **changes).validate()


def test_windowed_other_scores_and_derivatives_weight_each_active_crop_equally():
    metrics = MetricConfig()
    w = metrics.window_samples
    ref = torch.full((3, 2, 2 * w + 128), .2)
    ref[0, :, w:2*w] = 0
    ref[2, :, :2*w] = 0
    error = torch.full_like(ref, .02)
    error[1, :, :w] = .04
    error[1, :, w:2*w] = .08
    estimate = (ref + error).requires_grad_()
    value, counts = residual_other.windowed_other_sdr_loss(estimate, ref)
    expected = [windowed_sdr(r.numpy(), e.detach().numpy(), metrics)
                for r, e in zip(ref, estimate, strict=True)]
    assert [s["active_windows"] for s in expected] == [1, 2, 0]
    assert counts == {"eligible_crops": 2, "active_windows": 3}
    assert value.item() == pytest.approx(-(expected[0]["db"] + expected[1]["db"]) / 2, abs=1e-6)
    gradient, = torch.autograd.grad(value, estimate)
    expected_gradient = torch.zeros_like(ref)
    for crop, window, denominator in ((0, 0, 2), (1, 0, 4), (1, 1, 4)):
        region = slice(window * w, (window + 1) * w)
        delta = (estimate.detach()[crop, :, region].double() - ref[crop, :, region].double())
        expected_gradient[crop, :, region] = (20 / torch.log(torch.tensor(10., dtype=torch.float64))
                                              * delta / (delta.square().sum() + metrics.epsilon) / denominator)
    torch.testing.assert_close(gradient, expected_gradient, rtol=1e-6, atol=0.)
    empty, counts = residual_other.windowed_other_sdr_loss(estimate, torch.zeros_like(ref))
    assert empty.item() == 0 and counts == {"eligible_crops": 0, "active_windows": 0}
    assert not torch.autograd.grad(empty, estimate)[0].count_nonzero()
    for trial, target in ((estimate[..., :w-1], ref[..., :w-1]), (estimate.double(), ref), (estimate, ref.requires_grad_())):
        with pytest.raises(ValueError, match="aligned stereo FP32"):
            residual_other.windowed_other_sdr_loss(trial, target)


def test_windowed_other_config_binds_the_loss_and_restores_the_legacy_recipe():
    configs = Path(__file__).resolve().parents[1] / "configs"
    control = json.loads((configs / "bass-hard-solo-repeat.json").read_text())
    mixed = json.loads((configs / "bass-residual-window-sdr.json").read_text())
    assert mixed == {**control, "residual_other_weight": control["residual_other_weight"] / 2,
                     "residual_other_sdr_weight": .043483909366881006}
    config = TrainingConfig(**mixed).validate()
    for invalid in (-.1, True, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="weights"):
            replace(config, residual_other_sdr_weight=invalid).validate()
    assert replace(config, residual_other_weight=0.).validate().uses_residual_other
    for changes in ({"residual_peer_sha256": None}, {"target_source": "drums"}):
        with pytest.raises(ValueError):
            replace(config, residual_other_weight=0., **changes).validate()
    peers = config.residual_peer_sha256
    old = residual_other.specification(.5, peers)
    assert old == {"version": "bass-residual-other-v1", "weight": .5, "peer_checkpoint_sha256": peers,
                   "role": "ema", "mixture_power_floor": 1e-6}
    new = residual_other.specification(.5, peers, sdr_weight=.04)
    common = {"model_family": "specialist", "target_source": "bass", "residual_peer_sha256": peers,
              "residual_other_weight": .5}
    current = {**common, "residual_other_sdr_weight": .04}
    _validate_training_provenance(common, {"current_stage_residual_other": old})
    _validate_training_provenance(current, {"current_stage_residual_other": new})
    for recipe, spec in ((common, new), (current, old), ({**current, "residual_other_sdr_weight": .02}, new),
                         (current, {**new, "activity_dbfs": -55.}), (current, {**new, "peer_checkpoint_sha256": {}})):
        with pytest.raises(ValueError, match="Residual Other"):
            _validate_training_provenance(recipe, {"current_stage_residual_other": spec})
