from dataclasses import replace

import pytest
import torch

from stemgenrt import losses
from stemgenrt._losses import residual_other
from stemgenrt.banded import BandSeparator
from stemgenrt.checkpoint import ParameterEMA, state_sha256
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
    row = losses.grouped_update(bass, torch.optim.Adam(bass.parameters(), lr=3e-4, foreach=False),
                                ParameterEMA(bass), mixture, targets, step=1, warmup_samples=128,
                                ordinary_microbatch=4, target_source="bass", residual_other=supervision)
    assert row["accumulation_policy"]["version"] == "bass-residual-other-v1"
    assert row["groups"]["ordinary"]["residual_other_loss_unweighted"] == pytest.approx(float(expected_term))
    assert "residual_other_weight" not in row["groups"]["auxiliary"]
    assert row["ema_updates"] == 1
    for name, model in peers.items():
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
