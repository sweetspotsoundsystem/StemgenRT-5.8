"""Synthetic orchestration and evaluation integration; no dataset training."""
import json
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf
import torch

from stemgenrt import trainer, checkpoint
from stemgenrt.banded import BandSeparator, SOURCE_ORDER
from stemgenrt.compact import CompactSeparator
from stemgenrt.evaluation import NativeRenderer, EvaluationTrack, stream_track
from stemgenrt.replacement import VocalReplacement


@pytest.mark.parametrize("sources", [("vocals",), SOURCE_ORDER])
def test_banded_replacement_streams_partial_final_hop_without_changing_parent(tmp_path, sources):
    parent = CompactSeparator(sources=SOURCE_ORDER, hidden_size=16, layers=1).eval()
    candidate = BandSeparator(sources=sources, band_width=16, global_width=16, layers=1).eval()
    before = checkpoint.state_sha256(parent.state_dict())
    system = VocalReplacement(parent, candidate)
    audio = np.random.default_rng(71).normal(0, .1, (2, 1797)).astype(np.float32)
    path = tmp_path / "synthetic.wav"
    sf.write(path, audio.T, 44100, subtype="FLOAT")
    intervals = ((17, 280), (998, 1523), (1778, 1797))
    track = EvaluationTrack("synthetic", path, (path,) * 4, intervals)
    found, _ = stream_track(NativeRenderer(system), track, frames=audio.shape[-1], unroll_hops=3)
    with torch.inference_mode():
        padded = torch.from_numpy(np.pad(audio, ((0, 0), (0, (-1797) % 128 + 128))))[None]
        full = system.render(padded)
        base = parent.render(padded)
        proposal = candidate.render(padded)
    torch.testing.assert_close(full.deployed[:, :2], base.deployed[:, :2], rtol=0, atol=0)
    torch.testing.assert_close(full.deployed[:, 2], proposal.deployed[:, sources.index("vocals")], rtol=0, atol=0)
    for value, (start, stop) in zip(found, intervals):
        expected = full.deployed[0, ..., 128 + start:128 + stop].numpy()
        np.testing.assert_allclose(value, expected, atol=1e-7, rtol=1e-5)
        np.testing.assert_allclose(value.sum(0), audio[:, start:stop], atol=1e-7, rtol=0)
    assert checkpoint.state_sha256(parent.state_dict()) == before


@pytest.mark.parametrize("target_source", ["vocals", None])
def test_real_trainer_selects_bands_and_retains_identity_on_resume(monkeypatch, tmp_path, target_source):
    """Use real factory, optimizer, EMA and serialization with a short test loss.

    Data synthesis and the short MSE update replace expensive data/SDR work;
    the complete source-view objective is exercised in test_banded_training.
    """
    corpus = SimpleNamespace(sha256="synthetic-fixed", tracks=("synthetic",), root_weights={"synthetic": 1.})
    monkeypatch.setattr(trainer, "load_manifest", lambda *a, **kw: corpus)
    class Dataset(torch.utils.data.Dataset):
        def __len__(self):
            return 48
        def __getitem__(self, index):
            generator = torch.Generator().manual_seed(index + 128)
            sources = torch.randn(4, 2, 512, generator=generator) * .01
            return sources.sum(0), sources
    monkeypatch.setattr(trainer, "make_dataset", lambda *a: Dataset())
    def update(model, optimizer, ema, mixture, targets, *, step, **kwargs):
        assert type(model) is BandSeparator
        assert kwargs["target_source"] == target_source
        truth = targets[:, 2:3] if target_source else targets
        optimizer.zero_grad(set_to_none=True)
        output = model.render(mixture)
        loss = (output.deployed[..., 128:] - truth[..., :-128]).square().mean()
        loss.backward()
        optimizer.step()
        ema.update(model, step=step)
        return {"step": step, "weighted_loss": float(loss.detach())}
    monkeypatch.setattr(trainer, "grouped_update", update)
    config = trainer.TrainingConfig(model_family="banded", target_source=target_source,
        band_width=16, band_global_width=16, band_layers=1, steps=3, warmup=0,
        checkpoint_every=1, workers=0, device="cpu", precision="fp32")
    first = trainer.train(config, "synthetic.json", tmp_path / "first", stop_after=1)
    resumed = trainer.train(config, "synthetic.json", tmp_path / "resumed", stop_after=2,
        resume=first["checkpoint"]["path"], sha256=first["checkpoint"]["sha256"])
    saved_config = json.loads((tmp_path / "resumed/config.json").read_text())
    assert not any(k.startswith("compact_") for k in saved_config)
    assert saved_config["model_family"] == "banded" and saved_config["band_layers"] == 1
    assert resumed["resumed_from_step"] == 1 and resumed["next_sample_index"] == 32
    loaded = checkpoint.load_model(resumed["checkpoint"]["path"], expected_sha256=resumed["checkpoint"]["sha256"])
    assert loaded.provenance["training_updates"] == loaded.provenance["current_stage_updates"] == 2
    assert loaded.provenance["parent_training_updates"] == 0
    assert loaded.sources == (("vocals",) if target_source else SOURCE_ORDER)
