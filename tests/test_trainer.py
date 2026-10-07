"""Training orchestration without running the large audio model on CPU."""
from __future__ import annotations

from dataclasses import asdict, replace
import copy
import hashlib
import json
import random
import signal
import threading
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from stemgenrt import trainer


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(.25))
        self.register_buffer("fixed", torch.tensor(1.))


class TinyEMA:
    def __init__(self, model, *, decay):
        self.updates = 0
        self.decay = decay

    def update(self, model, *, step):
        assert step == self.updates + 1
        self.updates = step


def endpoint(step=0):
    model = TinyModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-5, foreach=False)
    ema = TinyEMA(model, decay=.995)
    for index in range(step):
        optimizer.zero_grad(set_to_none=True)
        model.weight.square().backward()
        optimizer.step()
        ema.update(model, step=index + 1)
    return model, optimizer, ema


@pytest.fixture
def harness(monkeypatch):
    config = trainer.TrainingConfig(steps=6, warmup=2, workers=0,
                                    device="cpu", precision="fp32", data_start=320,
                                    past_filter=False, track_sampling="uniform",
                                    extra_ordinary_primary_sdr_weight=0.)
    corpus = SimpleNamespace(sha256="training-bytes", tracks=("training-track",),
                             root_weights={"recordings": 1.}, split="train")
    calls = SimpleNamespace(indices=[], updates=[], saves=[], datasets=[], loads=[])

    def manifest(path, **kwargs):
        calls.loads.append((path, kwargs))
        return corpus

    class AddressedDataset(torch.utils.data.Dataset):
        def __len__(self):
            return calls.datasets[-1][2]

        def __getitem__(self, index):
            calls.indices.append(index)
            return torch.full((2, 8), float(index)), torch.full((4, 2, 8), float(index))

    def dataset(tracks, effective_config, final_index):
        calls.datasets.append((tracks, effective_config, final_index))
        return AddressedDataset()

    def update(model, optimizer, ema, mixture, targets, *, step, **kwargs):
        assert step == ema.updates + 1
        assert all(state["step"].item() == step - 1 for state in optimizer.state.values())
        calls.updates.append((step, optimizer.param_groups[0]["lr"], model, optimizer, ema))
        optimizer.zero_grad(set_to_none=True)
        model.weight.square().backward()
        optimizer.step()
        ema.update(model, step=step)
        return {"step": step, "weighted_loss": float(model.weight.detach())}

    def save(path, model, optimizer, ema, **kwargs):
        assert kwargs["step"] == ema.updates
        assert all(state["step"].item() == ema.updates for state in optimizer.state.values())
        calls.saves.append(kwargs)
        return {"path": str(path), "sha256": "saved-bytes", "step": kwargs["step"]}

    monkeypatch.setattr(trainer, "load_manifest", manifest)
    monkeypatch.setattr(trainer, "configure_determinism", lambda config: torch.device("cpu"))
    monkeypatch.setattr(trainer, "StemgenRT58", TinyModel)
    monkeypatch.setattr(trainer, "ParameterEMA", TinyEMA)
    monkeypatch.setattr(trainer, "make_dataset", dataset)
    monkeypatch.setattr(trainer, "remix_batch", lambda mixture, targets, **kwargs:
                        (mixture, targets, None, None))
    monkeypatch.setattr(trainer, "batch_recipes", lambda config, first: [{"sample_index": first}])
    monkeypatch.setattr(trainer, "grouped_update", update)
    monkeypatch.setattr(trainer, "save_training_checkpoint", save)
    return config, corpus, calls, update


def test_learning_rate_uses_original_completed_update_schedule():
    config = trainer.TrainingConfig()
    assert trainer.learning_rate(0, config) == config.lr / config.warmup
    assert trainer.learning_rate(config.warmup - 1, config) == config.lr
    assert trainer.learning_rate(config.warmup, config) == pytest.approx(config.lr)
    assert trainer.learning_rate(config.steps - 1, config) == config.min_lr
    assert trainer.learning_rate(config.warmup + 1, config) < config.lr
    with pytest.raises(ValueError):
        trainer.learning_rate(config.steps, config)


def test_current_config_selects_frozen_teacher_baseline():
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    selected = json.loads((root / 'configs/current-training.json').read_text())
    config = trainer.TrainingConfig(**selected).validate()
    assert config.steps == 2000 and config.data_start == 4132000
    assert config.seed == 20261102 and config.data_seed == 60
    assert config.precision == 'bf16' and config.microbatch_size == 16 and config.auxiliary_microbatch_size == 2
    assert config.attention_window == 32 and not config.past_filter
    assert config.track_sampling == 'uniform' and config.extra_ordinary_primary_sdr_weight == .2
    assert config.teacher_coefficient == 1.0 and config.teacher_checkpoint == "models/7d865c68-3d5dd56b.th"
    assert config.checkpoint_every == 50 and config.ema_decay == .995
    defaults = trainer.TrainingConfig().validate()
    assert defaults.precision == config.precision and defaults.past_filter == config.past_filter
    assert defaults.microbatch_size == 16 and defaults.auxiliary_microbatch_size == 2
    assert defaults.track_sampling == config.track_sampling
    assert defaults.extra_ordinary_primary_sdr_weight == config.extra_ordinary_primary_sdr_weight


@pytest.mark.parametrize("changes", [
    {"target_source": "lead"}, {"target_source": 2}, {"target_source": False},
    {"batch_size": 8}, {"microbatch_size": 17}, {"auxiliary_microbatch_size": 3},
    {"crop_samples": 132224}, {"data_start": 1}, {"warmup": 2000}, {"min_lr": 1e-3},
    {"device": "cpu", "precision": "bf16"}, {"root_weights": {"recordings": float("nan")}},
    {"extra_ordinary_primary_sdr_weight": -.2}, {"extra_ordinary_primary_sdr_weight": float("nan")},
    {"extra_ordinary_primary_sdr_weight": .4}, {"extra_ordinary_primary_sdr_weight": True},
    {"track_sampling": "unknown"}, {"track_sampling": None},
    {"attention_window": 64}, {"attention_window": True},
    {"attention_window": 128}, {"past_filter": True}, {"track_sampling": "duration"},
])
def test_config_rejects_incompatible_scientific_settings(changes):
    with pytest.raises(ValueError):
        replace(trainer.TrainingConfig(), **changes).validate()


def test_fresh_stop_keeps_original_horizon_and_one_update_per_address(harness, tmp_path):
    config, _, calls, _ = harness
    result = trainer.train(config, "train.json", tmp_path / "run", stop_after=2)
    assert result["step"] == 2 and result["schedule_steps"] == 6
    assert calls.loads == [("train.json", {"expected_split": "train"})]
    assert calls.indices == list(range(320, 352))
    assert [row[:2] for row in calls.updates] == [
        (1, trainer.learning_rate(0, config)), (2, trainer.learning_rate(1, config))]
    assert len(calls.saves) == 1
    assert calls.saves[0]["step"] == 2 and calls.saves[0]["next_sample_index"] == 352
    assert calls.saves[0]["config"]["steps"] == 6
    assert "target_source" not in calls.saves[0]["config"]
    assert calls.saves[0]["data_identity"] == {
        "manifest_sha256": "training-bytes", "sampling_root_order": ["recordings"]}
    rows = [json.loads(line) for line in (tmp_path / "run/metrics.jsonl").read_text().splitlines()]
    assert [row["first_sample_index"] for row in rows] == [320, 336]


def test_specialization_is_forwarded_and_persisted_for_exact_resume(harness, monkeypatch, tmp_path):
    config, _, calls, update = harness
    selected = replace(config, target_source="vocals")

    def selected_update(*args, **kwargs):
        assert kwargs["target_source"] == "vocals"
        return update(*args, **kwargs)

    monkeypatch.setattr(trainer, "grouped_update", selected_update)
    trainer.train(selected, "train.json", tmp_path / "run", stop_after=2)
    assert calls.saves[0]["config"]["target_source"] == "vocals"


def test_teacher_sees_final_remix_and_persists_portable_identity(harness, monkeypatch, tmp_path):
    from stemgenrt import teacher
    config, _, calls, update = harness
    config = replace(config, extra_ordinary_primary_sdr_weight=.2, teacher_coefficient=1.,
                     teacher_checkpoint='movable/teacher.th')
    expected_spec = teacher.specification(1.)
    rendered = []
    class Provider:
        def __init__(self, path, *, coefficient):
            assert path == 'movable/teacher.th' and coefficient == 1.
            self.specification = expected_spec
        def _load(self): pass
        def render(self, mixture):
            rendered.append(mixture.clone())
            assert torch.all(mixture >= 1320)
            return mixture[:, None].repeat(1, 4, 1, 1) * .125
    monkeypatch.setattr(teacher, 'CPUTrainingTeacher', Provider)
    monkeypatch.setattr(trainer, 'remix_batch', lambda mixture, targets, **kwargs:
                        (mixture + 1000, targets, None, None))
    def teacher_update(model, optimizer, ema, mixture, targets, **kwargs):
        assert kwargs.pop('teacher_coefficient') == 1.
        expected = mixture[:, None].repeat(1, 4, 1, 1) * .125
        assert torch.equal(kwargs.pop('teacher_targets'), expected)
        assert model.provenance[teacher.PROVENANCE_KEY] == expected_spec
        return update(model, optimizer, ema, mixture, targets, **kwargs)
    monkeypatch.setattr(trainer, 'grouped_update', teacher_update)
    trainer.train(config, 'train.json', tmp_path / 'run', stop_after=2)
    assert len(rendered) == 2
    assert 'teacher_checkpoint' not in calls.saves[0]['config']
    assert calls.saves[0]['config']['teacher_supervision'] == expected_spec
    rows = [json.loads(line) for line in (tmp_path / 'run/metrics.jsonl').read_text().splitlines()]
    assert all(r['teacher_supervision_sha256'] == teacher.supervision_sha(expected_spec) for r in rows)
    assert len({r['teacher_targets_sha256'] for r in rows}) == 2


def test_resume_uses_restored_objects_cursor_identities_and_rng(harness, monkeypatch, tmp_path):
    config, corpus, calls, _ = harness
    model, optimizer, ema = endpoint(2)
    expected_config = asdict(replace(config, root_weights=corpus.root_weights))
    for key in ("model_family", "compact_hidden_size", "compact_layers", "band_width", "band_global_width", "band_layers",
                "specialist_feature_n_fft", "specialist_waveform_basis"):
        expected_config.pop(key)
    expected_config.pop("target_source")
    expected_config.pop("past_filter")
    expected_config.pop("attention_window")
    expected_config.pop("extra_ordinary_primary_sdr_weight")  # Legacy baseline checkpoint identity.
    expected_config.pop("teacher_coefficient")
    expected_config.pop("teacher_checkpoint")
    expected_config.pop("track_sampling")
    before_python, before_numpy, before_torch = random.getstate(), np.random.get_state(), torch.get_rng_state()
    generator = torch.Generator().manual_seed(173)
    resumed_torch = generator.get_state()
    expected_next_torch = torch.rand(3, generator=generator)
    local_python = random.Random(193)
    resumed_python, expected_next_python = local_python.getstate(), local_python.random()
    local_numpy = np.random.RandomState(197)
    resumed_numpy, expected_next_numpy = local_numpy.get_state(), local_numpy.rand()

    def restore(path, **kwargs):
        assert path == "previous.pt"
        assert not any(key.startswith("specialist_") for key in kwargs["config"])
        assert kwargs == {"sha256": "previous-bytes", "config": expected_config,
                          "data_identity": {"manifest_sha256": "training-bytes",
                                            "sampling_root_order": ["recordings"]},
                          "device": torch.device("cpu"), "precision": "fp32", "restore_rng": True}
        torch.set_rng_state(resumed_torch)
        random.setstate(resumed_python)
        np.random.set_state(resumed_numpy)
        return SimpleNamespace(model=model, optimizer=optimizer, ema=ema,
                               step=2, next_sample_index=352)

    monkeypatch.setattr(trainer, "load_training_checkpoint", restore)
    monkeypatch.setattr(trainer, "ParameterEMA", lambda *args, **kwargs:
                        pytest.fail("Resume must retain the saved EMA"))
    try:
        result = trainer.train(config, "train.json", tmp_path / "resumed", resume="previous.pt",
                               sha256="previous-bytes", stop_after=4)
        assert calls.indices == list(range(352, 384))
        assert [row[:2] for row in calls.updates] == [
            (3, trainer.learning_rate(2, config)), (4, trainer.learning_rate(3, config))]
        assert all(row[2:] == (model, optimizer, ema) for row in calls.updates)
        assert ema.updates == 4 and result["resumed_from_step"] == 2
        assert result["next_sample_index"] == 384
        # Constructing/iterating the DataLoader must not consume restored RNG.
        assert torch.equal(torch.rand(3), expected_next_torch)
        assert random.random() == expected_next_python
        assert np.random.rand() == expected_next_numpy
    finally:
        random.setstate(before_python)
        np.random.set_state(before_numpy)
        torch.set_rng_state(before_torch)


def test_primary_sdr_option_reaches_updates_and_checkpoint_config(harness, monkeypatch, tmp_path):
    config, _, calls, original = harness
    config = replace(config, extra_ordinary_primary_sdr_weight=.2)
    observed = []
    def update(*args, **kwargs):
        observed.append(kwargs["extra_ordinary_primary_sdr_weight"])
        return original(*args, **kwargs)
    monkeypatch.setattr(trainer, "grouped_update", update)
    trainer.train(config, "train.json", tmp_path / "candidate", stop_after=2)
    assert observed == [.2, .2]
    assert calls.saves[0]["config"]["extra_ordinary_primary_sdr_weight"] == .2
    saved_config = json.loads((tmp_path / "candidate/config.json").read_text())
    assert saved_config["extra_ordinary_primary_sdr_weight"] == .2


def test_resume_rejects_inconsistent_cursor_before_creating_run(harness, monkeypatch, tmp_path):
    config, _, calls, _ = harness
    model, optimizer, ema = endpoint(2)
    monkeypatch.setattr(trainer, "load_training_checkpoint", lambda *args, **kwargs:
                        SimpleNamespace(model=model, optimizer=optimizer, ema=ema,
                                        step=2, next_sample_index=351))
    output = tmp_path / "wrong-cursor"
    with pytest.raises(ValueError, match="cursor"):
        trainer.train(config, "train.json", output, resume="previous.pt", sha256="previous-bytes")
    assert not output.exists() and not calls.updates and not calls.saves


def test_resume_identity_keeps_sampler_root_order(harness, monkeypatch, tmp_path):
    config, _, calls, _ = harness
    config = replace(config, root_weights={"second": .5, "first": .5})

    def restore(path, **kwargs):
        # Canonical JSON hashes alone cannot distinguish the two dict orders.
        assert kwargs["config"]["root_weights"] == {"first": .5, "second": .5}
        assert kwargs["data_identity"]["sampling_root_order"] == ["second", "first"]
        raise ValueError("Training data identity changed")

    monkeypatch.setattr(trainer, "load_training_checkpoint", restore)
    output = tmp_path / "wrong-order"
    with pytest.raises(ValueError, match="identity"):
        trainer.train(config, "train.json", output, resume="previous.pt", sha256="previous-bytes")
    assert not output.exists() and not calls.updates


def test_validation_split_disjointness_and_identity_are_bound(harness, monkeypatch, tmp_path):
    from stemgenrt import data
    config, corpus, calls, _ = harness
    validation = SimpleNamespace(sha256="validation-bytes", tracks=("validation-track",), split="valid")
    checks = []

    def manifest(path, **kwargs):
        calls.loads.append((path, kwargs))
        return corpus if path == "train.json" else validation

    monkeypatch.setattr(trainer, "load_manifest", manifest)
    monkeypatch.setattr(data, "require_disjoint", lambda train, valid: checks.append((train, valid)))
    trainer.train(config, "train.json", tmp_path / "validated", validation_manifest="valid.json", stop_after=1)
    assert calls.loads == [("train.json", {"expected_split": "train"}),
                           ("valid.json", {"expected_split": "valid"})]
    assert checks == [(corpus.tracks, validation.tracks)]
    assert calls.saves[0]["data_identity"]["validation_manifest_sha256"] == "validation-bytes"


def test_signal_during_update_saves_only_complete_adam_ema_endpoint(harness, monkeypatch, tmp_path):
    config, _, calls, original_update = harness
    handlers, installs = {}, []
    previous = {signal.SIGTERM: object(), signal.SIGINT: object()}

    def install(sig, handler):
        old = handlers.get(sig, previous[sig])
        handlers[sig] = handler
        installs.append((sig, handler))
        return old

    def update(*args, **kwargs):
        handlers[signal.SIGTERM](signal.SIGTERM, None)
        return original_update(*args, **kwargs)

    monkeypatch.setattr(trainer.signal, "signal", install)
    monkeypatch.setattr(trainer, "grouped_update", update)
    result = trainer.train(config, "train.json", tmp_path / "interrupted")
    assert result["status"] == "interrupted" and result["step"] == 1
    assert len(calls.updates) == len(calls.saves) == 1
    assert calls.saves[0]["step"] == 1 and calls.saves[0]["next_sample_index"] == 336
    assert handlers == previous and len(installs) == 4


def test_threaded_library_call_does_not_install_signal_handlers(harness, monkeypatch, tmp_path):
    config, _, calls, _ = harness
    monkeypatch.setattr(trainer.signal, "signal", lambda *args: pytest.fail("Background signal handler"))
    outcome = []

    def run():
        try:
            outcome.append(trainer.train(config, "train.json", tmp_path / "thread", stop_after=1))
        except BaseException as error:
            outcome.append(error)

    thread = threading.Thread(target=run)
    thread.start()
    thread.join(timeout=10)
    assert not thread.is_alive()
    assert len(outcome) == 1 and isinstance(outcome[0], dict), outcome
    assert outcome[0]["step"] == 1 and len(calls.saves) == 1


@pytest.mark.parametrize("alter_reference", [False, True])
def test_resume_verifies_unsaved_updates_before_journal_or_checkpoint(harness, monkeypatch, tmp_path, alter_reference):
    config, _, calls, update = harness
    config = replace(config, checkpoint_every=1)
    saved = []
    def capture(model, optimizer, ema, *args, **kwargs):
        result = update(model, optimizer, ema, *args, **kwargs)
        if kwargs['step'] == 2:
            saved.append(copy.deepcopy((model, optimizer, ema)))
        return result
    monkeypatch.setattr(trainer, 'grouped_update', capture)
    original = tmp_path / 'original'
    trainer.train(config, 'train.json', original)
    path = original / 'metrics.jsonl'
    original_bytes = path.read_bytes()
    if alter_reference:
        rows = [json.loads(line) for line in original_bytes.splitlines()]
        rows[2]['weighted_loss'] += 1
        # The explicitly supplied digest authenticates this wrong reference;
        # its numerical disagreement must still prevent a checkpoint write.
        path = tmp_path / 'wrong-reference.jsonl'
        path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    model, optimizer, ema = saved[0]
    monkeypatch.setattr(trainer, 'load_training_checkpoint', lambda *a, **k:
        SimpleNamespace(model=model, optimizer=optimizer, ema=ema, step=2, next_sample_index=352))
    calls.saves.clear()
    output = tmp_path / 'resumed'
    kwargs = dict(resume='previous.pt', sha256='previous-bytes', replay_journal=path, replay_sha256=digest)
    if alter_reference:
        with pytest.raises(ValueError, match='scientific update differs at step 3'):
            trainer.train(config, 'train.json', output, **kwargs)
        assert not calls.saves and (output / 'metrics.jsonl').read_bytes() == b''
        assert not (output / 'result.json').exists()
    else:
        result = trainer.train(config, 'train.json', output, **kwargs)
        assert result['replay_verification']['verified_steps'] == [3, 4, 5, 6]
        assert [item['step'] for item in calls.saves] == [3, 4, 5, 6]
        assert calls.saves[-1]['metadata']['replay_verification'] == result['replay_verification']
        assert result['schedule_steps'] == 6 and result['resumed_from_step'] == 2
    assert (original / 'metrics.jsonl').read_bytes() == original_bytes
