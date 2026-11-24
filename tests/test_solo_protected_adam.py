import copy

import pytest
import torch

from stemgenrt import checkpoint
from stemgenrt.banded import BandSeparator
from stemgenrt.solo_guard_training import training_config, update
from stemgenrt.solo_protected_adam import project_displacement
from stemgenrt.specialist import SpecialistSeparator


def test_project_an_actual_adam_step_preserving_orthogonal_displacement_and_moments():
    param = torch.nn.Parameter(torch.tensor([1., -2.]))
    optimizer = torch.optim.Adam([param], lr=.1, foreach=False)
    start = param.detach().clone()
    param.grad = torch.tensor([-2., 3.])
    optimizer.step()
    proposed = param.detach().clone()
    state = {k: v.clone() for k, v in optimizer.state[param].items()}
    guard = [torch.tensor([2., 0.])]
    observed = project_displacement([param], [start], guard, apply=False)
    assert observed["conflict"] and not observed["applied"] and torch.equal(param, proposed)
    corrected = project_displacement([param], [start], guard, apply=True)
    assert corrected["applied"] and corrected["before_dot"] == observed["before_dot"]
    assert abs(corrected["after_dot"]) <= corrected["rounding_tolerance"]
    assert param[0] == start[0] and param[1] == proposed[1]
    assert all(torch.equal(value, optimizer.state[param][k]) for k, value in state.items())
    for gradient in (torch.zeros(2), torch.tensor([0., 1.])):
        unchanged = param.detach().clone()
        obs = project_displacement([param], [start], [gradient], apply=True)
        assert not obs["applied"] and obs["after_dot"] <= 0 and torch.equal(param, unchanged)
    with pytest.raises(ValueError):
        project_displacement([param], [start], [], apply=True)


def test_full_network_update_recovery_and_terminal_ema(tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(83)
    dims = dict(band_width=16, global_width=16, layers=1, feature_n_fft=1024)
    parents = torch.nn.ModuleDict({
        "drums": SpecialistSeparator(source="drums", waveform_basis=0, **dims),
        "bass": SpecialistSeparator(source="bass", waveform_basis=0, **dims),
        "vocals": BandSeparator(sources=("vocals",), **dims),
    }).eval().requires_grad_(False)
    base = checkpoint.state_sha256(parents.state_dict())
    data = [[torch.randn(2, 1024) * .04 for _ in range(3)] for _ in range(2)]
    guard = torch.randn(2, 1024) * .07
    models, opt, averages = {}, {}, {}
    p = {"lr": 1e-5, "ema_decay": .98, "schedule": [0, 1], "initial_state_sha256": base}
    for arm in ("plain", "protected"):
        model = copy.deepcopy(parents["bass"]).train().requires_grad_(True)
        model.provenance["parent_training_updates"] = 40
        models[arm] = model
        opt[arm] = torch.optim.Adam(model.parameters(), lr=p["lr"], foreach=False)
        averages[arm] = checkpoint.ParameterEMA(model, decay=p["ema_decay"])
    results = {a: update(m, opt[a], averages[a], parents, data, guard, 256, "cpu", a) for a, m in models.items()}
    for field in ("before_dot", "adam_step_norm", "guard_gradient_norm", "guard_loss", "main_loss"):
        assert results["plain"][field] == results["protected"][field]
    model, optimizer, ema = models["protected"], opt["protected"], averages["protected"]
    assert ema.updates == 1 and len(optimizer.state) == model.parameter_tensor_count
    config = training_config(model, p, "protected")
    ck = tmp_path / "checkpoint.pt"
    stamp = checkpoint.save_training_checkpoint(ck, model, optimizer, ema, step=1, next_sample_index=2,
           config=config, data_identity={"plan": "f" * 64}, metadata={"history": [results["protected"]]}, compressed=False)
    resumed = checkpoint.load_training_checkpoint(ck, sha256=stamp["sha256"], config=config, data_identity={"plan": "f" * 64})
    assert resumed.metadata["history"] == [results["protected"]] and resumed.next_sample_index == 2
    kwargs = (parents, data, guard, 256, "cpu", "protected")
    assert update(model, optimizer, ema, *kwargs) == update(resumed.model, resumed.optimizer, resumed.ema, *kwargs)
    assert checkpoint.state_sha256(model.state_dict()) == checkpoint.state_sha256(resumed.model.state_dict())
    assert checkpoint.state_sha256(ema.parameters) == checkpoint.state_sha256(resumed.ema.parameters)
    assert checkpoint.state_sha256(parents.state_dict()) == base
    final = checkpoint.save_training_checkpoint(ck, resumed.model, resumed.optimizer, resumed.ema, step=2,
             next_sample_index=4, config=config, data_identity={"plan": "f" * 64}, compressed=False)
    deployed = checkpoint.load_model(ck, expected_sha256=final["sha256"], role="ema")
    assert deployed.provenance["training_updates"] == 42
    assert all(torch.equal(value, dict(deployed.named_parameters())[k]) for k, value in ema.parameters.items())
