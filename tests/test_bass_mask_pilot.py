import numpy as np
import pytest
import torch

from stemgenrt import bass_mask_pilot as pilot, checkpoint
from stemgenrt.mask_teacher_evaluation import compare
from stemgenrt.mask_teacher_training import render, estimates
from stemgenrt.specialist import SpecialistSeparator


def test_projection_and_shared_auxiliary_gradients():
    torch.manual_seed(51)
    mix, drum, vox, bass, other = torch.randn(5, 2, 384)
    power = (bass.double().square().sum() / (bass.double().square().sum()+other.double().square().sum())).float()
    target = pilot.teacher_target(bass, other, mix, drum, vox, "power")
    torch.testing.assert_close(target, bass+power*(mix-((drum+bass)+vox)-other))
    assert pilot.teacher_target(bass, other, mix, drum, vox, "direct") is bass
    refs, emitted = torch.randn(2, 4, 2, 384)
    base = emitted.clone().requires_grad_()
    scales = pilot.training_scales(emitted, refs, torch.ones_like(refs))
    pair = torch.stack((target, pilot.physical_other(mix, drum, target, vox))).requires_grad_()
    base_grad = torch.autograd.grad(pilot.mask_loss(base, refs, scales), base)[0]
    taught_grad, teacher_grad = torch.autograd.grad(pilot.mask_loss(base, refs, scales, pair),
                                                    (base, pair), allow_unused=True)
    assert torch.equal(base_grad[2:], taught_grad[2:]) and not torch.equal(base_grad[:2], taught_grad[:2])
    assert teacher_grad is None and not scales.requires_grad
    with pytest.raises(ValueError):
        pilot.teacher_target(bass, other, mix, drum, vox, "different")


def test_two_updates_recover_exactly_and_stream_the_selected_arm(tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(12)
    parent = SpecialistSeparator(source="bass", band_width=16, global_width=16, layers=1, feature_n_fft=1024).eval()
    parent.provenance["training_updates"] = 40
    start = checkpoint.state_sha256(parent.state_dict())
    parents = {s: c*64 for s, c in zip(("drums","bass","vocals"),"abc",strict=True)}
    plan = "d"*64
    schedule = [{"id":"track","offset":0}, {"id":"track","offset":4096}]
    audio = torch.randn(3, 2, 768)
    teacher = torch.randn(2, 2, 512)
    truth = torch.randn(4, 2, 512)
    mix, drum, vox = audio[:, :, 256:]
    models, optimizers = pilot.new_students(parent)

    def update(models, optimizers):
        for arm in pilot.ARMS:
            model, opt = models[arm], optimizers[arm]
            opt.zero_grad(set_to_none=True)
            rendered = render(model, audio, 256)
            outputs = estimates(rendered, mix, drum, vox)
            loss = pilot.mask_loss(outputs, truth, torch.ones(4), teacher if arm=="student" else None)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(pilot.mask_parameters(model).values(), 1.)
            opt.step()

    update(models, optimizers)
    path = tmp_path/"paired.pt"
    history = [{"address": schedule[0]}]
    first_hash = pilot.save_step(path, models, optimizers, parents, start, plan, "direct", 2, history)
    with pytest.raises(ValueError):
        pilot.assemble_endpoint(parent, path, first_hash, "student", parents)
    random_after = torch.rand(7)
    update(models, optimizers)
    clones, clone_opts = pilot.new_students(parent)
    loaded = pilot.restore_students(path, clones, clone_opts, plan_sha256=plan, target="direct",
                                    parents=parents, base_state_sha256=start, schedule=schedule)
    assert torch.equal(random_after, torch.rand(7)) and loaded == history
    update(clones, clone_opts)
    digest = pilot.save_step(path, clones, clone_opts, parents, start, plan, "direct", 2, history+[{"address":schedule[1]}])
    assert checkpoint.state_sha256(parent.state_dict()) == start
    for arm in pilot.ARMS:
        endpoint = pilot.assemble_endpoint(parent, path, digest, arm, parents)
        assert all(torch.equal(v, models[arm].state_dict()[k]) for k,v in endpoint.state_dict().items())
        assert all(torch.equal(v, parent.state_dict()[k]) for k,v in endpoint.state_dict().items()
                   if k not in pilot.mask_parameters(endpoint))
        assert torch.equal(render(endpoint,audio,256), render(models[arm],audio,256))
        assert endpoint.provenance["training_updates"] == 42
    for changed in ({"bass":"different"}, {**parents,"vocals":"z"*64}):
        with pytest.raises(ValueError):
            pilot.assemble_endpoint(parent, path, digest, "student", changed)


def test_artist_bootstrap_preserves_paired_errors_and_absent_sign():
    rows = []
    for i in range(4):
        error = np.ones((3,4))*(i+1)
        error[2] *= np.array([.5,.5,.8,2])
        rows.append({"error":error.tolist(),"signal":[90,50,90,60],"boundary":np.ones((3,2)).tolist()})
    strata = {"a":np.arange(2),"b":np.arange(2,4)}
    draws = np.array([[0,0,2,2],[1,1,3,3],[0,1,2,3]])
    result = compare(rows, 2, 1, strata, draws)
    np.testing.assert_allclose(result["mean_db"], 10*np.log10([2,2,1.25,2]), atol=1e-10)
    np.testing.assert_allclose(result["joint_ci95_db"], [10*np.log10(2)]*2, atol=1e-10)
