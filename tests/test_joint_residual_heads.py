import pytest
import torch
from torch import nn

from stemgenrt import bass_mask_pilot as paired, checkpoint, joint_residual_heads as joint
from stemgenrt.banded import BandSeparator
from stemgenrt.joint_residual_training import render_heads, update
from stemgenrt.specialist import SpecialistSeparator


def test_equal_forward_with_controlled_residual_gradients():
    torch.manual_seed(53)
    drum,bass,vox,mix = torch.randn(4,2,384)
    drum.requires_grad_()
    vox.requires_grad_()
    bass=torch.stack((bass,torch.zeros_like(bass))).requires_grad_()
    result=[joint.predictions(drum,bass,vox,mix,role) for role in joint.RECIPE.arms]
    assert torch.equal(*result) and torch.equal(result[0][3],mix-((drum+bass[0])+vox))
    grads=[torch.autograd.grad(r[3].sum(),(drum,bass,vox),allow_unused=True) for r in result]
    assert torch.equal(grads[0][1],grads[1][1])
    assert all(torch.count_nonzero(grads[0][i])==0 and torch.all(grads[1][i]==-1) for i in (0,2))


def test_real_three_head_update_recovers_and_deploys_all_stems(tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(36)
    dims={"band_width":16,"global_width":16,"layers":1,"feature_n_fft":1024}
    parents=nn.ModuleDict({"drums":SpecialistSeparator(source="drums",waveform_basis=0,**dims),
                           "bass":SpecialistSeparator(source="bass",**dims),"vocals":BandSeparator(sources=("vocals",),**dims)}).eval()
    for child in parents.values():
        child.provenance["training_updates"]=40
    parents.requires_grad_(False)
    ids={s:c*64 for s,c in zip(joint.STEMS,"abc",strict=True)}
    base=checkpoint.state_sha256(parents.state_dict())
    options={"recipe":joint.RECIPE,"parameters":joint.head_parameters}
    models,opts=paired.new_students(parents,**options)
    schedule=[{"id":"a","offset":0,"auxiliary":"solo"},{"id":"a","offset":1024,"auxiliary":"absent"}]
    wave={k:torch.randn(2,768)*.1 for k in (*joint.STEMS,"other")}
    wave["mixture"]=sum(wave.values())
    first=update(models,opts,parents,wave,256,"solo")
    assert first["bass_only"]["loss"]==first["all_stems"]["loss"]
    assert all(set(first[arm]["gradient_norms"])==set(joint.STEMS) for arm in joint.RECIPE.arms)
    path=tmp_path/"joint.pt"
    history=[{"address":schedule[0],**first}]
    initial=paired.save_step(path,models,opts,ids,base,"e"*64,"reference_stems",2,history,**options)
    with pytest.raises(ValueError):
        joint.assemble_joint_endpoints(parents,path,initial,"all_stems",ids)
    clones,clone_opts=paired.new_students(parents,**options)
    restored=paired.restore_students(path,clones,clone_opts,plan_sha256="e"*64,target="reference_stems",parents=ids,
                                     base_state_sha256=base,schedule=schedule,**options)
    second=update(models,opts,parents,wave,256,"absent")
    assert second==update(clones,clone_opts,parents,wave,256,"absent")
    final=paired.save_step(path,clones,clone_opts,ids,base,"e"*64,"reference_stems",2,restored+[{"address":schedule[1],**second}],**options)
    assert checkpoint.state_sha256(parents.state_dict())==base
    for arm in joint.RECIPE.arms:
        endpoint=joint.assemble_joint_endpoints(parents,path,final,arm,ids)
        assert checkpoint.state_sha256(endpoint.state_dict())==checkpoint.state_sha256(models[arm].state_dict())
        assert all(torch.equal(t,parents.state_dict()[k]) for k,t in endpoint.state_dict().items() if k not in joint.head_parameters(endpoint))
        assert all(endpoint[s].provenance["training_updates"]==42 for s in joint.STEMS)
        audio=torch.stack((wave["mixture"],wave["bass"]))
        expected=render_heads(models[arm],audio,256)
        actual=render_heads(endpoint,audio,256)
        assert all(torch.equal(a,b) for a,b in zip(actual,expected,strict=True))
    with pytest.raises(ValueError):
        joint.assemble_joint_endpoints(parents,path,final,"all_stems",{**ids,"drums":"f"*64})
