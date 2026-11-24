"""Paired artist transfer with independent Drums/Vocals safeguards."""
import numpy as np
import torch

from . import checkpoint, joint_residual_heads as joint
from .bass_mask_pilot import physical_other
from .mask_teacher_evaluation import compare
from .peer_readout_training import allowed, boundary_error, energy, read, sha, save


def activity(ref):
    windows = ref[:, :176400].reshape(2, 4, 44100)
    return int((np.square(windows, dtype=np.float64).mean((0,2)) >= 1e-5).sum())


def transfer(p, digest, out, tracks, parents, parent_ids):
    from .joint_residual_training import check_track, read_audio, render_heads
    seal = read(out/"endpoint.json")
    path = out/"paired.pt"
    if seal["plan_sha256"] != digest or seal["paired_sha256"] != sha(path) or seal["steps_per_arm"] != len(p["schedule"]):
        raise ValueError("Transfer needs the completed paired run")
    models = [parents]
    for arm in joint.RECIPE.arms:
        model = joint.assemble_joint_endpoints(parents, path, seal["paired_sha256"], arm, parent_ids)
        if any(seal["final_states"][arm][s] != checkpoint.state_sha256(model[s].state_dict()) for s in joint.STEMS):
            raise ValueError("Sealed specialist weights differ")
        models.append(model)
    folder=out/"transfer"
    folder.mkdir(exist_ok=True)
    seen=set()
    warm=p["warmup_samples"]
    for i,a in enumerate(p["transfer"]):
        allowed(p)
        file=folder/f"{i:03d}.json"
        identity={"plan_sha256":digest,"paired_sha256":seal["paired_sha256"], **a}
        if file.exists():
            saved=read(file)
            if not all(saved[k]==v for k,v in identity.items()):
                raise ValueError("Different saved artist evaluation")
            continue
        track=tracks[a["id"]]
        check_track(track,a)
        signal,peer_signal=np.zeros(4),np.zeros(2)
        active,peer_active=np.zeros(2,dtype=np.int64),np.zeros(2,dtype=np.int64)
        errors,peer_errors,seams=np.zeros((3,4)),np.zeros((3,2)),np.zeros((3,2))
        with torch.inference_mode():
            for at in a["offsets"]:
                allowed(p)
                wave=read_audio(track,at,p,seen)
                audio=torch.stack((wave["mixture"],wave["bass"],wave["mixture"]-wave["bass"]))
                mix=audio[0,:,warm:]
                b,o,d,v=(wave[k][:,warm:].numpy() for k in ("bass","other","drums","vocals"))
                refs=(b,o,b,np.zeros_like(b))
                signal += [energy(x) for x in (b,o,b,audio[2,:,warm:].numpy())]
                peer_signal += [energy(x) for x in (d,v)]
                active += [activity(x) for x in (b,o)]
                peer_active += [activity(x) for x in (d,v)]
                for j,model in enumerate(models):
                    drum,bass,vox=render_heads(model,audio,warm)
                    residual=physical_other(mix,drum,bass[0],vox)
                    predicted=[x.numpy() for x in (bass[0],residual,bass[1],bass[2])]
                    if not np.array_equal(predicted[1],mix.numpy()-((drum.numpy()+predicted[0])+vox.numpy())):
                        raise ValueError("Physical residual differs from the emitted stems")
                    errors[j] += [energy(x.astype(np.float64)-y) for x,y in zip(predicted,refs,strict=True)]
                    peer_errors[j] += [energy(drum.numpy().astype(np.float64)-d),energy(vox.numpy().astype(np.float64)-v)]
                    seams[j] += [boundary_error(x,y) for x,y in zip(predicted[:2],(b,o),strict=True)]
        save(file,{**identity,"signal":signal.tolist(),"error":errors.tolist(),"boundary":seams.tolist(),
                   "active":active.tolist(),"peer_signal":peer_signal.tolist(),"peer_error":peer_errors.tolist(),"peer_active":peer_active.tolist()})
        print("Scored artist",i+1,a["name"],flush=True)


def report(p,digest,out):
    seal=read(out/"endpoint.json")
    if seal["plan_sha256"]!=digest or seal["paired_sha256"]!=sha(out/"paired.pt"):
        raise ValueError("Different paired endpoint")
    all_rows=[];hashes={}
    for i,a in enumerate(p["transfer"]):
        f=out/"transfer"/f"{i:03d}.json"; row=read(f)
        if (row["plan_sha256"]!=digest or row["paired_sha256"]!=seal["paired_sha256"]
                or any(row[k]!=v for k,v in a.items())):
            raise ValueError("Invalid transfer observation")
        all_rows.append(row)
        hashes[f.name]=sha(f)
    rows=[r for r in all_rows if min(r["active"])>0]
    roots=sorted({r["root"] for r in all_rows})
    labels=np.array([r["root"] for r in rows])
    strata={s:np.flatnonzero(labels==s) for s in roots}
    if len(roots)!=2 or not all(len(ix) for ix in strata.values()):
        raise ValueError("Both corpora must have scoreable artists")
    rng=np.random.default_rng(p["seed"])
    draws=np.concatenate([rng.choice(ix,(20000,len(ix))) for ix in strata.values()],axis=1)
    effects={}
    for name,arm,base in (("all_minus_bass_only",2,1),("all_minus_release",2,0),("bass_only_minus_release",1,0)):
        effects[name]=compare(rows,arm,base,strata,draws)
        peers={}
        for j,s in enumerate(("drums","vocals")):
            selected=[r for r in all_rows if r["peer_active"][j]>0]
            err=np.array([r["peer_error"] for r in selected])[:,:,j]
            ref=np.array([r["peer_signal"][j] for r in selected])[:,None]
            scores=np.clip(10*np.log10((ref+1e-12)/(err+1e-12)),-60,60)
            peers[s]={"mean_db":float((scores[:,arm]-scores[:,base]).mean()),"scoreable_artists":len(selected),
                      "counts":{s:sum(r["root"]==s for r in selected) for s in roots}}
        effects[name]["peers"]=peers
    matched,release=effects["all_minus_bass_only"],effects["all_minus_release"]
    checks={"coverage":min(len(ix) for ix in strata.values())>=8,
            "matched_joint":matched["joint_mean_db"]>=.05 and matched["joint_ci95_db"][0]>0,
            "matched_stems":min(matched["mean_db"][:2])>0,
            "matched_corpora":all(sum(v[:2])>0 for v in matched["per_corpus_mean_db"].values()),
            "release_stems":min(release["mean_db"][:2])>=.05,"release_joint":release["joint_ci95_db"][0]>0,
            "release_solo":release["mean_db"][2]>=-.1,"release_absent":release["mean_db"][3]<=.5,
            "release_boundary":max(release["boundary_error_change_db"])<=.25,
            "active_peers":all(v["mean_db"]>=-.10 and min(v["counts"].values())>=5 for e in (matched,release) for v in e["peers"].values())}
    result={"plan_sha256":digest,"paired_sha256":seal["paired_sha256"],"steps_per_arm":seal["steps_per_arm"],
            "dual_active_artists":len(rows),"artist_files":hashes,"effects":effects,"checks":checks,
            "advance_to_development":all(checks.values()),"scope":"Artist-disjoint transfer on known training corpora"}
    save(out/"result.json",result)
    print("Joint residual minus Bass-only routing",matched["joint_mean_db"],matched["joint_ci95_db"],flush=True)
