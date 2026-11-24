"""Fit and score frequency-calibrated offline teaching targets on artist-disjoint data."""
import argparse
from pathlib import Path

import numpy as np
import torch

from . import checkpoint, data, teacher, teacher_frequency_projection as projection
from .mask_teacher_training import authenticated_track, render
from .peer_readout_training import allowed, energy, read, record_lock, save, sha

SCHEMA = "frequency-calibrated-teacher-target-v1"
ARMS = ("release", "direct", "power", "scalar", "frequency")


def render_clip(track, offset, engine, models, p):
    warm, scored = p["warmup_samples"], p["scored_samples"]
    if offset < 0 or offset % 128 or offset+warm+scored > track.effective_frames:
        raise ValueError("Invalid physical crop")
    mix, bass, other = [data.CounterAddressedCropDataset._read(track.files[k],offset=offset,frames=warm+scored)
                        for k in ("mixture","bass","other")]
    with torch.inference_mode():
        d, release, v = [render(models[k],mix[None],warm)[0].numpy() for k in ("drums","bass","vocals")]
    native = teacher._predict(*engine, mix.numpy())[:,:,warm:]
    m, rb, ro = [x.numpy()[:,warm:] for x in (mix,bass,other)]
    direct, native_other = native[1], native[3]
    direct_other = m-((d+direct)+v)
    return m,rb,ro,d,release,v,direct,native_other,direct_other,direct_other-native_other


def artist_record(item, track, engine, models, p, fitted):
    ref, active = np.zeros(2), np.zeros(2,dtype=np.int64)
    gram, cross, direct_error, mixture_energy = np.zeros(projection.BANDS), np.zeros((2,projection.BANDS)), np.zeros(2), 0.
    scores = np.zeros((len(ARMS),2))
    for at in item["offsets"]:
        allowed(p)
        m,rb,ro,d,release,v,tb,to,direct_other,x = render_clip(track,at,engine,models,p)
        for j,r in enumerate((rb,ro)):
            ref[j] += energy(r)
            active[j] += (np.square(r[:,:176400].reshape(2,4,44100),dtype=np.float64).mean((0,2))>=1e-5).sum()
        if fitted is None:
            target_b = rb.astype(np.float64)-tb
            target_o = direct_other.astype(np.float64)-ro
            g,c = projection.clip_statistics(x,target_b,target_o)
            gram += g
            cross += c
            direct_error += [energy(target_b),energy(target_o)]
            mixture_energy += energy(m)
        else:
            power = np.float32(np.clip(energy(tb)/max(energy(tb)+energy(to),1e-20),.05,.95))
            arms = (release,tb,tb+power*x,*(projection.projected_bass(tb,x,fitted[k]["weights"])
                                            for k in ("scalar","frequency")))
            for j,b in enumerate(arms):
                o = m-((d+b)+v)
                if not np.isfinite(o).all() or np.max(np.abs((((d+b)+v)+o)-m))>=2e-6:
                    raise ValueError("Lost physical output consistency")
                scores[j] += [energy(b.astype(np.float64)-rb),energy(o.astype(np.float64)-ro)]
    result = {**item,"active":active.tolist(),"reference":ref.tolist()}
    if fitted is None:
        result.update(gram=gram.tolist(),cross=cross.tolist(),direct_error=direct_error.tolist(),mixture_energy=float(mixture_energy))
    else:
        result["errors"] = scores.tolist()
    return result


def records(p, digest, out, split, weights_hash):
    values, hashes = [], {}
    for i,item in enumerate(p[split]):
        file = out/split/f"{i:03d}.json"
        row = read(file)
        if (row["plan_sha256"]!=digest or row["weights_sha256"]!=weights_hash or any(row[k]!=v for k,v in item.items())):
            raise ValueError("Artist record identity differs")
        values.append(row)
        hashes[file.name] = sha(file)
    return values,hashes


def fit(p,digest,out):
    rows,hashes = records(p,digest,out,"fit",None)
    coefficients = {role:projection.fit_projection(rows,scalar=(role=="scalar")) for role in ("scalar","frequency")}
    weights = {"plan_sha256":digest,"fit_records":hashes,"coefficients":coefficients,"edges_hz":projection.EDGES_HZ,
               "teacher_sha256":p["teacher"]["sha256"],"fit_artists":[a["artist"] for a in p["fit"]]}
    weights["edges_hz"] = list(weights["edges_hz"])
    file = out/"weights.json"
    if file.exists() and read(file)!=weights:
        raise ValueError("Different calibration weights already sealed")
    if not file.exists():
        save(file,weights)


def report(p,digest,out):
    fit(p,digest,out)
    weight_hash = sha(out/"weights.json")
    items,hashes = records(p,digest,out,"transfer",weight_hash)
    rows = [r for r in items if min(r["active"])>0]
    roots = np.array([r["root"] for r in rows])
    strata = {s:np.flatnonzero(roots==s) for s in sorted({r["root"] for r in items})}
    if len(strata)!=2 or not all(len(ix) for ix in strata.values()):
        raise ValueError("Both corpora must contribute")
    scores = np.clip(10*np.log10((np.asarray([r["reference"] for r in rows])[:,None]+1e-12)
                                 /(np.asarray([r["errors"] for r in rows])+1e-12)),-60.,60.)
    rng = np.random.default_rng(p["seed"])
    draws = np.concatenate([rng.choice(ix,(20000,len(ix))) for ix in strata.values()],axis=1)
    effects = {}
    for candidate,control in (("frequency","release"),("frequency","direct"),("frequency","scalar"),
                              ("frequency","power"),("scalar","direct"),("direct","release"),("power","release")):
        gain = scores[:,ARMS.index(candidate)]-scores[:,ARMS.index(control)]
        vals = np.column_stack((gain,gain.mean(1)))
        effects[candidate+"_minus_"+control] = {"mean_db":vals.mean(0).tolist(),
            "ci95_db":np.quantile(vals[draws].mean(1),[.025,.975],axis=0).tolist(),
            "corpus_means_db":{s:vals[ix].mean(0).tolist() for s,ix in strata.items()}}
    rel,direct,scalar = [effects["frequency_minus_"+k] for k in ("release","direct","scalar")]
    checks = {"coverage":min(map(len,strata.values()))>=8,
              "release_ordinary":min(rel["mean_db"][:2])>=.2 and all(min(v[:2])>0 for v in rel["corpus_means_db"].values()),
              "release_joint":rel["mean_db"][2]>=.5 and rel["ci95_db"][0][2]>0,
              "direct_bass_cost":direct["mean_db"][0]>=-.5,"direct_joint_benefit":direct["ci95_db"][0][2]>0,
              "frequency_benefit":scalar["mean_db"][2]>=.05 and scalar["ci95_db"][0][2]>0}
    result = {"plan_sha256":digest,"weights_sha256":weight_hash,"transfer_records":hashes,"scoreable_artists":len(rows),
              "effects":effects,"checks":checks,"qualifies_for_student":all(checks.values()),
              "scope":"Offline teaching targets on known corpora","student_optimizer_updates":0}
    save(out/"result.json",result)
    print(result["checks"],flush=True)


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("stage",choices=("statistics","fit","transfer","report"))
    cli.add_argument("--plan",type=Path,required=True)
    cli.add_argument("--plan-sha256",required=True)
    cli.add_argument("--output",type=Path,required=True)
    cli.add_argument("--shard",type=int,default=0)
    cli.add_argument("--shards",type=int,default=1)
    a=cli.parse_args()
    if sha(a.plan)!=a.plan_sha256 or not 0<=a.shard<a.shards:
        raise ValueError("Invalid plan or shard")
    p=read(a.plan);allowed(p)
    artists=[x["artist"] for x in p["fit"]+p["transfer"]]
    if (p["schema"]!=SCHEMA or len(artists)!=len(set(artists)) or (p["warmup_samples"],p["scored_samples"])!=(88064,176512)
            or any(sha(f)!=h for f,h in (p["source_bindings"]|p["input_bindings"]).items())):
        raise ValueError("Changed experiment or overlapping artists")
    out=a.output;out.mkdir(exist_ok=True,parents=True)
    with record_lock(out/"plan.lock"):
        marker=out/"identity.json"
        if marker.exists() and read(marker)!={"sha256":a.plan_sha256}:
            raise ValueError("Incompatible output directory")
        if not marker.exists():
            save(marker,{"sha256":a.plan_sha256})
        if a.stage=="fit":
            fit(p,a.plan_sha256,out)
            return
        if a.stage=="report":
            report(p,a.plan_sha256,out)
            return
    split="fit" if a.stage=="statistics" else "transfer"
    fitted=weight_hash=None
    if split=="transfer":
        weights=read(out/"weights.json")
        if weights["plan_sha256"]!=a.plan_sha256:
            raise ValueError("Wrong frozen projection")
        fitted,weight_hash=weights["coefficients"],sha(out/"weights.json")
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    teacher_path=Path(p["teacher"]["path"])
    if (teacher_path.stat().st_size!=teacher.CHECKPOINT_BYTES or sha(teacher_path)!=teacher.CHECKPOINT_SHA256
            or p["teacher"]["sha256"]!=teacher.CHECKPOINT_SHA256):
        raise ValueError("Invalid offline teacher")
    with torch.random.fork_rng(devices=[]):
        engine=teacher._load_model(teacher_path)
    models={s:checkpoint.load_model(v["path"],expected_sha256=v["sha256"],role="ema") for s,v in p["models"].items()}
    ds=data.load_manifest(p["manifest"]["path"],expected_sha256=p["manifest"]["sha256"],expected_split="train",verify_audio_hashes=False)
    tracks={t.track_id:t for t in ds.tracks};seen=set()
    folder=out/split;folder.mkdir(exist_ok=True)
    for i in range(a.shard,len(p[split]),a.shards):
        allowed(p);item=p[split][i];file=folder/f"{i:03d}.json"
        with record_lock(file.with_suffix(".lock")):
            if file.exists():
                row=read(file)
                if row["plan_sha256"]!=a.plan_sha256 or row["weights_sha256"]!=weight_hash or any(row[k]!=v for k,v in item.items()):
                    raise ValueError("Wrong cached artist")
                continue
            track=authenticated_track(tracks,item,seen)
            row=artist_record(item,track,engine,models,p,fitted)
            save(file,{**row,"plan_sha256":a.plan_sha256,"weights_sha256":weight_hash})
            print(split,i+1,item["name"],flush=True)
    if torch.cuda.is_initialized():
        raise ValueError("Offline target calibration requires CPU FP32")


if __name__=="__main__":
    main()
