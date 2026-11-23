"""Paired held-artist measures for an output-head training experiment."""
import numpy as np
import torch

from . import bass_mask_pilot as pilot
from . import checkpoint
from .peer_readout_training import allowed, boundary_error, energy, read, save, sha


def transfer(plan, digest, directory, tracks, parents, parent_digests):
    from .mask_teacher_training import authenticated_track, estimates, get_views, render
    endpoint, path = read(directory/"endpoint.json"), directory/"paired.pt"
    if (endpoint["plan_sha256"] != digest or endpoint["steps_per_arm"] != len(plan["schedule"])
            or endpoint["paired_sha256"] != sha(path)):
        raise ValueError("Require the final paired training checkpoint")
    bass_models = [parents["bass"]]
    for arm in pilot.ARMS:
        model = pilot.assemble_endpoint(parents["bass"], path, endpoint["paired_sha256"], arm, parent_digests)
        if checkpoint.state_sha256(model.state_dict()) != endpoint["state_sha256"][arm]:
            raise ValueError("Loaded endpoint differs from the sealed weights")
        bass_models.append(model)
    warm = plan["warmup_samples"]
    folder = directory/"transfer"
    folder.mkdir(exist_ok=True)
    seen = set()
    for i, item in enumerate(plan["transfer"]):
        allowed(plan)
        dest = folder/f"{i:03d}.json"
        identity = {"id": item["id"], "artist": item["artist"], "root": item["root"], "offsets": item["offsets"],
                    "plan_sha256": digest, "paired_sha256": endpoint["paired_sha256"]}
        if dest.exists():
            if any(read(dest)[k] != v for k, v in identity.items()):
                raise ValueError("Inconsistent saved transfer record")
            continue
        track = authenticated_track(tracks, item, seen)
        signal, active = np.zeros(4), np.zeros(2, dtype=np.int64)
        error, seam = np.zeros((3, 4)), np.zeros((3, 2))
        with torch.inference_mode():
            for at in item["offsets"]:
                allowed(plan)
                audio, truth = get_views(track, at, plan)
                refs = truth.numpy()
                windows = refs[:2, :, :176400].reshape(2, 2, 4, 44100)
                active += (np.square(windows, dtype=np.float64).mean(axis=(1, 3)) >= 1e-5).sum(1)
                signal += [energy(x) for x in (*refs[:3], audio[2, :, warm:].numpy())]
                drums, vocals = (render(parents[s], audio[:1], warm)[0] for s in ("drums", "vocals"))
                mix = audio[0, :, warm:]
                for j, model in enumerate(bass_models):
                    preds = estimates(render(model, audio, warm), mix, drums, vocals)
                    if not torch.equal(preds[1], pilot.physical_other(mix, drums, preds[0], vocals)):
                        raise ValueError("Lost physical residual equality")
                    values = preds.numpy()
                    error[j] += [energy(pred.astype(np.float64)-ref) for pred, ref in zip(values, refs, strict=True)]
                    seam[j] += [boundary_error(pred, ref) for pred, ref in zip(values[:2], refs[:2], strict=True)]
        save(dest, {**identity, "signal": signal.tolist(), "error": error.tolist(),
                    "boundary": seam.tolist(), "active": active.tolist()})
        print("Transfer", i+1, item["name"], flush=True)


def compare(rows, arm, control, strata, draws):
    error = np.asarray([r["error"] for r in rows])
    ref = np.asarray([r["signal"] for r in rows])[:, None]
    score = np.clip(10*np.log10((ref+1e-12)/(error+1e-12)), -60., 60.)
    delta = score[:, arm] - score[:, control]
    delta[:, 3] = 10*np.log10((error[:, arm, 3]+1e-12)/(error[:, control, 3]+1e-12))
    joint = delta[:, :2].mean(1)
    seam = np.asarray([r["boundary"] for r in rows])
    seam_change = 10*np.log10((seam[:, arm]+1e-12)/(seam[:, control]+1e-12))
    return {"mean_db": delta.mean(0).tolist(), "joint_mean_db": float(joint.mean()),
            "joint_ci95_db": np.quantile(joint[draws].mean(1), [.025, .975]).tolist(),
            "per_corpus_mean_db": {s: delta[ix].mean(0).tolist() for s, ix in strata.items()},
            "positive_joint": int((joint > 0).sum()), "boundary_error_change_db": seam_change.mean(0).tolist()}


def report(plan, digest, directory):
    endpoint = read(directory/"endpoint.json")
    if endpoint["plan_sha256"] != digest or endpoint["paired_sha256"] != sha(directory/"paired.pt"):
        raise ValueError("Wrong training endpoint")
    rows, hashes = [], {}
    for i, item in enumerate(plan["transfer"]):
        dest = directory/"transfer"/f"{i:03d}.json"
        row = read(dest)
        if ([row[k] for k in ("id", "root", "artist", "offsets")] != [item[k] for k in ("id", "root", "artist", "offsets")]
                or row["paired_sha256"] != endpoint["paired_sha256"] or row["plan_sha256"] != digest):
            raise ValueError("Transfer identities or checkpoint differ")
        hashes[dest.name] = sha(dest)
        if min(row["active"]) > 0:
            rows.append(row)
    labels = np.array([r["root"] for r in rows])
    strata = {s: np.flatnonzero(labels == s) for s in sorted({r["root"] for r in plan["transfer"]})}
    if len(strata) != 2 or not all(len(ix) for ix in strata.values()):
        raise ValueError("Require scoreable artists from both corpora")
    rng = np.random.default_rng(plan["seed"])
    draws = np.concatenate([rng.choice(ix, (20000, len(ix))) for ix in strata.values()], axis=1)
    release = compare(rows, 2, 0, strata, draws)
    matched = compare(rows, 2, 1, strata, draws)
    checks = {"coverage": min(len(ix) for ix in strata.values()) >= 8,
              "matched_joint": matched["joint_mean_db"] >= .05 and matched["joint_ci95_db"][0] > 0,
              "matched_stems": min(matched["mean_db"][:2]) > 0,
              "matched_corpora": all(sum(v[:2]) > 0 for v in matched["per_corpus_mean_db"].values()),
              "release_stems": min(release["mean_db"][:2]) >= .05,
              "release_joint": release["joint_ci95_db"][0] > 0,
              "release_solo": release["mean_db"][2] >= -.10, "release_absent": release["mean_db"][3] <= .50,
              "release_boundary": max(release["boundary_error_change_db"]) <= .25}
    result = {"plan_sha256": digest, "paired_sha256": endpoint["paired_sha256"], "target": plan["target"],
              "steps_per_arm": endpoint["steps_per_arm"], "scope": "artist-disjoint transfer within known training corpora",
              "artists": len(rows), "transfer_files": hashes, "metrics": ["Bass", "physical Other", "isolated Bass", "absent level"],
              "student_minus_supervised": matched, "student_minus_release": release,
              "supervised_minus_release": compare(rows, 1, 0, strata, draws), "checks": checks,
              "advance_to_development": all(checks.values())}
    save(directory/"result.json", result)
    print("Student-minus-supervised joint dB", matched["joint_mean_db"], matched["joint_ci95_db"], flush=True)
