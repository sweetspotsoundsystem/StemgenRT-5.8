"""Evaluate final EMA endpoints on paired, artist-disjoint Bass and physical Other examples."""
import numpy as np
import torch

from . import bass_mask_pilot, checkpoint
from .mask_teacher_evaluation import compare
from .mask_teacher_training import authenticated_track, estimates, get_views, render
from .peer_readout_training import allowed, boundary_error, energy, read, save, sha
from .solo_guard_training import parent_model
from .solo_protected_adam import ARMS


def endpoints(directory, digest, steps):
    result = {}
    for arm in ARMS:
        path = directory / arm / "checkpoint.pt"
        row = read(directory / arm / "endpoint.json")
        if (row["arm"], row["plan_sha256"], row["step"], row["sha256"]) != (arm, digest, steps, sha(path)):
            raise ValueError("Expected the sealed final training checkpoint for " + arm)
        result[arm] = row
    return result


@torch.inference_mode()
def transfer(plan, digest, directory, tracks):
    seals = endpoints(directory, digest, plan["steps"])
    released = {s: checkpoint.load_model(d["path"], expected_sha256=d["sha256"], role="ema")
                for s, d in plan["models"].items()}
    bass = [released["bass"], parent_model(plan)]
    for arm, seal in seals.items():
        model = checkpoint.load_model(directory / arm / "checkpoint.pt", expected_sha256=seal["sha256"], role="ema")
        if checkpoint.state_sha256(model.state_dict()) != seal["ema_state_sha256"]:
            raise ValueError("Final EMA weights differ from their receipt")
        bass.append(model)
    hashes = {arm: seal["sha256"] for arm, seal in seals.items()}
    folder = directory / "transfer"
    folder.mkdir(exist_ok=True)
    warm = plan["warmup_samples"]
    seen = set()
    for i, item in enumerate(plan["transfer"]):
        allowed(plan)
        dest = folder / f"{i:03d}.json"
        identity = {k: item[k] for k in ("id", "artist", "root", "offsets")}
        identity.update(plan_sha256=digest, endpoint_sha256=hashes)
        if dest.exists():
            existing = read(dest)
            if any(existing[k] != v for k, v in identity.items()):
                raise ValueError("Saved transfer record differs from its inputs")
            continue
        track = authenticated_track(tracks, item, seen)
        signal = np.zeros(4)
        active = np.zeros(2, dtype=np.int64)
        error = np.zeros((len(bass), 4))
        seams = np.zeros((len(bass), 2))
        for at in item["offsets"]:
            allowed(plan)
            audio, truth = get_views(track, at, plan)
            refs = truth.numpy()
            windows = refs[:2, :, :176400].reshape(2, 2, 4, 44100)
            active += (np.square(windows, dtype=np.float64).mean(axis=(1, 3)) >= 1e-5).sum(1)
            signal += [energy(x) for x in (*refs[:3], audio[2, :, warm:].numpy())]
            drum, vocal = (render(released[s], audio[:1], warm)[0] for s in ("drums", "vocals"))
            mix = audio[0, :, warm:]
            for j, model in enumerate(bass):
                predictions = estimates(render(model, audio, warm), mix, drum, vocal)
                if not torch.equal(predictions[1], bass_mask_pilot.physical_other(mix, drum, predictions[0], vocal)):
                    raise ValueError("Physical Other differs from the streaming residual")
                preds = predictions.numpy()
                error[j] += [energy(pred.astype(np.float64) - ref) for pred, ref in zip(preds, refs, strict=True)]
                seams[j] += [boundary_error(pred, ref) for pred, ref in zip(preds[:2], refs[:2], strict=True)]
        save(dest, {**identity, "active": active.tolist(), "signal": signal.tolist(),
                    "error": error.tolist(), "boundary": seams.tolist()})
        print("Transfer", i + 1, item["name"], flush=True)


def with_isolation(rows, arm, base, strata, draws):
    result = compare(rows, arm, base, strata, draws)
    signal = np.array([row["signal"][2] for row in rows])
    errors = np.array([row["error"] for row in rows])[:, :, 2]
    solo = np.clip(10 * np.log10((signal[:, None] + 1e-12) / (errors + 1e-12)), -60., 60.)
    diffs = solo[:, arm] - solo[:, base]
    result["isolated_ci95_db"] = np.quantile(diffs[draws].mean(1), [.025, .975]).tolist()
    return result


def report(plan, digest, directory):
    seals = endpoints(directory, digest, plan["steps"])
    hashes = {a: s["sha256"] for a, s in seals.items()}
    rows, files = [], {}
    for i, item in enumerate(plan["transfer"]):
        path = directory / "transfer" / f"{i:03d}.json"
        row = read(path)
        if (any(row[k] != item[k] for k in ("id", "artist", "root", "offsets")) or
                row["plan_sha256"] != digest or row["endpoint_sha256"] != hashes):
            raise ValueError("Transfer record identity differs")
        files[path.name] = sha(path)
        if min(row["active"]) > 0:
            rows.append(row)
    labels = np.array([r["root"] for r in rows])
    strata = {s: np.flatnonzero(labels == s) for s in sorted({r["root"] for r in plan["transfer"]})}
    if len(strata) != 2 or not all(len(idx) for idx in strata.values()):
        raise ValueError("Both corpora require scoreable artists")
    rng = np.random.default_rng(plan["seed"])
    draws = np.concatenate([rng.choice(ix, (20000, len(ix))) for ix in strata.values()], axis=1)
    arms = [(3, 2, "protected_minus_plain"), (3, 1, "protected_minus_four_parent"),
            (3, 0, "protected_minus_release"), (2, 1, "plain_minus_four_parent"), (1, 0, "four_parent_minus_release")]
    pairs = {name: with_isolation(rows, arm, control, strata, draws) for arm, control, name in arms}
    matched, parent, released = [pairs[name] for _, _, name in arms[:3]]
    checks = {
        "coverage": min(len(ix) for ix in strata.values()) >= 8,
        "corrections": seals["protected"]["corrections"] > 0 and seals["plain"]["corrections"] == 0,
        "matched_isolated": matched["mean_db"][2] >= .30 and matched["isolated_ci95_db"][0] > 0,
        "matched_joint": matched["joint_mean_db"] >= -.03,
        "matched_stems": min(matched["mean_db"][:2]) >= -.05,
        "parent_isolated": parent["mean_db"][2] >= .25,
        "parent_joint": parent["joint_mean_db"] >= 0,
        "parent_stems": min(parent["mean_db"][:2]) >= -.03,
        "release_solo": released["mean_db"][2] >= -.10,
        "release_stems": min(released["mean_db"][:2]) >= .05,
        "release_absent": released["mean_db"][3] <= .50,
        "release_boundaries": max(released["boundary_error_change_db"]) <= .25,
    }
    save(directory / "result.json", {"plan_sha256": digest, "endpoint_sha256": hashes, "transfer_files": files,
         "artists": len(rows), "per_corpus": {k: len(v) for k, v in strata.items()}, "steps_per_arm": plan["steps"],
         "metrics": ["Bass", "physical Other", "isolated Bass", "absent level"], "comparisons": pairs,
         "checks": checks, "advance_to_development": all(checks.values()),
         "scope": "Previously used transfer panel within known training corpora"})
    print("Isolated Bass versus matched Adam", matched["mean_db"][2], matched["isolated_ci95_db"], flush=True)
