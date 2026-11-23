"""Fit and evaluate a causal peer-conditioned Bass readout from frozen specialist checkpoints."""
import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np
import torch

from stemgenrt import checkpoint, data
from stemgenrt.bass_peer_readout import HOP, N_FEATURES, SCHEMA, PeerFeatures, apply_design, fit_readout
from stemgenrt.model import render_scored_context

ROLES = ("release", "static", "conditional")
VIEWS = ("ordinary_bass", "ordinary_other", "isolated_bass", "absent_bass")


def read(path):
    return json.loads(Path(path).read_bytes())


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(path, value, *, binary=False):
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False, mode="wb" if binary else "w") as stream:
        temporary = Path(stream.name)
        try:
            if binary:
                np.savez_compressed(stream, **value)
            else:
                json.dump(value, stream, indent=2, allow_nan=False)
                stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    temporary.replace(path)


def allowed(plan):
    if any(Path(p).exists() for p in plan.get("stop_files", [])):
        raise SystemExit("Study paused")


def energy(value):
    return np.square(value, dtype=np.float64).sum()


@contextmanager
def record_lock(path):
    with path.open("a+b") as lock:
        if os.name == "nt":
            import msvcrt
            if lock.tell() == 0:
                lock.write(b"0")
                lock.flush()
            lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl
            fcntl.flock(lock, fcntl.LOCK_EX)
        yield


def aligned_audio(track, offset, models, plan):
    warm, center = plan["warmup_samples"], plan["scored_samples"]
    if offset < 0 or offset % HOP or offset + warm + center > track.effective_frames:
        raise ValueError("Crop outside the registered recording")
    wave = {name: data.CounterAddressedCropDataset._read(track.files[name], offset=offset, frames=warm + center)
            for name in ("mixture", "bass", "other")}
    audio = torch.stack((wave["mixture"], wave["bass"], wave["mixture"] - wave["bass"]))
    feature_warm = 512
    aligned = audio[..., warm - feature_warm:]
    outputs = []
    for name in ("drums", "bass", "vocals"):
        result = render_scored_context(models[name], audio, warmup_samples=warm-feature_warm, carry_state=True)
        if result.flush_hops != 1 or not torch.equal(result.physical_mixture, aligned):
            raise ValueError("Physical output alignment failed")
        outputs.append(result.deployed[:, 0])
    estimates = torch.stack(outputs, dim=1).numpy()
    refs = [wave["bass"][:, warm:].numpy(), wave["other"][:, warm:].numpy()]
    return aligned.numpy(), estimates, refs


def moments():
    return {"gram": np.zeros((4, N_FEATURES, N_FEATURES)), "cross": np.zeros((4, N_FEATURES)),
            "error": np.zeros(4), "mixture": np.zeros(4)}


def accumulator():
    return {"signal": np.zeros(4), "error": np.zeros((3, 4)), "boundary": np.zeros((3, 2)),
            "active": np.zeros(2, dtype=np.int64)}


def boundary_error(estimate, ref):
    residual = estimate.astype(np.float64) - ref
    return energy(residual[:, HOP::HOP] - residual[:, HOP-1:-1:HOP])


def one_clip(track, offset, models, plan, result, weights):
    mixture, stems, refs = aligned_audio(track, offset, models, plan)
    feature_warm = 512
    if weights is not None:
        window = plan["sample_rate"]
        for k, ref in enumerate(refs):
            windows = ref[:, :ref.shape[-1] // window * window].reshape(2, -1, window)
            result["active"][k] += (np.square(windows, dtype=np.float64).mean(axis=(0, 2)) >= 1e-5).sum()
    for view, indices in enumerate(((0, 1), (2,), (3,))):
        mix, estimates = mixture[view], stems[view]
        features = PeerFeatures()
        features.design(mix[:, :feature_warm], estimates[..., :feature_warm])
        # Score entire clips and include the boundaries between computation blocks.
        if weights is not None and view == 0:
            boundary_parts = [[] for _ in ROLES]
        for start in range(feature_warm, mix.shape[-1], HOP * 64):
            end = min(start + HOP * 64, mix.shape[-1])
            m, e = mix[:, start:end], estimates[..., start:end]
            core = slice(start-feature_warm, end-feature_warm)
            X = features.design(m, e)
            bass, other = e[1], m - ((e[0] + e[1]) + e[2])
            r_b, r_o = refs[0][:, core], refs[1][:, core]
            if weights is None:
                x64 = X.astype(np.float64)
                gram = x64.T @ x64
                targets = (r_b.astype(np.float64) - bass, other.astype(np.float64) - r_o) if view == 0 else (
                    (r_b.astype(np.float64) - bass) if view == 1 else -bass.astype(np.float64),)
                for k, target in zip(indices, targets, strict=True):
                    result["gram"][k] += gram
                    result["cross"][k] += x64.T @ target.T.reshape(-1)
                    result["error"][k] += energy(target)
                    result["mixture"][k] += energy(m)
            else:
                reference = (r_b, r_o) if view == 0 else ((r_b,) if view == 1 else (np.zeros_like(r_b),))
                for k, ref in zip(indices, reference, strict=True):
                    result["signal"][k] += energy(m if view == 2 else ref)
                for arm, role in enumerate(ROLES):
                    if role == "release":
                        b, o = bass, other
                    else:
                        corrected = apply_design(m, e, X, weights[role])
                        if not np.array_equal(corrected[0], e[0]) or not np.array_equal(corrected[2], e[2]):
                            raise ValueError("A frozen peer changed")
                        b, o = corrected[1], corrected[3]
                    if not np.array_equal(o, m - ((e[0] + b) + e[2])):
                        raise ValueError("Physical residual mismatch")
                    for k, ref, estimate in zip(indices, reference, (b, o) if view == 0 else (b,), strict=True):
                        result["error"][arm, k] += energy(estimate.astype(np.float64) - ref)
                    if view == 0:
                        boundary_parts[arm].append(np.stack((b, o)))
        if weights is not None and view == 0:
            for arm, parts in enumerate(boundary_parts):
                for stem, estimate in enumerate(np.concatenate(parts, axis=-1)):
                    result["boundary"][arm, stem] += boundary_error(estimate, refs[stem])


def artifact(directory, split, index):
    return directory / split / (f"{index:03d}.npz" if split == "fit" else f"{index:03d}.json")


def fit_all(plan, digest, directory):
    stats = []
    receipts = {}
    for i, item in enumerate(plan["fit"]):
        path = artifact(directory, "fit", i)
        with np.load(path, allow_pickle=False) as values:
            if str(values["plan"]) != digest or str(values["track_id"]) != item["id"]:
                raise ValueError("Fitting record does not match the plan")
            stats.append({k: values[k] for k in moments()})
        receipts[path.name] = sha(path)
    weights = {k: fit_readout(stats, conditional=(k == "conditional")).tolist() for k in ROLES[1:]}
    result = {"schema": SCHEMA, "plan_sha256": digest, "parents": plan["models"], "fitting_records": receipts,
              "fit_artists": [x["artist"] for x in plan["fit"]], "weights": weights}
    target = directory / "weights.json"
    if target.exists() and read(target) != result:
        raise ValueError("An existing fitted model differs")
    if not target.exists():
        save(target, result)


def report(plan, digest, directory):
    weight_hash = sha(directory / "weights.json")
    rows = []
    for i, item in enumerate(plan["transfer"]):
        row = read(artifact(directory, "transfer", i))
        if row["plan"] != digest or row["weights_sha256"] != weight_hash or row["track_id"] != item["id"]:
            raise ValueError("Transfer record does not match the fitted model")
        if min(row["active"]) > 0:
            rows.append(row)
    if not rows:
        raise ValueError("No dual-active evaluation artists")
    roots = np.asarray([r["root"] for r in rows])
    names = sorted({v["root"] for v in plan["transfer"]})
    gains, seams = [], []
    for r in rows:
        signal = np.asarray(r["signal"])
        error = np.asarray(r["error"])
        scores = np.clip(10 * np.log10((signal + 1e-12) / (error + 1e-12)), -60., 60.)
        delta = scores - scores[0]
        delta[:, 3] = 10 * np.log10((error[:, 3] + 1e-12) / (error[0, 3] + 1e-12))
        gains.append(delta)
        seam = np.asarray(r["boundary"])
        seams.append(10 * np.log10((seam + 1e-12) / (seam[0] + 1e-12)))
    gains = np.asarray(gains)
    strat = [np.flatnonzero(roots == n) for n in names]
    if any(len(ids) == 0 for ids in strat):
        raise ValueError("Both registered corpora must contribute")
    rng = np.random.default_rng(plan["bootstrap_seed"])
    draws = np.concatenate([rng.choice(ids, size=(20000, len(ids))) for ids in strat], axis=1)
    result = {"plan_sha256": digest, "weight_sha256": weight_hash, "training_corpora_only": True,
              "fit_artists": len(plan["fit"]), "scoreable_transfer_artists": len(rows),
              "transfer_ids": [r["track_id"] for r in rows], "metrics": list(VIEWS), "arms": {}}
    for j, role in enumerate(ROLES[1:], 1):
        vals = gains[:, j]
        joint = vals[:, :2].mean(axis=1)
        means = vals.mean(axis=0)
        ci = np.quantile(joint[draws].mean(axis=1), [.025, .975])
        rootscores = {name: vals[ids].mean(axis=0).tolist() for name, ids in zip(names, strat, strict=True)}
        boundary = np.asarray(seams)[:, j].mean(axis=0)
        checks = {"coverage": min(map(len, strat)) >= 8, "ordinary": bool(np.all(means[:2] >= .10)),
                  "both_corpora": all(min(v[:2]) > 0 for v in rootscores.values()),
                  "joint_lower_positive": bool(ci[0] > 0), "solo": bool(means[2] >= -.1),
                  "absent": bool(means[3] <= .5), "boundary": bool(np.max(boundary) <= .25)}
        result["arms"][role] = {"mean_db": means.tolist(), "per_corpus_mean_db": rootscores,
                                "joint_mean_db": float(joint.mean()), "joint_ci95_db": ci.tolist(),
                                "boundary_error_change_db": boundary.tolist(), "checks": checks}
    extra = (gains[:, 2, :2] - gains[:, 1, :2]).mean(axis=1)
    result["conditional_minus_static_joint_db"] = float(extra.mean())
    result["conditional_minus_static_ci95_db"] = np.quantile(extra[draws].mean(axis=1), [.025, .975]).tolist()
    result["advance_to_development"] = all(result["arms"]["conditional"]["checks"].values()) and extra.mean() > 0
    result["advance_to_development"] = bool(result["advance_to_development"])
    save(directory / "result.json", result)
    print(json.dumps(result), flush=True)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("stage", choices=("statistics", "fit", "transfer", "report"))
    cli.add_argument("--plan", type=Path, required=True)
    cli.add_argument("--plan-sha256", required=True)
    cli.add_argument("--output", type=Path, required=True)
    cli.add_argument("--shard", type=int, default=0)
    cli.add_argument("--shards", type=int, default=1)
    args = cli.parse_args()
    if not 0 <= args.shard < args.shards or sha(args.plan) != args.plan_sha256:
        raise ValueError("Wrong shard or experiment plan")
    plan, output, digest = read(args.plan), args.output, args.plan_sha256
    allowed(plan)
    if (plan["schema"] != SCHEMA or plan["sample_rate"] != 44100
            or {x["artist"] for x in plan["fit"]} & {x["artist"] for x in plan["transfer"]}):
        raise ValueError("Invalid protocol or overlapping artist identities")
    for split in ("fit", "transfer"):
        if len({x["artist"] for x in plan[split]}) != len(plan[split]):
            raise ValueError("Expected one track per artist")
    for name, value in (plan["input_bindings"] | plan["source_bindings"]).items():
        if sha(name) != value:
            raise ValueError("Changed experiment input: " + name)
    output.mkdir(parents=True, exist_ok=True)
    with record_lock(output / "plan.lock"):
        identity = output / "plan.json"
        if identity.exists() and read(identity) != {"sha256": digest}:
            raise ValueError("This output directory already belongs to another plan")
        if not identity.exists():
            save(identity, {"sha256": digest})
    if args.stage == "fit":
        fit_all(plan, digest, output)
        return
    if args.stage == "report":
        report(plan, digest, output)
        return
    split = "fit" if args.stage == "statistics" else "transfer"
    path = output / split
    path.mkdir(exist_ok=True)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    ds = data.load_manifest(plan["manifest"]["path"], expected_split="train",
                            expected_sha256=plan["manifest"]["sha256"], verify_audio_hashes=False)
    tracks = {t.track_id: t for t in ds.tracks}
    models = {k: checkpoint.load_model(v["path"], expected_sha256=v["sha256"], role="ema", device="cpu")
              for k, v in plan["models"].items()}
    weights, weight_hash = None, None
    if split == "transfer":
        fit_all(plan, digest, output)
        model = read(output / "weights.json")
        weights, weight_hash = model["weights"], sha(output / "weights.json")
    for i in range(args.shard, len(plan[split]), args.shards):
        item = plan[split][i]
        allowed(plan)
        target = artifact(output, split, i)
        with record_lock(target.with_suffix(".lock")):
            if target.exists():
                if weights is None:
                    with np.load(target, allow_pickle=False) as cached:
                        if str(cached["plan"]) != digest or str(cached["track_id"]) != item["id"]:
                            raise ValueError("Different cached training record")
                else:
                    cached = read(target)
                    if (cached["plan"], cached["track_id"], cached["weights_sha256"]) != (digest, item["id"], weight_hash):
                        raise ValueError("Different cached transfer record")
                continue
            track = tracks[item["id"]]
            if track.name != item["name"] or track.root_id != item["root"]:
                raise ValueError("Track identity changed")
            for key in ("mixture", "bass", "other"):
                f = track.files[key]
                if sha(f.path) != f.sha256:
                    raise ValueError("Source audio changed")
            record = moments() if weights is None else accumulator()
            with torch.inference_mode():
                for offset in item["offsets"]:
                    allowed(plan)
                    one_clip(track, offset, models, plan, record, weights)
            if weights is None:
                save(target, {**record, "track_id": item["id"], "plan": digest}, binary=True)
            else:
                save(target, {**{k: v.tolist() for k, v in record.items()}, "plan": digest, "track_id": item["id"],
                              "root": item["root"], "artist": item["artist"], "weights_sha256": weight_hash})
            print(split, i, item["name"], flush=True)


if __name__ == "__main__":
    main()
