"""Train and score paired causal Bass mask heads on registered, artist-disjoint audio."""
import argparse
from pathlib import Path
import random

import numpy as np
import torch

from . import bass_mask_pilot as pilot
from . import checkpoint, data
from .model import render_scored_context
from .peer_readout_training import allowed, read, record_lock, save, sha


def validate_plan(plan):
    if (plan["schema"] != pilot.SCHEMA or plan["lr"] != pilot.LR or plan["steps"] != len(plan["schedule"])
            or plan["warmup_samples"] != 88064 or plan["scored_samples"] != 176512):
        raise ValueError("Unknown training protocol")
    fit, transfer = plan["fit"], plan["transfer"]
    artists = [a["artist"] for a in fit+transfer]
    if len(set(artists)) != len(artists) or len({a["id"] for a in fit+transfer}) != len(artists):
        raise ValueError("Fitting and transfer require distinct artists and tracks")
    schedules = [(a["id"], a["offset"]) for a in plan["schedule"]]
    expected = [(a["id"], at) for a in fit for at in a["offsets"]]
    if sorted(schedules) != sorted(expected) or len(schedules) != len(set(schedules)):
        raise ValueError("Training schedule differs from the agreed fitting panel")
    for name, digest in (plan["source_bindings"] | plan["input_bindings"]).items():
        if sha(name) != digest:
            raise ValueError("Changed protocol file " + name)
    if plan["target"] not in ("direct", "power"):
        raise ValueError("A qualified teaching target is required")


def get_views(track, offset, plan):
    length = plan["warmup_samples"] + plan["scored_samples"]
    if type(offset) is not int or offset < 0 or offset % 128 or offset+length > track.effective_frames:
        raise ValueError("Bad physical sample address")
    wave = [data.CounterAddressedCropDataset._read(track.files[k], offset=offset, frames=length)
            for k in ("mixture", "bass", "other")]
    mix, bass, other = wave
    audio = torch.stack((mix, bass, mix-bass))
    refs = torch.stack((bass, other, bass, torch.zeros_like(bass)))[..., plan["warmup_samples"]:]
    return audio, refs


def authenticated_track(tracks, item, seen):
    track = tracks[item["id"]]
    if (track.name, track.root_id) != (item["name"], item["root"]):
        raise ValueError("Registered track identity differs")
    if track.track_id not in seen:
        for role in ("mixture", "bass", "other"):
            file = track.files[role]
            if sha(file.path) != file.sha256:
                raise ValueError("Audio changed since manifest creation")
        seen.add(track.track_id)
    return track


def render(model, audio, warm):
    result = render_scored_context(model, audio, warmup_samples=warm, carry_state=True)
    if result.flush_hops != 1 or not torch.equal(result.physical_mixture, audio[..., warm:]):
        raise ValueError("Lost physical output alignment")
    return result.deployed[:, 0]


def estimates(bass, mix, drums, vocals):
    return torch.stack((bass[0], pilot.physical_other(mix, drums, bass[0], vocals), bass[1], bass[2]))


def train_step(models, optimizers, parents, teacher, audio, truth, plan):
    from .teacher import _predict
    warm = plan["warmup_samples"]
    views = audio[..., warm:]
    mix = views[0]
    with torch.no_grad():
        drums, vocals = (render(parents[s], audio[:1], warm)[0] for s in ("drums", "vocals"))
        released = estimates(render(parents["bass"], audio, warm), mix, drums, vocals)
        scales = pilot.training_scales(released, truth, views[[0, 0, 1, 2]])
        native = torch.from_numpy(_predict(*teacher, audio[0].numpy()))[..., warm:]
        target = pilot.teacher_target(native[1], native[3], mix, drums, vocals, plan["target"])
        teacher_pair = torch.stack((target, pilot.physical_other(mix, drums, target, vocals)))
    metrics = {}
    for arm in pilot.ARMS:
        model, optimizer = models[arm], optimizers[arm]
        optimizer.zero_grad(set_to_none=True)
        output = estimates(render(model, audio, warm), mix, drums, vocals)
        loss = pilot.mask_loss(output, truth, scales, teacher_pair if arm == "student" else None)
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite loss")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(pilot.mask_parameters(model).values(), 1., error_if_nonfinite=True)
        optimizer.step()
        metrics[arm] = {"loss": float(loss.detach()), "gradient_norm": float(norm)}
    return metrics


def fit(plan, digest, directory, tracks, parents, parent_digests):
    from . import teacher as engine
    teacher_path = Path(plan["teacher"]["path"])
    if (sha(teacher_path) != plan["teacher"]["sha256"] or plan["teacher"]["sha256"] != engine.CHECKPOINT_SHA256
            or teacher_path.stat().st_size != engine.CHECKPOINT_BYTES):
        raise ValueError("Wrong frozen offline teacher")
    with torch.random.fork_rng(devices=[]):
        teacher = engine._load_model(teacher_path)
    models, optimizers = pilot.new_students(parents["bass"])
    base_hash = checkpoint.state_sha256(parents["bass"].state_dict())
    live = directory/"paired.pt"
    schedule = plan["schedule"]
    history = pilot.restore_students(live, models, optimizers, plan_sha256=digest, target=plan["target"],
                                     parents=parent_digests, base_state_sha256=base_hash, schedule=schedule) if live.exists() else []
    items = {a["id"]: a for a in plan["fit"]}
    seen = set()
    for address in schedule[len(history):]:
        allowed(plan)
        track = authenticated_track(tracks, items[address["id"]], seen)
        audio, truth = get_views(track, address["offset"], plan)
        metrics = train_step(models, optimizers, parents, teacher, audio, truth, plan)
        history.append({"address": address, **metrics})
        pilot.save_step(live, models, optimizers, parent_digests, base_hash, digest,
                        plan["target"], len(schedule), history)
        print("Paired update", len(history), track.name, flush=True)
    sealed = {"plan_sha256": digest, "paired_sha256": sha(live), "steps_per_arm": len(history),
              "target": plan["target"], "parents": parent_digests,
              "state_sha256": {k: checkpoint.state_sha256(v.state_dict()) for k, v in models.items()}}
    path = directory/"endpoint.json"
    if path.exists() and read(path) != sealed:
        raise ValueError("The sealed endpoint differs from the recorded state")
    if not path.exists():
        save(path, sealed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("fit", "transfer", "report"))
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if sha(args.plan) != args.plan_sha256:
        raise ValueError("Incorrect registered plan digest")
    plan = read(args.plan)
    allowed(plan)
    validate_plan(plan)
    directory = args.output
    directory.mkdir(exist_ok=True, parents=True)
    from . import mask_teacher_evaluation as evaluation
    with record_lock(directory/"pilot.lock"):
        identity = directory/"identity.json"
        if identity.exists() and read(identity) != {"plan_sha256": args.plan_sha256}:
            raise ValueError("This output belongs to a different plan")
        if not identity.exists():
            save(identity, {"plan_sha256": args.plan_sha256})
        if args.stage == "report":
            evaluation.report(plan, args.plan_sha256, directory)
            return
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        random.seed(plan["seed"])
        np.random.seed(plan["seed"])
        torch.manual_seed(plan["seed"])
        ds = data.load_manifest(plan["manifest"]["path"], expected_sha256=plan["manifest"]["sha256"],
                                expected_split="train", verify_audio_hashes=False)
        tracks = {t.track_id: t for t in ds.tracks}
        parents = {s: checkpoint.load_model(p["path"], expected_sha256=p["sha256"], role="ema", device="cpu")
                   for s, p in plan["models"].items()}
        digests = {s: p["sha256"] for s, p in plan["models"].items()}
        if args.stage == "fit":
            fit(plan, args.plan_sha256, directory, tracks, parents, digests)
        else:
            evaluation.transfer(plan, args.plan_sha256, directory, tracks, parents, digests)
        if torch.cuda.is_initialized():
            raise RuntimeError("Paired CPU pilot initialized CUDA")


if __name__ == "__main__":
    main()
