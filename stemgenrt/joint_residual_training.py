"""Matched training and transfer for residual gradients through three independent output heads."""
import argparse
from pathlib import Path
import random

import numpy as np
import torch
from torch import nn

from . import bass_mask_pilot as paired, checkpoint, data, joint_residual_heads as joint
from .mask_teacher_training import render
from .peer_readout_training import allowed, read, record_lock, save, sha


def validate(plan):
    scheduled = [(a["id"], a["offset"]) for a in plan["schedule"]]
    expected = [(x["id"], at) for x in plan["fit"] for at in x["offsets"]]
    artists = [x["artist"] for x in plan["fit"]+plan["transfer"]]
    if (plan["schema"] != joint.RECIPE.schema or plan["steps"] != len(scheduled) or not scheduled
            or sorted(scheduled) != sorted(expected) or len(set(scheduled)) != len(scheduled)
            or len(set(artists)) != len(artists) or plan["lr"] != paired.LR
            or plan["warmup_samples"] != 88064 or plan["scored_samples"] != 176512
            or any(a["auxiliary"] not in ("solo", "absent") for a in plan["schedule"])):
        raise ValueError("Invalid joint mask training recipe, addresses or artist assignments")
    for f, h in (plan["input_bindings"] | plan["source_bindings"]).items():
        if sha(f) != h:
            raise ValueError("Changed experiment file " + f)


def read_audio(track, offset, plan, authenticated):
    keys = ("mixture", *joint.STEMS, "other")
    length = plan["warmup_samples"]+plan["scored_samples"]
    if offset < 0 or offset % 128 or offset+length > track.effective_frames:
        raise ValueError("Crop falls outside its registered track")
    if track.track_id not in authenticated:
        for k in keys:
            media = track.files[k]
            if sha(media.path) != media.sha256:
                raise ValueError("Audio identity changed for " + track.track_id)
        authenticated.add(track.track_id)
    return {k: data.CounterAddressedCropDataset._read(track.files[k], offset=offset, frames=length) for k in keys}


def check_track(track, item):
    if (track.track_id, track.name, track.root_id) != (item["id"], item["name"], item["root"]):
        raise ValueError("Registered track identity changed")


def render_heads(models, audio, warm):
    drum, vocal = (render(models[s], audio[:1], warm)[0] for s in ("drums", "vocals"))
    return drum, render(models["bass"], audio, warm), vocal


def update(models, opts, parents, wave, warm, aux):
    mix, bass = wave["mixture"], wave["bass"]
    view, target = (bass, bass) if aux == "solo" else (mix-bass, torch.zeros_like(bass))
    audio = torch.stack((mix, view))
    truth = torch.stack((wave["drums"], bass, wave["vocals"], wave["other"], target))[..., warm:]
    with torch.no_grad():
        released = joint.predictions(*render_heads(parents, audio, warm), mix[:,warm:], "all_stems")
        scales = paired.training_scales(released, truth, audio[[0,0,0,0,1],:,warm:])
    metrics = {}
    for arm in joint.RECIPE.arms:
        model, optimizer = models[arm], opts[arm]
        optimizer.zero_grad(set_to_none=True)
        out = joint.predictions(*render_heads(model, audio, warm), mix[:,warm:], arm)
        loss = joint.normalized_loss(out, truth, scales)
        if not torch.isfinite(loss):
            raise ArithmeticError("Nonfinite joint loss")
        loss.backward()
        norms = {s: float(torch.nn.utils.clip_grad_norm_(model[s].backbone.mask_heads.parameters(), 1., error_if_nonfinite=True))
                 for s in joint.STEMS}
        optimizer.step()
        metrics[arm] = {"loss": float(loss.detach()), "gradient_norms": norms}
    return metrics


def fit(p, digest, out, tracks, parents, parent_ids):
    options = {"recipe": joint.RECIPE, "parameters": joint.head_parameters}
    students, opts = paired.new_students(parents, **options)
    base = checkpoint.state_sha256(parents.state_dict())
    path = out/"paired.pt"
    history = []
    if path.exists():
        history = paired.restore_students(path, students, opts, plan_sha256=digest, target="reference_stems",
                                          parents=parent_ids, base_state_sha256=base, schedule=p["schedule"], **options)
    items = {x["id"]: x for x in p["fit"]}
    authenticated = set()
    for address in p["schedule"][len(history):]:
        allowed(p)
        track = tracks[address["id"]]
        check_track(track, items[address["id"]])
        wave = read_audio(track, address["offset"], p, authenticated)
        stats = update(students, opts, parents, wave, p["warmup_samples"], address["auxiliary"])
        if not history and stats["bass_only"]["loss"] != stats["all_stems"]["loss"]:
            raise ValueError("The first matched forward losses differ")
        history.append({"address": address, **stats})
        paired.save_step(path, students, opts, parent_ids, base, digest, "reference_stems", len(p["schedule"]), history, **options)
        print("Completed paired update", len(history), track.name, flush=True)
    seal = {"plan_sha256": digest, "steps_per_arm": len(history), "paired_sha256": sha(path),
            "parents": parent_ids, "final_states": {arm: {s: checkpoint.state_sha256(model[s].state_dict())
                                                          for s in joint.STEMS} for arm,model in students.items()}}
    receipt = out/"endpoint.json"
    if receipt.exists() and read(receipt) != seal:
        raise ValueError("Final model differs from the sealed endpoint")
    if not receipt.exists():
        save(receipt, seal)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("fit","transfer","report"))
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    a = parser.parse_args()
    if sha(a.plan) != a.plan_sha256:
        raise ValueError("Invalid plan checksum")
    p = read(a.plan)
    allowed(p)
    validate(p)
    a.output.mkdir(exist_ok=True,parents=True)
    from . import joint_residual_evaluation as evaluation
    with record_lock(a.output/"run.lock"):
        identity = a.output/"identity.json"
        if identity.exists() and read(identity) != {"sha256":a.plan_sha256}:
            raise ValueError("Output directory belongs to a different experiment")
        if not identity.exists():
            save(identity, {"sha256":a.plan_sha256})
        if a.stage == "report":
            evaluation.report(p, a.plan_sha256, a.output)
            return
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        random.seed(p["seed"])
        np.random.seed(p["seed"])
        torch.manual_seed(p["seed"])
        manifest = data.load_manifest(p["manifest"]["path"],expected_sha256=p["manifest"]["sha256"],expected_split="train",verify_audio_hashes=False)
        tracks = {t.track_id:t for t in manifest.tracks}
        parent_ids = {s:p["models"][s]["sha256"] for s in joint.STEMS}
        parents = nn.ModuleDict({s:checkpoint.load_model(p["models"][s]["path"],expected_sha256=parent_ids[s],role="ema",device="cpu")
                                 for s in joint.STEMS}).eval().requires_grad_(False)
        if a.stage == "fit":
            fit(p,a.plan_sha256,a.output,tracks,parents,parent_ids)
        else:
            evaluation.transfer(p,a.plan_sha256,a.output,tracks,parents,parent_ids)
        if torch.cuda.is_initialized():
            raise RuntimeError("The joint mask pilot requires CPU FP32")


if __name__=="__main__":
    main()
