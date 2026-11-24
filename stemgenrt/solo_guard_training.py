"""Train complete matched Bass specialists with an optional post-Adam isolation constraint."""
import argparse
import copy
from pathlib import Path
import random
import signal

import numpy as np
import torch

from . import bass_mask_pilot, checkpoint, data
from .bass_replicate_average import average_bass_sampling_runs
from .mask_teacher_training import authenticated_track, estimates, render
from .peer_readout_training import allowed, read, record_lock, save, sha
from .solo_protected_adam import ARMS, SCHEMA, project_displacement


def parent_model(plan):
    keys = ("original_fixed", "original_alternating", "repeat_fixed", "repeat_alternating")
    parents = [plan["averaging_parents"][k] for k in keys]
    model = average_bass_sampling_runs([(row["path"], row["sha256"]) for row in parents])
    if checkpoint.state_sha256(model.state_dict()) != plan["initial_state_sha256"]:
        raise ValueError("The four-parent model differs from its registered initial weights")
    return model


def training_config(model, plan, arm):
    return {
        "model_family": "specialist", "target_source": "bass", "band_width": model.band_width,
        "band_global_width": model.global_width, "band_layers": model.layers,
        "specialist_feature_n_fft": model.feature_n_fft, "specialist_waveform_basis": model.waveform_basis,
        "precision": "fp32", "steps": len(plan["schedule"]), "batch_size": 2, "data_start": 0,
        "lr": plan["lr"], "ema_decay": plan["ema_decay"],
        "solo_guard": {"schema": SCHEMA, "arm": arm, "initial_state_sha256": plan["initial_state_sha256"]},
    }


def crop(track, key, offset, plan):
    length = plan["warmup_samples"] + plan["scored_samples"]
    if offset < 0 or offset % 128 or offset + length > track.effective_frames:
        raise ValueError("Invalid physical training coordinates")
    return data.CounterAddressedCropDataset._read(track.files[key], offset=offset, frames=length)


def ordinary_loss(model, parents, raw, warm, device):
    mix, bass, other = (tensor.to(device) for tensor in raw)
    inputs = torch.stack((mix, bass, mix - bass))
    scored = inputs[..., warm:]
    truth = torch.stack((bass, other, bass, torch.zeros_like(bass)))[..., warm:]
    with torch.no_grad():
        drums, vocals = (render(parents[s], inputs[:1], warm)[0] for s in ("drums", "vocals"))
        frozen = estimates(render(parents["bass"], inputs, warm), scored[0], drums, vocals)
        scales = bass_mask_pilot.training_scales(frozen, truth, scored[[0, 0, 1, 2]])
    current = estimates(render(model, inputs, warm), scored[0], drums, vocals)
    return bass_mask_pilot.mask_loss(current, truth, scales)


def update(model, optimizer, ema, parents, examples, guard, warm, device, arm):
    if arm not in ARMS or len(examples) != 2:
        raise ValueError("Require a registered Adam arm and two ordinary artists")
    parameters = list(model.parameters())
    guard = guard.to(device)
    prediction = render(model, guard[None], warm)[0]
    truth = guard[:, warm:]
    guard_loss = (prediction - truth).square().mean() / truth.square().mean().clamp_min(1e-8)
    gradient = torch.autograd.grad(guard_loss, parameters)

    optimizer.zero_grad(set_to_none=True)
    main_loss = 0.
    for example in examples:
        part = ordinary_loss(model, parents, example, warm, device) / len(examples)
        if not torch.isfinite(part):
            raise ValueError("Nonfinite main loss")
        part.backward()
        main_loss += float(part.detach())
    norm = float(torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True))
    before = [parameter.detach().clone() for parameter in parameters]
    optimizer.step()
    effect = project_displacement(parameters, before, gradient, apply=arm == "protected")
    ema.update(model, step=ema.updates + 1)
    return {"main_loss": main_loss, "guard_loss": float(guard_loss.detach()), "main_gradient_norm": norm, **effect}


def validate(plan):
    if (plan["schema"] != SCHEMA or plan["lr"] != 1e-5 or plan["ema_decay"] != .98 or
            len(plan["schedule"]) != plan["steps"] or plan["steps"] != 192 or
            (plan["warmup_samples"], plan["scored_samples"]) != (88064, 176512)):
        raise ValueError("Invalid protected Adam recipe")
    fit = {row["id"]: row for row in plan["fit"]}
    held = {row["artist"] for row in plan["transfer"]}
    if len(held) != len(plan["transfer"]) or {r["artist"] for r in fit.values()} & held:
        raise ValueError("Require artist-disjoint training and transfer")
    for step in plan["schedule"]:
        entries = [*step["ordinary"], step["guard"]]
        if (len(step["ordinary"]) != 2 or len({fit[r["id"]]["artist"] for r in entries}) != 3 or
                len({fit[r["id"]]["root"] for r in step["ordinary"]}) != 2 or
                not step["guard"]["bass_power"] >= 1e-5):
            raise ValueError("Require independent audible guard and one main artist per corpus")
    for path, digest in (plan["source_bindings"] | plan["input_bindings"]).items():
        if sha(path) != digest:
            raise ValueError("Experiment input changed: " + path)


def fit_arm(plan, digest, out, tracks, arm, device):
    folder = out / arm
    folder.mkdir(exist_ok=True)
    path, seal = folder / "checkpoint.pt", folder / "endpoint.json"
    if seal.exists():
        record = read(seal)
        if record["plan_sha256"] != digest or record["arm"] != arm or sha(path) != record["sha256"]:
            raise ValueError("The sealed arm is inconsistent")
        return

    start = parent_model(plan)
    parents = {s: checkpoint.load_model(plan["models"][s]["path"], expected_sha256=plan["models"][s]["sha256"],
                                        role="ema", device=device) for s in ("drums", "vocals")}
    parents["bass"] = start.to(device).eval().requires_grad_(False)
    config = training_config(start, plan, arm)
    identity = {"plan_sha256": digest, "manifest_sha256": plan["manifest"]["sha256"],
                "initial_state_sha256": plan["initial_state_sha256"]}
    receipt = folder / "latest.json"
    if path.exists():
        state = checkpoint.load_training_checkpoint(path, config=config, data_identity=identity, device=device, precision="fp32")
        model, optimizer, ema = state.model, state.optimizer, state.ema
        if receipt.exists():
            last = read(receipt)
            if last["step"] > state.step or (last["step"] == state.step and sha(path) != last["sha256"]):
                raise ValueError("Recovery checkpoint differs from its durable receipt")
        history = state.metadata["history"]
        if state.step != len(history) or [h["address"] for h in history] != plan["schedule"][:state.step]:
            raise ValueError("Adam cursor and paired addresses differ")
    else:
        model = copy.deepcopy(start).train().requires_grad_(True)
        count = start.provenance["training_updates"]
        model.provenance = {
            "architecture_version": start.architecture_metadata["version"], "parent_training_updates": count,
            "training_updates": count, "current_stage_updates": 0, "initialization": "four-parent sampling average",
            "parent_model_provenance": copy.deepcopy(start.provenance), "solo_guard": config["solo_guard"],
        }
        optimizer = torch.optim.Adam(model.parameters(), lr=plan["lr"], foreach=False)
        ema = checkpoint.ParameterEMA(model, decay=plan["ema_decay"])
        history = []

    def commit():
        stamp = checkpoint.save_training_checkpoint(path, model, optimizer, ema, step=len(history),
                    next_sample_index=2 * len(history), config=config, data_identity=identity,
                    metadata={"history": history}, compressed=False)
        save(receipt, stamp)

    stop = False

    def interrupt(signum, frame):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGINT, interrupt)
    seen = set()
    items = {row["id"]: row for row in plan["fit"]}
    warm = plan["warmup_samples"]
    for address in plan["schedule"][len(history):]:
        if stop or any(Path(f).exists() for f in plan.get("stop_files", [])):
            if history and (not receipt.exists() or read(receipt)["step"] != len(history)):
                commit()
            return
        examples = []
        for row in address["ordinary"]:
            track = authenticated_track(tracks, items[row["id"]], seen)
            examples.append([crop(track, s, row["offset"], plan) for s in ("mixture", "bass", "other")])
        row = address["guard"]
        track = authenticated_track(tracks, items[row["id"]], seen)
        guard = crop(track, "bass", row["offset"], plan)
        if abs(float(guard[:, warm:].double().square().mean()) - row["bass_power"]) > 1e-12:
            raise ValueError("Guard audio differs from its registration")
        observed = update(model, optimizer, ema, parents, examples, guard, warm, device, arm)
        history.append({"address": address, **observed})
        if len(history) % 8 == 0 or len(history) == plan["steps"] or stop:
            commit()
        print(arm, len(history), "protected" if observed["applied"] else "Adam", flush=True)

    digest_checkpoint = sha(path)
    endpoint = checkpoint.load_model(path, expected_sha256=digest_checkpoint, role="ema")
    save(seal, {"plan_sha256": digest, "arm": arm, "step": len(history), "sha256": digest_checkpoint,
         "ema_state_sha256": checkpoint.state_sha256(endpoint.state_dict()),
         "conflicts": sum(r["conflict"] for r in history), "corrections": sum(r["applied"] for r in history),
         "mean_conflicting_squared_step_fraction": float(np.mean([r["conflicting_squared_step_fraction"] for r in history]))})


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("stage", choices=(*ARMS, "transfer", "report"))
    cli.add_argument("--plan", type=Path, required=True)
    cli.add_argument("--plan-sha256", required=True)
    cli.add_argument("--output", type=Path, required=True)
    cli.add_argument("--device", choices=("cuda", "cpu"), required=True)
    args = cli.parse_args()
    if sha(args.plan) != args.plan_sha256:
        raise ValueError("Incorrect experiment digest")
    plan = read(args.plan)
    allowed(plan)
    validate(plan)
    required = plan["training_device"] if args.stage in ARMS else "cpu"
    if args.device != required:
        raise ValueError("Incorrect device for this stage")

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    random.seed(plan["seed"])
    np.random.seed(plan["seed"])
    torch.manual_seed(plan["seed"])
    dataset = data.load_manifest(plan["manifest"]["path"], expected_sha256=plan["manifest"]["sha256"],
                                 expected_split="train", verify_audio_hashes=False)
    args.output.mkdir(exist_ok=True, parents=True)
    with record_lock(args.output / "training.lock"):
        marker = args.output / "identity.json"
        if marker.exists() and read(marker) != {"sha256": args.plan_sha256}:
            raise ValueError("Output belongs to another experiment")
        if not marker.exists():
            save(marker, {"sha256": args.plan_sha256})
        tracks = {t.track_id: t for t in dataset.tracks}
        if args.stage in ARMS:
            fit_arm(plan, args.plan_sha256, args.output, tracks, args.stage, args.device)
        else:
            from . import solo_guard_evaluation as evaluation
            if args.stage == "transfer":
                evaluation.transfer(plan, args.plan_sha256, args.output, tracks)
            else:
                evaluation.report(plan, args.plan_sha256, args.output)


if __name__ == "__main__":
    main()
