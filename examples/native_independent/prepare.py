"""Prepare a self-contained M4/M4 Pro worker probe from authenticated models.

This example runs from the source checkout and includes a matching training
archive. Synthetic runtime probes do not select checkpoints by audio quality.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from stemgenrt import checkpoint as ck
from stemgenrt.banded import BandSeparator
from stemgenrt.band_export import export_fp32 as export_band
from stemgenrt.specialist import SpecialistSeparator
from stemgenrt.specialist_export import export_fp32 as export_specialist
from stemgenrt.independent import IndependentStems

if __package__:
    from .analyze import require, sha
else:
    from analyze import require, sha


def write(path, value):
    with Path(path).open("x") as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + "\n")


def verify_training_archive(path, expected_sha256, *, root=ROOT):
    """Check the actual model/trainer/data/objective/recovery/export bytes."""
    require(sha(path) == expected_sha256, "Training archive SHA256 differs")
    files = sorted((root / "stemgenrt").rglob("*.py")) + sorted((root / "stemgenrt").rglob("*.json"))
    files += [root / name for name in ("train_streaming.py", "pyproject.toml", "LICENSE",
        "configs/bass-specialist.json", "configs/drums-specialist.json")]
    require(files and all(file.is_file() and not file.is_symlink() for file in files), "Missing training sources")
    with tarfile.open(path) as archive:
        members = {}
        for member in archive.getmembers():
            parts = member.name.split("/", 1)
            if len(parts) == 2 and member.isfile():
                require(parts[1] not in members, "Duplicate source archive member")
                members[parts[1]] = member
        for file in files:
            key = file.relative_to(root).as_posix()
            member = members.get(key)
            require(member is not None and member.size == file.stat().st_size, "Training archive omitted or changed " + key)
            require(archive.extractfile(member).read() == file.read_bytes(), "Training archive differs from loaded source: " + key)
    return {file.relative_to(root).as_posix(): sha(file) for file in files}


def load_endpoint(path, digest, source, expected_updates):
    model = ck.load_model(path, expected_sha256=digest, role="ema")
    expected = (source,) if source != "joint" else ("drums", "bass", "vocals", "other")
    require(model.sources == expected, "Checkpoint source order differs")
    require(type(model) is (SpecialistSeparator if source in ("bass", "drums") else BandSeparator),
            "Choose a specialist or band model matching this probe")
    require(model.provenance.get("training_updates", 0) > 0, "Checkpoint has no recorded training updates")
    if expected_updates is not None:
        require(model.provenance["current_stage_updates"] == expected_updates, "Checkpoint is not the requested endpoint")
    # The native runner deliberately checks the selected default architecture.
    require(model.band_width == (80 if source == "bass" else 96)
            and model.global_width == (160 if source == "bass" else 192) and model.layers == 2,
            "This probe requires the selected default geometry")
    if source in ("bass", "drums"):
        require(model.feature_n_fft == (4096 if source == "bass" else 1024)
                and model.waveform_basis == (0 if source == "bass" else 256), "Specialist branch geometry differs")
    return model


def prepare(args):
    torch.set_num_threads(1)
    require(not torch.cuda.is_initialized(), "Prepare on CPU")
    training_sources = verify_training_archive(args.training_source, args.training_source_sha256)
    output = args.output.resolve()
    require(not output.exists(), "Use a new output directory")
    output.mkdir(parents=True)
    for name in ("models", "reference", "sdk/include", "sdk/lib", "training-source"):
        (output / name).mkdir(parents=True)
    for name in ("benchmark.cpp", "scheduling.h", "run_macos.py", "analyze.py", "README.md"):
        shutil.copyfile(HERE / name, output / name)
    shutil.copytree(args.sdk_root / "include", output / "sdk/include", dirs_exist_ok=True, symlinks=False)
    for name in ("libonnxruntime.1.26.0.dylib", "libonnxruntime.dylib"):
        shutil.copyfile(args.sdk_root / "lib/libonnxruntime.1.26.0.dylib", output / "sdk/lib" / name)
    for name in ("LICENSE", "ThirdPartyNotices.txt"):
        shutil.copyfile(args.sdk_root / name, output / "sdk" / name)
    shutil.copyfile(ROOT / "LICENSE", output / "LICENSE")
    shutil.copyfile(args.training_source, output / "training-source" / args.training_source.name)
    total = 1024 + 10000
    rng = np.random.default_rng(20261003)
    audio = rng.normal(0, .04, (total, 2, 128)).astype(np.float32)
    times = np.arange(total * 128, dtype=np.float64).reshape(total, 1, 128) / 44100
    audio += (.1 * np.sin(2 * np.pi * times * np.array([220., 251.])[None, :, None])).astype(np.float32)
    audio[:4] = 0
    audio[8, 0, 17] = audio[8, 1, 19] = 1
    audio.astype("<f4").tofile(output / "input.f32")
    models, native, waveforms = {}, {}, {}
    arms = ("vocals", "bass", "drums") + (("joint",) if args.joint_checkpoint else ())
    for arm in arms:
        scratch = args.scratch_specialists and arm in ("bass", "drums")
        if scratch:
            torch.manual_seed(args.seed)
            model = SpecialistSeparator(source=arm)
            digest = None
        else:
            digest = getattr(args, arm + "_sha256")
            model = load_endpoint(getattr(args, arm + "_checkpoint"), digest, arm, args.expected_updates)
        model.eval().requires_grad_(False)
        key = arm + ("-scratch-fp32" if scratch else "-ema-fp32")
        report = (export_specialist if arm in ("bass", "drums") else export_band)(model, output / "models" / (key + ".onnx"))
        write(output / "models" / (key + ".json"), report)
        state, values, emitted = model.initial_state(1), [], []
        with torch.inference_mode():
            for hop in audio[:192]:
                audio_out, state = model.forward_chunk(torch.from_numpy(hop[None].copy()), state)
                tensors = (audio_out, *state)
                require(all(torch.isfinite(value).all() for value in tensors), "Nonfinite native reference")
                values.append(np.concatenate([value.numpy().reshape(-1) for value in tensors]))
                emitted.append(audio_out[0].numpy().copy())
        golden = output / "reference" / (key + ".f32")
        np.asarray(values, dtype="<f4").tofile(golden)
        native[arm], waveforms[arm] = model, np.asarray(emitted)
        models[key] = {"path": "models/" + key + ".onnx", "sha256": report["onnx_sha256"],
            "source_order": list(model.sources), "precision": "fp32", "trained": not scratch,
            "role": "untrained architecture probe" if scratch else "authenticated EMA checkpoint",
            "checkpoint_sha256": digest, "provenance": model.provenance,
            "weight_sha256": report["weight_sha256"], "budget": model.compute_budget(), "interface": report["interface"],
            "reference": str(golden.relative_to(output)), "reference_sha256": sha(golden), "reference_hops": 192,
            "reference_kind": "Native PyTorch FP32; every output and state at every hop"}
        print(json.dumps({"prepared": key, "trained": not scratch}), flush=True)
    combined = IndependentStems(drums=native["drums"], bass=native["bass"], vocals=native["vocals"])
    state, assembly = None, []
    with torch.inference_mode():
        for index, hop in enumerate(audio[:192]):
            result = combined.render(torch.from_numpy(hop[None].copy()), state)
            state = result.state
            for source_index, arm in enumerate(("drums", "bass", "vocals")):
                np.testing.assert_array_equal(result.deployed[0, source_index].numpy(), waveforms[arm][index, 0])
            np.testing.assert_array_equal(result.delayed_mixture[0].numpy(), audio[index - 1] if index else np.zeros_like(hop))
            assembly.append(np.concatenate((result.deployed[0, 3].numpy().reshape(-1), hop.reshape(-1))))
    assembly_path = output / "reference/combined-assembly.f32"
    np.asarray(assembly, dtype="<f4").tofile(assembly_path)
    keys = {arm: next(key for key, value in models.items() if value["source_order"] ==
                    ([arm] if arm != "joint" else ["drums", "bass", "vocals", "other"])) for arm in arms}
    cases = {arm + "_fp32": [keys[arm]] for arm in arms}
    cases["combined_dbv_residual_fp32"] = [keys[arm] for arm in ("drums", "bass", "vocals")]
    names = list(cases)
    interpretation = (f"{len(names)} order-balanced segments per case, not a continuous endurance run. "
        + ("Bass/drums weights are untrained architecture probes. " if args.scratch_specialists else "All three stems use authenticated EMA checkpoints. ")
        + "Synthetic runtime checks do not establish separation quality or plugin/DAW deadlines.")
    protocol = {"schema": "independent-stem-m4-worker-probe-v1", "prepared_utc": datetime.now(timezone.utc).isoformat(),
        "onnxruntime_version": "1.26.0", "allowed_chips": ["Apple M4", "Apple M4 Pro"],
        "sample_rate": 44100, "hop_samples": 128, "graph_alignment_samples": 128, "algorithmic_delay_samples": 256,
        "input_file": "input.f32", "input_sha256": sha(output / "input.f32"), "synthetic_audio_only": True,
        "cases": cases, "models": models, "case_assembly": {name: name == "combined_dbv_residual_fp32" for name in names},
        "assembly_reference": str(assembly_path.relative_to(output)), "assembly_reference_sha256": sha(assembly_path),
        "round_order": [names[index:] + names[:index] for index in range(len(names))],
        "warmup_hops": 1024, "measured_hops_per_trial": 10000, "measured_hops_per_case": 10000 * len(names),
        "preflight_warmup_hops": 64, "preflight_measured_hops": 128,
        "reference_tolerances": {"waveform_atol": 1e-5, "state_atol": 5e-5, "rtol": 2e-5},
        "execution": "CPU, sequential, intra/inter 1, spinning off, KleidiAI off, preallocated tensors, user-interactive QoS",
        "measurement_scope": "Complete graphs, copies, independent states and physical residual Other on one paced worker; no plugin queue or DAW callback",
        "three_model_case": "Independent drums, bass and vocals with separate analysis/synthesis and residual Other from the delayed physical mixture",
        "combined_compute_budget": combined.compute_budget(), "all_three_stems_trained": not args.scratch_specialists,
        "quality_measured": False, "plugin_deadlines_qualified": False, "interpretation": interpretation}
    write(output / "protocol.json", protocol)
    write(output / "provenance.json", {"training_source_archive_sha256": sha(args.training_source),
        "training_source_files_sha256": training_sources, "prepared_with_torch": str(torch.__version__),
        "preparation_source_sha256": sha(__file__), "no_dataset_audio_opened": True, "scratch_seed": args.seed})
    for path in output.rglob("*"):
        require(not path.is_symlink(), "The packet must contain regular files, not aliases")
    write(output / "manifest.json", {"schema": "independent-native-probe-package-v1",
        "files": {str(path.relative_to(output)): sha(path) for path in sorted(output.rglob("*")) if path.is_file()}})
    print(json.dumps({"prepared": str(output), "cases": list(cases), "native_mac_measured": False}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sdk-root", type=Path, required=True, help="Official ONNX Runtime 1.26.0 osx-arm64 SDK")
    parser.add_argument("--training-source", type=Path, required=True)
    parser.add_argument("--training-source-sha256", required=True)
    parser.add_argument("--expected-updates", type=int)
    parser.add_argument("--scratch-specialists", action="store_true", help="Use untrained default bass/drums geometry")
    parser.add_argument("--seed", type=int, default=20261003)
    for arm in ("vocals", "bass", "drums", "joint"):
        parser.add_argument("--" + arm + "-checkpoint", type=Path, required=arm == "vocals")
        parser.add_argument("--" + arm + "-sha256", required=arm == "vocals")
    args = parser.parse_args()
    for arm in ("bass", "drums", "joint"):
        path, digest = getattr(args, arm + "_checkpoint"), getattr(args, arm + "_sha256")
        if (path is None) != (digest is None):
            parser.error(arm + " requires both checkpoint and SHA256")
        if arm in ("bass", "drums") and ((args.scratch_specialists and path is not None) or
                                          (not args.scratch_specialists and path is None)):
            parser.error("Choose both specialist checkpoints or --scratch-specialists")
    if args.expected_updates is not None and args.expected_updates <= 0:
        parser.error("--expected-updates must be positive")
    prepare(args)


if __name__ == "__main__":
    main()
