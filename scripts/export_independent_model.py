#!/usr/bin/env python3
"""Export authenticated independent EMA models and native plugin parity fixtures.

The output is experimental. A matching 13-state host is required. Distribute
the matching source package, including training, with any candidate handoff.
"""
import argparse
import json
from pathlib import Path
import struct
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from stemgenrt import checkpoint as ck
from stemgenrt.independent import IndependentStems, SOURCE_ORDER
from stemgenrt.independent_export import export_fp32
from examples.native_independent.prepare import load_endpoint


def write(path, value):
    with Path(path).open("x") as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + "\n")


def native_fixture(model, output, report, checkpoints):
    """Use the original PyTorch models, never ONNX output, as the oracle."""
    lengths = [1, 127, 128, 129, 255, 256, 257, 16521]
    rng = np.random.default_rng(20261010)
    path = output / "independent-pytorch.bin"
    with path.open("xb") as stream, torch.inference_mode():
        stream.write(b"SGRTG001" + struct.pack("<I", len(lengths)))
        for frames in lengths:
            audio = rng.normal(0, .08, (2, frames)).astype(np.float32)
            phase = np.arange(frames, dtype=np.float64) / 44100
            audio += (.1 * np.sin(2*np.pi*np.array([47., 251.])[:, None]*phase)).astype(np.float32)
            # Pad a partial hop once, then submit precisely one zero flush hop.
            padded = np.pad(audio, ((0, 0), (0, (-frames) % 128 + 128)))
            state, emitted = None, []
            for start in range(0, padded.shape[1], 64*128):
                result = model.render(torch.from_numpy(padded[None, :, start:start+64*128].copy()), state)
                state = result.state
                emitted.append(result.deployed[0].numpy())
            aligned = np.concatenate(emitted, axis=-1)[..., 128:128+frames]
            assert aligned.shape == (4, 2, frames) and np.isfinite(aligned).all()
            stream.write(struct.pack("<I", frames))
            stream.write(audio.astype("<f4").tobytes())
            stream.write(aligned.astype("<f4").tobytes())
    write(output / "independent-pytorch.json", {
        "format": "SGRTG001: LE uint32 case count, then LE uint32 frames, input[2,T], deployed[4,2,T] LE float32",
        "fixture_sha256": ck.file_sha256(path), "generator_sha256": ck.file_sha256(__file__),
        "deployment_graph_sha256": report["onnx_sha256"], "model_state_sha256": report["weight_sha256"],
        "checkpoints": checkpoints, "state_names": report["interface"]["state_names"],
        "state_shapes": report["interface"]["input_shapes"][1:], "frames": lengths,
        "sample_rate": 44100, "hop": 128, "source_order": list(SOURCE_ORDER),
        "runtime_inference_used": False,
        "reference": "Independent native PyTorch EMA models; zero state, one graph flush, exact partial-EOF crop"})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-updates", type=int, required=True)
    for source in SOURCE_ORDER[:3]:
        parser.add_argument("--"+source+"-checkpoint", type=Path, required=True)
        parser.add_argument("--"+source+"-sha256", required=True)
        parser.add_argument("--"+source+"-updates", type=int,
                            help="Expected current-stage updates for this stem; overrides --expected-updates")
        parser.add_argument("--"+source+"-parent-updates", type=int, default=0,
                            help="Expected training updates before this stem's current stage (default: 0)")
    args = parser.parse_args()
    if args.output.exists() or args.expected_updates <= 0:
        parser.error("Require a new output directory and positive expected updates")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    assert not torch.cuda.is_initialized()
    children, checkpoints = {}, {}
    for source in SOURCE_ORDER[:3]:
        path, digest = getattr(args, source+"_checkpoint"), getattr(args, source+"_sha256")
        updates = getattr(args, source+"_updates")
        updates = args.expected_updates if updates is None else updates
        parent_updates = getattr(args, source+"_parent_updates")
        if updates <= 0 or parent_updates < 0:
            parser.error("Require positive stem updates and nonnegative parent updates")
        child = load_endpoint(path, digest, source, updates, expected_parent_updates=parent_updates)
        children[source] = child.eval().requires_grad_(False)
        checkpoints[source] = {"sha256": digest, "role": "ema", "updates": updates,
            "parent_updates": parent_updates, "total_updates": child.provenance["training_updates"],
            "state_sha256": ck.state_sha256(child.state_dict()),
            "ema_parameters_sha256": ck.state_sha256(dict(child.named_parameters()))}
    model = IndependentStems(**children)
    args.output.mkdir(parents=True)
    report = export_fp32(model, args.output / "model.onnx")
    write(args.output / "model.json", report)
    write(args.output / "checkpoints.json", checkpoints)
    native_fixture(model, args.output, report, checkpoints)
    files = sorted((ROOT/"stemgenrt").rglob("*.py")) + sorted((ROOT/"stemgenrt").rglob("*.json"))
    files += sorted((ROOT/"configs").glob("*.json"))
    files += [Path(__file__), ROOT/"pyproject.toml", ROOT/"train_streaming.py"]
    write(args.output / "source-bindings.json", {str(p.relative_to(ROOT)): ck.file_sha256(p) for p in files})
    print(json.dumps({"output": str(args.output), "onnx_sha256": report["onnx_sha256"],
        "bytes": report["bytes"], "states": report["state_count"],
        "updates_per_child": {source: row["updates"] for source, row in checkpoints.items()},
        "native_host_qualified": False}), flush=True)


if __name__ == "__main__":
    main()
