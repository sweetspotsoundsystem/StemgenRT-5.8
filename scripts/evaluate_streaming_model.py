#!/usr/bin/env python3
"""Evaluate the current native or ONNX model using an explicit audio manifest."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    model = parser.add_mutually_exclusive_group(required=True)
    model.add_argument("--checkpoint", type=Path,
                       help="Native inference checkpoint or portable training checkpoint")
    model.add_argument("--onnx", type=Path, help="Independent or legacy eight-state ONNX export")
    parser.add_argument("--sha256", required=True, help="Expected SHA-256 of the model file")
    parser.add_argument("--role", choices=("raw", "ema"),
                       help="Checkpoint weights; defaults to native declared role or training EMA")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--unroll-hops", type=int, default=64,
                       help="Group size for continuous evaluation (not a callback benchmark)")
    args = parser.parse_args(argv)
    if args.unroll_hops <= 0:
        parser.error("--unroll-hops must be positive")
    if args.onnx is not None and args.role is not None:
        parser.error("--role only applies to a checkpoint; ONNX weights are already selected")
    if args.output.exists():
        parser.error("--output must be a new file")

    from stemgenrt.evaluation import NativeRenderer, OnnxRenderer, evaluate_manifest

    if args.onnx is not None:
        from stemgenrt.streaming import StreamingSeparator
        renderer = OnnxRenderer(StreamingSeparator(args.onnx, expected_sha256=args.sha256))
        identity = {"format": "onnx", "sha256": args.sha256}
    else:
        import torch
        from stemgenrt.checkpoint import load_model

        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        model = load_model(args.checkpoint, expected_sha256=args.sha256,
                           role=args.role, device="cpu")
        renderer = NativeRenderer(model.eval())
        identity = {"format": "checkpoint", "sha256": args.sha256,
                    "checkpoint_weight_role": model.provenance.get("checkpoint_weight_role"),
                    "role_request": args.role}

    result = evaluate_manifest(renderer, args.manifest, unroll_hops=args.unroll_hops,
                               progress=lambda row: print(json.dumps(row), flush=True))
    result["model"] = identity
    result["batch_size"] = 1
    result["unroll_hops"] = args.unroll_hops
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation also protects against replacing an output created
    # while this potentially long evaluation was running.
    with args.output.open("x") as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(args.output), "aggregate": result["aggregate"]}))


if __name__ == "__main__":
    main()
