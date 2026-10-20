#!/usr/bin/env python3
"""Create an experimental execution variant without replacing the release."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stemgenrt.independent_optimization import VARIANTS, optimize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--variant", choices=VARIANTS, default="layout-fp32")
    parser.add_argument("--output", type=Path, required=True, help="New ONNX file")
    args = parser.parse_args()
    report = args.output.with_suffix(".optimization.json")
    if report.exists():
        parser.error("Use a new output path; an existing transformation report is preserved")
    result = optimize(args.source, args.output, expected_sha256=args.sha256, variant=args.variant)
    with report.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
