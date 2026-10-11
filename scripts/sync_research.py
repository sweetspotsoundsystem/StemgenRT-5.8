#!/usr/bin/env python3
"""Record and verify a local source comparison for an intentional research port.

Hash working files, including uncommitted sources, without copying research
into the package. Numerical equivalence requires separate integration/parity
checks. Keep the resulting manifest local and outside the public checkout.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
SOURCE_SUFFIXES = {".py", ".cpp", ".h", ".sh", ".ps1"}


def digest(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.resolve() != path.absolute():
        raise ValueError(f"Require a regular, unaliased file: {path}")
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def working_files(root):
    names = subprocess.check_output(
        ["git", "-C", str(root), "ls-files", "-co", "--exclude-standard", "-z"])
    return sorted({name.decode() for name in names.split(b"\0") if name})


def comparison(source, production, plan_path, *, public=ROOT, extra_sources=()):
    source, production, public, plan_path = map(
        lambda value: Path(value).absolute(), (source, production, public, plan_path))
    if source == public or not plan_path.is_relative_to(source):
        raise ValueError("Use a separate public checkout and a plan inside the research checkout")
    plan_digest = digest(plan_path)
    plan = json.loads(plan_path.read_text())
    bound = plan["source_bindings"]
    if not isinstance(bound, dict) or not bound:
        raise ValueError("The active plan must bind its actual source files")
    paths = {path for path in (source / "research/direct").iterdir()
             if path.suffix in SOURCE_SUFFIXES or path.suffix == ".json"}
    paths.update(Path(name) for name in bound if Path(name).suffix in SOURCE_SUFFIXES)
    paths.update(production.glob("*.py"))
    paths.update((production / "tests").glob("*.py"))
    paths.update(production / name for name in ("full_config.json", "corpus_config.json"))
    for name in extra_sources:
        path = Path(name)
        path = path if path.is_absolute() else source / path
        if path.suffix not in SOURCE_SUFFIXES | {".json"}:
            raise ValueError(f"Additional comparison input must be source code or JSON: {path}")
        paths.add(path)
    inventory = {}
    for path in sorted(paths):
        if path.is_relative_to(source):
            key = "repository/" + path.relative_to(source).as_posix()
        elif path.is_relative_to(production):
            key = "production/" + path.relative_to(production).as_posix()
        elif str(path) in bound:
            key = "external/" + str(path)
        else:
            raise ValueError(f"An unbound source lies outside the supplied roots: {path}")
        actual = digest(path)
        expected = bound.get(str(path))
        if expected is not None and actual != expected:
            raise ValueError(f"Active plan source changed: {path}")
        inventory[key] = {"sha256": actual, "bytes": path.stat().st_size,
                          "bound_by_active_plan": expected is not None}
    trainer = plan.get("trainer_source")
    if trainer is not None and bound.get(trainer["path"]) != trainer["sha256"]:
        raise ValueError("The active trainer must belong to the plan's source bindings")
    public_files = {name: digest(public / name) for name in working_files(public)}
    if digest(plan_path) != plan_digest:
        raise ValueError("Active plan changed during comparison")
    return {"schema": "hs-tasnet-port-source-comparison-v1",
            "includes_working_tree_changes": True,
            "active_plan": {"path": plan_path.relative_to(source).as_posix(), "sha256": plan_digest},
            "research_files": inventory, "public_files": public_files,
            "scope": "Source identity only; numerical parity, full frozen inputs, tests and hardware require separate evidence"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--production-root", type=Path, required=True)
    parser.add_argument("--active-plan", type=Path, required=True)
    parser.add_argument("--extra-source", type=Path, action="append", default=[],
                        help="Additional selected source or JSON file; repeat as needed. "
                             "Relative paths are resolved against --source-root")
    parser.add_argument("--manifest", type=Path, required=True,
                        help="Local review artifact outside the public checkout")
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    manifest = args.manifest.absolute()
    if manifest.resolve().is_relative_to(ROOT):
        raise ValueError("Keep the local comparison manifest outside the public checkout")
    current = comparison(args.source_root, args.production_root, args.active_plan,
                         extra_sources=args.extra_source)
    if args.write:
        with manifest.open("x") as stream:
            json.dump(current, stream, indent=2, sort_keys=True)
            stream.write("\n")
    elif json.loads(manifest.read_text()) != current:
        raise ValueError("Sources or active plan differ from the reviewed manifest")
    print(f"Verified {len(current['research_files'])} research/helper sources and "
          f"{len(current['public_files'])} portable files; numerical checks remain separate")


if __name__ == "__main__":
    main()
