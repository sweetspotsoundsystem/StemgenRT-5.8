"""Download the exact released ONNX model shared with StemgenRT."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
from urllib.request import urlopen


MODELS = json.loads((Path(__file__).resolve().parents[1]
                     / "stemgenrt/streaming_models.json").read_text())


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def download(destination, *, release="current"):
    """Publish verified bytes, preserving any existing destination."""
    model = MODELS[release]
    if not model.get("url"):
        raise ValueError(
            f"{release} downloads have been withdrawn pending provenance and rights review. "
            "Only local compatibility metadata is retained; use --release current for the current model."
        )
    destination = Path(destination)
    if destination.is_symlink():
        raise FileExistsError(f"Refusing a symlink destination: {destination}")
    if destination.exists():
        if destination.stat().st_size == model["bytes"] and digest(destination) == model["sha256"]:
            return destination
        raise FileExistsError(f"Existing file is a different model: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, prefix=".hop128-", delete=False) as output:
            temporary = Path(output.name)
            checksum = hashlib.sha256()
            count = 0
            with urlopen(model["url"], timeout=30) as response:
                while True:
                    block = response.read(1024 * 1024)
                    if not block:
                        break
                    count += len(block)
                    if count > model["bytes"]:
                        raise ValueError("Downloaded model exceeds its expected size")
                    checksum.update(block)
                    output.write(block)
        if count != model["bytes"] or checksum.hexdigest() != model["sha256"]:
            raise ValueError("Downloaded model size or SHA-256 differs")
        # Same-directory hard link publishes complete bytes and fails if another
        # process created the destination while the download was in progress.
        os.link(temporary, destination)
        return destination
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--release", choices=tuple(MODELS), default="current")
    args = parser.parse_args()
    destination = args.output or (Path(__file__).resolve().parents[1]
                                  / "models" / MODELS[args.release]["filename"])
    try:
        result = download(destination, release=args.release)
    except ValueError as error:
        parser.error(str(error))
    print(f"Model ready: {result}")


if __name__ == "__main__":
    main()
