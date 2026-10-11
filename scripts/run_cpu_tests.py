#!/usr/bin/env python3
"""Run CPU pytest without its optional 'current' directory symlinks.

The monitored research artifact allocation rejects symlinks, including those
that pytest creates briefly while a test is running. Disabling only that
bookkeeping helper preserves real fixture directories and their accounting.
"""
import os
from pathlib import Path
import sys
from unittest.mock import patch


def main():
    os.environ.update(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                      OPENBLAS_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1")
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import pytest
    import _pytest.pathlib
    if not callable(getattr(_pytest.pathlib, "_force_symlink", None)):
        raise RuntimeError("Pytest's temporary-directory helper changed; review symlink suppression before running")
    with patch.object(_pytest.pathlib, "_force_symlink", lambda *args, **kwargs: None):
        return pytest.main(sys.argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())
