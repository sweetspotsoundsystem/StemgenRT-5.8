# Contributing

The released StemgenRT-5.8 model uses eight streaming states, 32 attention frames,
1024-sample analysis, 256-sample synthesis and 128-sample hops at 44.1 kHz.
Keep model changes together with their training, data, loss, checkpoint and
export support. Changes to parameter names, state shapes or checkpoint formats
must account for existing model files.

This research branch also contains experimental compact and banded separators.
Keep their training, recovery and export interfaces separate from the released
model; see [compact research](docs/compact-research.md). Synthetic CPU checks
do not qualify those models for a plugin release.

## Development setup

Use Python 3.12 or later. Install PyTorch 2.8.0 for your CPU or CUDA environment,
then install the development dependencies:

```bash
python -m pip install -e '.[streaming,training,onnx,test]' build
python scripts/download_streaming_model.py
```

Data augmentation tests require FFmpeg with the `rubberband` filter. Check it
with `ffmpeg -hide_banner -h filter=rubberband`.

## Checks

Run the CPU suite and build both distributions:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m pytest -q
python -m build
```

Tests cover streaming state and audio alignment, training objectives and
gradients, deterministic data augmentation, complete Adam/EMA/RNG recovery,
ONNX export, and waveform agreement with independent PyTorch fixtures. Update
the relevant tests when changing these behaviors. A passing CPU suite does not
measure separation quality or real-time plugin performance.

Keep checkpoints, datasets, generated model files and local experiment records
out of source distributions. For a released model update, update the immutable
download URL, checksum and matching waveform fixtures together.

Describe the user-visible change and validation in the pull request. Use a
squash merge for a focused release change. Python package versions are separate
from the model's 5.8 ms latency suffix; version 0.6.2 matches plugin v0.6.2.
