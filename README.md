# StemgenRT-5.8

This is a divergent fork of [Phil Wang's HS-TasNet implementation](https://github.com/lucidrains/HS-TasNet),
developed for the [StemgenRT audio plugin](https://github.com/sweetspotsoundsystem/stemgen-rt).
It has a different model and a breaking Python API: use `stemgenrt` and
`StemgenRT58` instead of `hs_tasnet`. Use the upstream repository if you need
the original HS-TasNet implementation. See [migration and provenance](docs/provenance.md).

Stereo streaming music separation into **drums, bass, vocals and other**.
The current model combines spectral and waveform branches, causal attention,
and recurrent branch memories. It uses **1024-sample analysis, 256-sample
synthesis, 128-sample hops and eight explicit FP32 states** at 44.1 kHz.
The **5.8** suffix identifies the latency variant: 128 samples of graph delay
plus StemgenRT's 128-sample worker queue give 256 samples at 44.1 kHz, rounded
to **5.8 ms of graph-plus-host algorithmic latency**, excluding audio-device
latency. Model revisions and Python package versions are tracked separately
from this latency suffix. Python package version **0.6.2** targets the model
released with [StemgenRT v0.6.2](https://github.com/sweetspotsoundsystem/stemgen-rt/releases/tag/v0.6.2).

## Run the released model

```bash
python -m pip install -e '.[streaming]'
python scripts/download_streaming_model.py
python examples/separate_streaming.py --help
```

```python
import numpy as np
from stemgenrt import StreamingSeparator

separator = StreamingSeparator("models/hop128.onnx")
audio = np.zeros((2, 44100), dtype=np.float32)
stems = separator.separate(audio)  # [4, 2, samples], aligned to the input
```

The download is pinned by size and SHA-256 to the integer graph shared with
[StemgenRT](https://github.com/sweetspotsoundsystem/stemgen-rt). ONNX Runtime
1.26.0 runs on one CPU thread with KleidiAI disabled, matching that release's
runtime settings. This Python API allocates memory and runs synchronously; use
it from a worker for playback integration. See [model interface](models/README.md).

## Native model, training and export

Install the desired CPU or CUDA build of PyTorch 2.8.0 first, then:

```bash
python -m pip install -e '.[training,onnx,test]'
```

```python
import torch
from stemgenrt import StemgenRT58, render_scored_context

model = StemgenRT58()  # Untrained weights; this does not load the release.
audio = torch.randn(1, 2, 768) * .02
scored = render_scored_context(model, audio, warmup_samples=256, carry_state=True)
scored.raw.square().mean().backward()
```

The native default uses 32 frames of causal attention and eight streaming states.
The frozen research baseline is the teacher-assisted EMA checkpoint at **4.564402
dB full-band SDR** on the fixed 14-track, 28-excerpt development panel. Its
training recipe is retained in `configs/current-training.json`: BF16 learned
operations, uniform track sampling, ordinary/auxiliary microbatches of 16/2,
and teacher coefficient 1.0. The teacher is absent from inference.

The released ONNX download is pinned to StemgenRT v0.6.2, which deploys the
frozen research checkpoint and scores **4.564148 dB** on the same panel.
The v0.6.1 product baseline remains the historical comparison and rollback
reference; see [provenance](docs/provenance.md).

The maintained package includes deterministic crop/pitch/remix augmentation,
the whole-group weighted source-view objective, Adam and EMA, lossless complete
recovery, native/ONNX evaluation and verified export. Read the
[training and evaluation guide](docs/training.md) for checkpoint requirements,
portable manifests and commands. Native FP32 weights cannot be reconstructed
losslessly from the released integer graph; provide a native checkpoint with
its SHA-256 or explicitly start from scratch.

## Supported source

| Module | Responsibility |
| --- | --- |
| `stemgenrt.model` | Current native model, states and detached context |
| `stemgenrt.data` | Portable manifests and deterministic training augmentation |
| `stemgenrt.losses` | Whole-group objectives and one Adam/EMA update |
| `stemgenrt.checkpoint` | Checkpoint verification and complete recovery |
| `stemgenrt.trainer` | Portable finite training and resume |
| `stemgenrt.evaluation` | Physical alignment and per-stem metrics |
| `stemgenrt.export` | Current fixed-geometry ONNX export |
| `stemgenrt.streaming` | Released and checksum-pinned custom ONNX inference |

`StemgenRT58` supports the current eight-state architecture. The repository also
includes independent single-output research models with separate state and
checkpoint interfaces: [causal vocal models](docs/compact-research.md) and
[independent bass/drums specialists](docs/independent-stems.md). Their matching
training, source-specific objectives, complete recovery, combined evaluation
and FP32 export are included. Candidate quality and measured native M4/M4 Pro
runtime remain required before deployment.

Earlier released architectures and draft papers remain in git history.

The implementation builds on [HS-TasNet](https://arxiv.org/abs/2402.17701) and
[Phil Wang's implementation](https://github.com/lucidrains/hs-tasnet).
