# StemgenRT-5.8

Model, training and inference code for [StemgenRT](https://github.com/sweetspotsoundsystem/stemgen-rt).
Version **0.7.0** uses three independently trained models for **Drums, Bass and
Vocals**, with **Other = mixture − Drums − Bass − Vocals**. Each model owns its
parameters and recurrent state. The released graph combines all three in one
self-contained FP32 ONNX file.

Stereo 44.1 kHz audio is processed in 128-sample hops with 13 persistent states.
The graph delays audio by 128 samples. The plug-in's worker reserve adds 128
samples with a 128-sample host buffer: **256 samples, or 5.805 ms**, excluding
hardware latency. Larger host buffers require additional reported reserve.
The repository's 5.8 suffix describes this latency setting, not a version number.

## Inference

```sh
python -m pip install -e '.[streaming]'
python scripts/download_streaming_model.py
python examples/separate_streaming.py --help
```

```python
import numpy as np
from stemgenrt import StreamingSeparator

separator = StreamingSeparator("models/independent.onnx")
audio = np.zeros((2, 44100), dtype=np.float32)
stems = separator.separate(audio)  # [4, 2, samples], aligned to the input
```

The download verifies the immutable graph's size and SHA-256. ONNX Runtime
1.26.0 uses one CPU thread with spinning and KleidiAI disabled. The synchronous
Python API allocates memory; audio applications should use an inference worker.
The plug-in provides that integration. See the [model interface](models/README.md).

## Training and export

Matching model definitions, objectives, data augmentation, Adam/EMA recovery,
evaluation and exporters are included. Install the appropriate PyTorch 2.8.0
build and then `python -m pip install -e '.[training,onnx,test]'`.

Train each source separately with `configs/vocals-independent.json`,
`configs/bass-specialist.json` and `configs/drums-specialist.json`. Each released
endpoint uses the parameter EMA after 40,000 updates. The exporter combines the
three authenticated checkpoints without sharing weights or redistributing
errors into the learned stems.

See [training and recovery](docs/training.md) and
[architecture and export](docs/independent-stems.md). Datasets and native training
checkpoints are supplied separately; the ONNX file does not contain optimizer,
EMA history or data-cursor state needed to resume training.

## Compatibility and limitations

The plug-in remains named **StemgenRT** and retains its existing host identity.
Version 0.7.0 replaces its previous joint model. Existing v0.6.2 Python inference
is available with `--release v0.6.2` and that model's explicit checksum.
`StemgenRT58` remains the legacy eight-state native class; use `IndependentStems`
with `BandSeparator` and `SpecialistSeparator` for the new architecture.

The listening comparison favored cleaner vocals, better vocal silence and less
vocal bleed in drums and bass. Other can contain stronger residual errors.
Physical M4/M4 Pro sustained real-time performance remains unmeasured for this
graph; the latency figure is not a CPU deadline guarantee.

This is a divergent fork of [Phil Wang's HS-TasNet implementation](https://github.com/lucidrains/HS-TasNet),
with a different model and Python API. See [provenance and migration](docs/provenance.md).
