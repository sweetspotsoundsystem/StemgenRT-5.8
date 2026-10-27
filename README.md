# StemgenRT-5.8

Model, training and inference code for [StemgenRT](https://github.com/sweetspotsoundsystem/stemgen-rt).
Version **0.8.0** retains the three independently trained models introduced in
0.7.0 for **Drums, Bass and
Vocals**, with **Other = mixture − Drums − Bass − Vocals**. Each model owns its
parameters and recurrent state. The released graph combines all three in one
self-contained FP32 ONNX file. This checkout prepares the 0.8.0 software release;
the model weights and immutable download identity remain unchanged. Exact graph
partitions support the plugin's three-worker deployment.

Stereo 44.1 kHz audio is processed in 128-sample hops with 13 persistent states.
The model's synthesis-frame algorithmic latency is **256 samples / 5.805 ms**.
Its graph alignment is 128 samples. Separately, the plug-in's worker reserve adds
128 samples at a 128-sample host buffer, reporting **256 samples / 5.805 ms PDC**.
The asynchronous plugin is 384 samples / 8.707 ms under the synthesis-frame
convention. These figures exclude device I/O; see the [model card](MODEL_CARD.md).

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
[architecture and export](docs/independent-stems.md). Acquire datasets separately under their own terms. Native training
checkpoints are not currently distributed; the ONNX file does not contain optimizer,
EMA history or data-cursor state needed to resume training.

## Compatibility and limitations

The plug-in remains named **StemgenRT** and retains its existing host identity.
Version 0.7.0 replaces its previous joint model. Legacy v0.6.x artifacts have unresolved additional training-provenance questions;
see the [weights notice](WEIGHTS-LICENSE.md) before using those baselines.
`StemgenRT58` remains the legacy eight-state native class; use `IndependentStems`
with `BandSeparator` and `SpecialistSeparator` for the new architecture.

The listening comparison favored cleaner vocals, better vocal silence and less
vocal bleed in drums and bass. Other can contain stronger residual errors.
The matching three-worker plugin passed balanced M4 sustained standalone
playback (60 measured minutes per mode) and an M1 Pro quick comparison.
The Ryzen 5 5500 Windows comparison improved substantially but retained
underruns at both tested host buffers. M4 Ableton stability is a user report;
other CPUs, DAWs and multiple-instance loads are not thereby qualified.
See the plugin's [runtime evidence](https://github.com/sweetspotsoundsystem/stemgen-rt/blob/main/docs/parallel-workers.md).
These results do not describe the synchronous Python API, and the latency
figure is not a CPU deadline guarantee.

This is a divergent fork of [Phil Wang's HS-TasNet implementation](https://github.com/lucidrains/HS-TasNet),
with a different model and Python API. See [provenance and migration](docs/provenance.md).

## Results, credit and licences

See the [model card and frozen results](MODEL_CARD.md),
[full-song benchmark and preparation tools](docs/benchmark.md), and
[CITATION.cff](CITATION.cff). The manuscript remains unpublished.
Code is MIT; [model weights have separate terms](WEIGHTS-LICENSE.md).
Phil Wang is credited for the upstream HS-TasNet implementation; the original
copyright notice remains in LICENSE.
