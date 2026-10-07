# Joint model training, ONNX export and listening

The experimental joint band model estimates drums, bass, vocals and Other with
four learned complex masks. Its native four outputs use no mixture correction.
The optional residual-Other policy retains its own drums, bass and vocals, then
calculates `Other = mixture - ((drums + bass) + vocals)`.

The completed first-seed 40,000-update EMA endpoint scores 4.837649 dB overall
SDR on the fixed 14-track development panel. Its residual-Other variant scores
4.828170 dB; the frozen parent scores 4.564402 dB. These are development results
from one seed. They do not establish native Mac performance or qualify a plugin
release. A vocal-replacement comparison using parent drums/bass is a separate
system from either complete joint variant.

## Matching training

[configs/band-joint.json](../configs/band-joint.json) is the exact first-seed joint
configuration: width 96 per band, global width 192, two layers, batch 16, FP32,
40,000 updates, detached warmup and parameter EMA. It uses the same deterministic
data, objectives and recovery pipeline described in [training](training.md).
The complete ordered corpus inventory is also required to reproduce the run.

```sh
python train_streaming.py --scratch --config configs/band-joint.json \
  --manifest /path/to/train.json --validation-manifest /path/to/valid.json \
  --output /allocated/new-joint-run
```

The checkpoint retains raw weights, Adam, EMA, RNG, data position, configuration
and provenance. Resume with `--resume checkpoint.pt --sha256 CHECKPOINT_SHA256`
in place of `--scratch`, using the original configuration and manifests.

## Export the authenticated EMA

Install the training and ONNX dependencies, then load the chosen checkpoint with
its known digest. A frozen experiment must also authenticate its completion proof
and checkpoint-selection policy before export.

```python
import json
from pathlib import Path
from stemgenrt.checkpoint import load_model
from stemgenrt.band_export import export_fp32

model = load_model("checkpoint.pt", expected_sha256="CHECKPOINT_SHA256", role="ema")
assert model.sources == ("drums", "bass", "vocals", "other")
report = export_fp32(model, Path("joint-fp32.onnx"))
Path("joint-fp32.json").write_text(json.dumps(report, indent=2) + "\n")
```

The FP32 graph includes its analysis FFT, network, inverse FFT and synthesis.
At the default geometry its inputs are:

| Input | Shape |
| --- | --- |
| `audio_chunk` | `[1, 2, 128]` |
| `audio_history` | `[1, 2, 896]` |
| `local_hidden` | `[2, 20, 96]` |
| `global_hidden` | `[2, 1, 192]` |
| `spectral_numerator_tail` | `[1, 4, 2, 128]` |

All values are float32. Start the states at zero and retain the four returned
`next_*` states between calls. `separated_chunk` has shape `[1, 4, 2, 128]` in
drums/bass/vocals/Other order. It is aligned 128 samples behind the input. The
host's 128-sample accumulation gives the intended 256-sample total delay at
44.1 kHz. This four-state graph requires its own integration; the released
plugin expects a different eight-state model.

## Listen to the exported model

[The joint ONNX example](../examples/separate_joint.py) accepts stereo 44.1 kHz
audio. It maintains continuous state, handles a partial final hop, flushes at
true EOF, and removes graph alignment exactly once. It writes four float32 WAVs
at native levels with no normalization and retains the original frame count.

```sh
python examples/separate_joint.py mixture.wav /allocated/joint-listening \
  --model joint-fp32.onnx --sha256 ONNX_SHA256
```

Add `--residual-other` to hear the physical residual variant. Use a different
output directory for each run; existing output is preserved. Compare all four
stems with the original sources and frozen baseline, using one shared playback
volume. Listen for drum attacks and cymbal decay, bass weight and definition,
vocal detail and leakage, and missing instruments or artifacts in Other.

The [native worker probe](../examples/native_independent/README.md) can include
the authenticated joint graph as a control on actual M4 and M4 Pro machines.
