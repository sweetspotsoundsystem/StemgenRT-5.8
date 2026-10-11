# Independent stem architecture and export

StemgenRT 0.7.0 combines three separately trained single-output models through
`IndependentStems(drums=..., bass=..., vocals=...)`. They receive the same input
and share no learned parameters or recurrent state. The wrapper checks source
order, ownership and timing. Other is the physical delayed mixture minus
`((Drums + Bass) + Vocals)`.

All components use stereo 44.1 kHz input, a 1024-point complex spectral carrier,
256-sample synthesis and 128-sample hops. The graph emits the preceding input
hop. Longer trailing analysis context adds past context without adding future
callbacks or synthesis delay.

| Component | Implementation | Local/global widths | Bands | Parameters |
| --- | --- | --- | --- | --- |
| Vocals | `BandSeparator(sources=("vocals",))` | 96 / 192 | 20 | 2,720,484 |
| Bass | `SpecialistSeparator(source="bass")` | 80 / 160 | 17 | 2,114,052 |
| Drums | `SpecialistSeparator(source="drums")` | 96 / 192 | 20 | 3,032,036 |

Each model has two recurrent layers. Vocal projections encode and decode each
band independently, with shared temporal processing and causal global context.
Bass uses finer low-frequency carrier partitions and a separate trailing
4096-point Hann analysis for log-magnitude features. Each FFT normalizes its
own current-frame stereo/bin features. Drums adds a 256-channel waveform
encoder/decoder gated by causal global context, aligned with spectral synthesis.
There is no normalization across output sources.

The total is 7,866,572 parameters and 13,252,672 dense learned MACs per hop.
These counts exclude FFT, normalization, nonlinear operations, copies and bias
additions. They are not measured CPU timings.

## Training and recovery

Use the three configurations described in [training](training.md). The vocal
configuration selects `model_family="banded"` and `target_source="vocals"`;
bass and drums select `model_family="specialist"` and their named source.
All train in FP32 with no teacher. Each target has a single output and its own
absence/preservation auxiliary views, loss reduction, Adam and parameter EMA.
Checkpoints retain all state needed for exact continuation.

## Export the combined graph

Authenticate all three native EMA checkpoints before exporting:

```sh
python scripts/export_independent_model.py --output /path/to/new-export \
  --drums-checkpoint /path/to/drums.pt --drums-sha256 DRUMS_SHA256 \
  --bass-checkpoint /path/to/bass.pt --bass-sha256 BASS_SHA256 \
  --vocals-checkpoint /path/to/vocals.pt --vocals-sha256 VOCALS_SHA256 \
  --expected-updates 40000
```

The output contains `model.onnx`, full architecture/interface metadata,
checkpoint identities, source bindings and independent native PyTorch fixtures.
The graph includes analysis FFTs, recurrent networks, masks, inverse FFTs,
overlap synthesis and residual Other. It has 13 explicit FP32 states; see the
[interface](../models/README.md) for names, shapes, reset and EOF rules.
Export uses copies and preserves source parameters, mode, gradients and RNG.

The original PyTorch models generate reference audio; ONNX outputs are never
used as their own oracle. The [native worker probe](../examples/native_independent/README.md)
also checks every returned state before measuring worker timing on a physical
Mac. The plug-in's complete callback/worker qualification and DAW playback
remain separate tests. The graph's historical export schema is retained.
