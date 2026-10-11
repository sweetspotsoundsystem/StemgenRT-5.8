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

## Optional vocal analysis extension

`BandSeparator(feature_n_fft=4096)` adds trailing Hann log-magnitude features
to the single-output vocal model. The 1024-point complex carrier, synthesis,
hop and alignment remain the same. At width 96, the additional projections
contain 393,408 parameters and use 393,408 dense MACs per hop. Vocal audio
history grows from 896 to 3,968 samples; hosts must use the exported shapes.
The combined system retains 13 states and totals 8,259,980 parameters and
13,646,080 dense MACs per hop. FFT overhead is excluded from those counts.

For a new training stage, `configs/vocals-long-analysis.json` selects this
variant; `configs/vocals-finetune.json` uses the original analysis with the
same training schedule. Initializing the long variant from a short-analysis
vocal checkpoint copies every existing tensor and starts the extra projections
at zero. The trainer creates fresh Adam and EMA for both configurations. Exact
`--resume` instead requires the saved architecture and full original recipe.

```sh
python train_streaming.py --config configs/vocals-long-analysis.json \
  --manifest /path/to/train.json --validation-manifest /path/to/valid.json \
  --checkpoint /path/to/vocals.pt --sha256 VOCALS_SHA256 --role ema \
  --output /path/to/new-vocal-run
```

FP32 single-model and combined export support this variant. Integer band
export currently supports only the original 1024-point vocal analysis.
The extension is an optional training capability; its separation quality and
physical Mac runtime have not been qualified for a release.

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

For a vocal fine-tuning endpoint with 8,000 current-stage updates after 40,000
parent updates, add `--vocals-updates 8000 --vocals-parent-updates 40000` while
keeping `--expected-updates 40000` for the unchanged bass/drums endpoints.
The export verifies both stages and records each stem's own update counts.

The original PyTorch models generate reference audio; ONNX outputs are never
used as their own oracle. The [native worker probe](../examples/native_independent/README.md)
also checks every returned state before measuring worker timing on a physical
Mac. The plug-in's complete callback/worker qualification and DAW playback
remain separate tests. The graph's historical export schema is retained.
