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

## Bounded vocal learning-rate comparison

`configs/vocals-lr-ablation.json` continues the released 1024-point vocal EMA
for 8,000 updates with fresh Adam and EMA. It matches `vocals-finetune.json`
except for a tenfold learning-rate schedule: 3e-4 to 3e-5, with the same
100-update warmup, crop addresses, batch sizes, augmentation and objectives.
Drums and bass stay fixed. Initialize with `--checkpoint`, `--sha256`, and
`--role ema`; use `--resume` only for exact recovery within this stage.

The research comparison retains complete checkpoints every 1,000 updates and
uses a fixed training/validation crop panel to inspect the learning curve.
Only the final 8,000-update EMA is eligible for the development decision.
Requirements remain at least +0.10 dB each for vocals and physical residual
Other, positive paired track medians, and the existing silence/preservation
limits. The configuration is an experiment recipe; quality, listening and
export qualification are recorded separately when completed.


## Source-view weight experiment

`configs/vocals-auxiliary-ablation.json` tests an outer `auxiliary_weight` of
1.0 against the completed 0.1 higher-rate continuation. It starts from the same
released 40,000-update vocal EMA with fresh Adam and EMA, and uses the same
8,000 updates, data addresses, augmentation, 3e-4 to 3e-5 schedule and original
1024-point architecture. Only the coefficient changes. The existing sixteen
ordinary examples and two full-context source views remain identical. The
instrumental/vocals-only multipliers stay `[1, 0.25]`, with their original joint
denominators. Ordinary reconstruction, primary SDR and absence weights remain
unchanged. This tests whether stronger controlled-view supervision improves
preservation and leakage while retaining separation gains; improvement is not
assumed. It is an experimental recipe, not a replacement for released weights.

The coefficient is validated, forwarded to complete-group gradient replay,
recorded in each update policy, and saved in the checkpoint configuration.
Resuming with a different value fails configuration authentication. Omitting
it retains 0.1 and the configuration identity of existing checkpoints.

Retain complete checkpoints every 1,000 updates and score the fixed training
and validation crop panels. Only the final 8,000-update EMA is eligible for the
full development decision. Vocals and physical residual Other must each gain
at least 0.10 dB, both paired track medians must improve, and all existing
silence and preservation checks must pass. Drums and bass stay frozen. There
is no automatic extension, intermediate checkpoint selection, or test-set use.

## Zero-auxiliary diagnostic

`configs/vocals-zero-auxiliary.json` sets the outer auxiliary coefficient to
0.0. It is otherwise identical to the higher-rate 0.1 continuation: the same
released 40,000-update vocal EMA initialization, fresh Adam/EMA, 8,000 updates,
data addresses, augmentation and 3e-4 to 3e-5 learning-rate schedule. The runtime
still constructs the same ordinary and auxiliary groups; the auxiliary scalar
and its gradient contribution are zero. The ordinary absence penalty remains.

This diagnostic measures separation with the auxiliary contribution removed.
It is not a released model or a mathematical upper bound on alternative
objectives. All nine development checks are still measured, including leakage
and preservation; their outcomes are not assumed in advance. Retain the eight
complete milestone checkpoints and fixed crop curves, and evaluate only the
final EMA on the full development panel. There is no automatic extension or
promotion of this diagnostic arm.


## Optional drums analysis extension

`SpecialistSeparator(source="drums", feature_n_fft=4096)` adds trailing Hann
log-magnitude features while retaining the 1024-point complex carrier and the
existing 256-channel waveform branch. It adds 393,408 parameters and dense MACs
per hop. With the released bass and vocals, the combined model has 8,259,980
parameters and 13,646,080 dense MACs per hop; FFT and elementwise work are excluded.

Drums audio history grows from 896 to 3,968 samples. The specialist still has five
states and the combined model has thirteen; hosts must read the exported shapes.
The 128-sample hop, 128-sample graph alignment and intended 256-sample total delay
are unchanged. FP32 standalone and combined export, checkpoint loading and the
portable native worker support the extension. Physical Mac runtime and model
quality require separate qualification.

Use `configs/drums-long-analysis-finetune.json` to initialize a fresh stage from a
released drums EMA. The trainer copies all existing tensors and initializes the
additional feature projections to zero, preserving the initial audio. The
matching control is `configs/drums-control-finetune.json`. Both create fresh Adam
and EMA; exact `--resume` restores the complete saved architecture and recipe.
See [the training comparison](training.md#drums-longer-analysis-comparison).

## Experimental execution optimization

The [deployment optimization tools](deployment-optimization.md) create separate
layout-specialized FP32, fused FP32 and selectively quantized candidates from the original independent
export. They preserve the training weights and 13-state interface. Numerical,
separation-quality and physical-host measurements remain required before any
candidate replaces a release.

`independent_export.export_fp32(model, new_path, batch_carrier_analysis=True)`
also exports an optional FP32 candidate that batches the three short carrier
FFTs and feature operations. Each stem retains its own history and registered
window, including unequal nonzero public states. This reduces three carrier DFT
dispatches to one, while preserving the amount of FFT arithmetic, all learned
parameters, the complete public state interface, residual formula and latency.
The default export remains separate analysis. Batched reductions can change
FP32 rounding; existing waveform/state tolerances still apply.

The candidate uses the same native model, training, objectives, data pipeline,
optimizer/EMA recovery and evaluation code supplied in this package. It does
not require retraining or change checkpoints. Export tests compare independent
PyTorch audio and states through streaming, resets and EOF, check independent
history/window inputs, and verify learned initializer bytes. It remains an
execution experiment until profiling and sustained physical-Mac measurements
show a benefit and the saved deployment graph passes its quality checks.
