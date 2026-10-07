# Independent bass and drums research models

`stemgenrt.specialist.SpecialistSeparator` provides one stereo bass or drums
estimate. These experimental architectures accompany the single-output vocal
model in `stemgenrt.banded`; they do not change the released plugin or its
eight-state interface. Constructor weights are untrained.

Bass uses 17 carrier bands with finer low-frequency partitions, width 80 local
state, width 160 global state and two causal recurrent layers. A separate
4096-point trailing Hann analysis adds stereo log-magnitude features to the
1024-point complex carrier features. Its window ends at the same received
sample as the carrier. Longer past context adds no future callbacks or synthesis
delay. Each FFT is normalized over stereo channels and bins within its own
current frame. `feature_n_fft=1024` disables the longer feature branch for a
declared technical control.

Drums uses 20 carrier bands, local width 96, global width 192 and two recurrent
layers. A 256-channel bias-free encoder and decoder produce a waveform residual
from the most recent 256 samples, gated by causal global context. Spectral and
waveform synthesis share the same physical output coordinates. A small nonzero
decoder initialization permits gradients through the complete branch.
`waveform_basis=0` disables this branch for a technical control.

| Default component | Parameters | Dense MACs per 128-sample hop |
| --- | ---: | ---: |
| Vocal band model | 2,720,484 | 4,805,568 |
| Bass specialist | 2,114,052 | 3,330,240 |
| Drums specialist | 3,032,036 | 5,116,864 |
| Combined | 7,866,572 | 13,252,672 |

The declared combined ceilings are 9 million parameters and 15 million dense
MACs per hop. Counts exclude FFTs, normalization, nonlinear operations, copies
and bias additions. They do not establish native M4 or M4 Pro performance.
The graph emits audio aligned 128 samples behind its received input. A host's
128-sample hop accumulation gives 256 samples, or 5.805 ms at 44.1 kHz.

## Training and exact recovery

Use `configs/bass-specialist.json` or `configs/drums-specialist.json` with the
normal training CLI and `--scratch`. Both select `model_family="specialist"`,
the named `target_source`, explicit geometry and FP32 training. BF16 and online
teacher supervision are rejected. The configurations declare 40,000 updates,
16 ordinary examples, 4/2 ordinary/auxiliary microbatches, six-second input
contexts, detached warmup, a 500-update warmup to 3e-4, cosine decay to 3e-5 and
parameter EMA decay 0.995. Every checkpoint includes the complete configuration,
raw weights, Adam, EMA, RNG streams and absolute crop cursor.

```bash
python train_streaming.py --config configs/bass-specialist.json \
  --manifest data/train.json --validation-manifest data/valid.json \
  --scratch --output runs/bass-first
```

The configurations retain the comparison experiment's ordinary crop addresses,
vocal-activity sampling probability, pitch/tempo recipe and remix seed. Ordinary
inputs can therefore be checked against its joint control by their recorded
hashes. Keeping the existing vocal-activity sampling bias is intentional for
this comparison; it is not a claim that this is the optimal bass/drums sampler.
Use the same ordered, checksum-bound manifests to reproduce actual input bytes.

For these single-output models, auxiliary view 0 removes the selected target
and view 1 retains only that target, including the entire warmup context.
Targets are then reduced to the model's single output without padding. The
existing whole-group example/window denominators, view weights `[1,0.25]` and
outer coefficient 0.1 are retained. Every microbatch replays its complete-group
output derivatives exactly before one Adam and EMA update. The historical
four-output source-specific objective retains its original vocal auxiliary
views, preserving its recovery recipe.

`stemgenrt-independent-specialist-training-v1` is a separate checkpoint schema.
Use `--resume` and the saved SHA256 in a new output directory for complete
continuation. The source, branch geometry, data identity and schedule cannot
change on resume. Unsaved committed journal updates can be verified with the
existing `--replay-journal` and `--replay-sha256` interface.

## Export and combined evaluation

`stemgenrt.specialist_export.export_fp32(model, new_path)` exports a complete
FP32 one-hop ONNX graph, including both analysis FFTs when enabled, masks,
inverse FFT, overlap synthesis and the drum waveform branch. Bass has four
states: audio history, local hidden state, global hidden state and spectral
numerator tail. Drums with a waveform branch has a fifth waveform numerator
tail. Keep all exported states between callbacks. Export creates a separate
copy and preserves the source model's mode, gradients, weights and RNG.

`stemgenrt.independent.IndependentStems(drums=..., bass=..., vocals=...)`
combines three frozen, genuinely single-output models. Each receives the same
audio and owns independent state. Other is the physical mixture minus the
three estimates. The wrapper rejects shared parameter objects, source-order
mismatches and incompatible timing. It supports `NativeRenderer` evaluation
through partial final hops and EOF. It does not redistribute errors into the
learned stems.

`stemgenrt.source_views.stream_source_views` renders target-only and
target-absent recordings continuously from the track origin. Each view keeps
independent state through unscored prefixes and gaps, with padding only at EOF.
The helper records input hashes, physical alignment and mixture closure.
`score_source_views` reports native leakage levels and target preservation;
silent model outputs remain included whenever the input window is active.

CPU checks cover future independence, all 128 impulse phases, grouped/literal
streaming, detached warmup, finite gradients, exact checkpoint continuation,
named absence/preservation views, combined residual alignment and continuous
ONNX waveform/state agreement. Candidate quality, combined listening and actual
native M4/M4 Pro runtime must be evaluated before deployment. Current vocal
anchor evidence is from a single seed; multi-seed repeatability is not established.
