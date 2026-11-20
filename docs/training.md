# Training, recovery and evaluation

The released model contains independently trained Vocals, Bass and Drums.
Install PyTorch 2.8.0 for your CPU/CUDA environment, then
`python -m pip install -e '.[training,onnx,test]'`. Audio augmentation requires
FFmpeg with the `rubberband` filter (`ffmpeg -hide_banner -h filter=rubberband`).

## Audio manifests

Prepare native stereo WAVs without resampling or normalization. Each song
folder contains `mixture.wav`, `drums.wav`, `bass.wav`, `vocals.wav` and
`other.wav`. A root may contain nested song folders. MoisesDB material must
first be prepared in this four-source layout; this tool does not map its
original instrument taxonomy.

Build a held-out inventory first, then exclude it from training:

```bash
python scripts/build_streaming_manifest.py \
  --root heldout=data/valid --split valid --output data/valid.json
python scripts/build_streaming_manifest.py \
  --root musdb18hq_train=data/musdb18hq/train \
  --root moisesdb_train=data/moisesdb/train \
  --weight musdb18hq_train=0.5 --weight moisesdb_train=0.5 \
  --split train --exclude-manifest data/valid.json --output data/train.json
```

The builder records file hashes, frame counts, source order and vocal activity
anchors, with roots relative to the manifest. It rejects duplicate mixture
bytes and excludes exact paths/bytes in held-out manifests. Keep different
encodings or related versions of a recording in the same split yourself.
`--exclude-name` can exclude song folder names explicitly. Training verifies
the audio inventory before starting; a provided validation inventory is also
checked for overlap. It is a split guard, not an automatic evaluation schedule.


## Source-specific recipes

Use `configs/vocals-independent.json`, `configs/bass-specialist.json` and
`configs/drums-specialist.json`. Each specifies its model geometry and target.
The release uses each final 40,000-update parameter EMA.

| Setting | Value |
| --- | --- |
| Input context | 88,064 warmup + 176,512 scored samples |
| Batch | 16 addressed ordinary crops |
| Ordinary / auxiliary microbatch | 4 / 2 |
| Root weights | 0.5 MUSDB18-HQ / 0.5 prepared MoisesDB |
| Sampling | Uniform tracks; vocal-active probability 0.85 |
| Pitch/tempo selection | 20%; expanded 300,672-sample context |
| Pitch / tempo | ±2 semitones; normal 5%, clamped to ±12% |
| Learning rate | 500-update warmup to 3e-4; cosine to 3e-5 at update 40,000 |
| Optimizer / gradient clipping | Adam / 5.0 |
| Parameter EMA | Decay 0.995 |
| Precision / teacher | FP32 / disabled |
| First absolute crop address | 8,000,000 |
| Checkpoint interval | 250 updates |

The sampler's vocal activity bias is retained for all three released recipes.
Reproducing input bytes requires the same ordered, checksum-bound manifests,
root order, seeds, augmentation dependencies and crop addresses. The configs
alone do not provide the licensed corpus. Warmup advances recurrent state with
a detached gradient graph; scored audio retains continuous state and its
physical output alignment.

### Objectives

Waveform, complex-STFT, active-window SDR, raw-output anchors and absence terms
select the model's single target. Source-specific reduction does not divide
that output's loss by four. The default extra ordinary primary-SDR weight 0.2
adds to the base 0.2. Absence coefficient is 0.1; relative raw-head anchor is 0.01.
Auxiliary view 0 removes the target and view 1 keeps only that target, including
the full warmup context. Their weights are `[1,0.25]` with outer coefficient 0.1.
The grouped implementation retains complete-group example/window denominators
and replays output derivatives before one Adam and EMA update.

### Start and resume

```sh
python train_streaming.py --config configs/vocals-independent.json \
  --manifest data/train.json --validation-manifest data/valid.json \
  --scratch --output runs/vocals
```

Run the bass and drums configurations separately. Constructors start untrained.
For a fresh stage from an authenticated native endpoint, replace `--scratch`
with `--checkpoint checkpoint.pt --sha256 CHECKPOINT_SHA256 --role ema`.
This initializes fresh Adam/EMA. To restore the complete original run:

```sh
python train_streaming.py --config configs/vocals-independent.json \
  --manifest data/train.json --validation-manifest data/valid.json \
  --resume runs/vocals/checkpoint.pt --sha256 RECOVERY_SHA256 \
  --output runs/vocals-resumed
```

Every run requires a new output directory. Checkpoints contain raw weights,
Adam, EMA, RNG streams, completed updates, absolute crop cursor, configuration
and input identity. Writes are atomic. SIGINT/SIGTERM completes the current
update and saves it. `--stop-after` retains the original learning-rate horizon.
Exact resume requires the original model/config/data identities, device family
and PyTorch build; preserve decoding, FFmpeg/Rubber Band and GPU environment too.

Following a crash, the checkpoint's completed update and cursor are authoritative.
If committed journal rows extend beyond that checkpoint, add
`--replay-journal runs/vocals/metrics.jsonl --replay-sha256 JOURNAL_SHA256`.
Every repeated update is checked against its losses, gradients, input and weight
hashes before it is committed. A mismatch stops continuation. Keep original
journals/checkpoints intact. A changed dataset or objective requires a fresh
stage, not `--resume`.

## Export

Use [the combined exporter](independent-stems.md#export-the-combined-graph) with
all three checkpoint hashes. Individual `band_export` and `specialist_export`
interfaces remain available. New graphs have their own hashes. Native models,
streaming inference and independent PyTorch fixtures validate numerical behavior;
quality and physical host timing need separate measurements.

## Evaluate

Evaluation uses a separate, explicit excerpt manifest. Paths are relative to
that JSON file. For example:

```json
{
  "schema_version": 1,
  "sample_rate": 44100,
  "source_order": ["drums", "bass", "vocals", "other"],
  "tracks": [{
    "name": "held-out-song",
    "mixture": "valid/song/mixture.wav",
    "sources": {
      "drums": "valid/song/drums.wav",
      "bass": "valid/song/bass.wav",
      "vocals": "valid/song/vocals.wav",
      "other": "valid/song/other.wav"
    },
    "excerpts": [{"start_seconds": 30, "duration_seconds": 15}]
  }]
}
```


```sh
python scripts/evaluate_streaming_model.py --manifest data/evaluation.json \
  --onnx models/independent.onnx --sha256 EXPECTED_ONNX_SHA256 \
  --output runs/onnx-scores.json
```

For native combined evaluation, load each checkpoint with `checkpoint.load_model`
and its digest/EMA role, construct `IndependentStems`, and pass it through
`NativeRenderer` to `evaluate_manifest`. Rendering begins at sample zero and
carries all state through prefixes and gaps. Only true EOF is zero-flushed.
Scores use physically aligned intervals. `source_views` provides continuous
target-absent and target-only leakage/preservation checks.

## Legacy support

`StemgenRT58`, `configs/current-training.json`, teacher supervision and
`scripts/export_streaming_model.py` retain the v0.6.2 eight-state training and
export path. Existing compact, joint and integer interfaces remain for checkpoint
compatibility and regression tests; they are not the 0.7.0 deployment model.


### Bass waveform continuation experiment

`configs/bass-control-finetune.json` and `configs/bass-waveform-finetune.json`
define a matched 8,000-update continuation from the released 40,000-update bass
EMA. Both use fresh Adam and EMA, FP32, the same ordered training examples,
augmentation and ordinary/auxiliary objectives, and a 3e-4 to 3e-5 learning-rate
schedule. The waveform arm adds 128 basis channels to the existing specialist.
The trainer copies the original tensors and initializes the new decoder to zero,
so the initial bass audio is unchanged. Exact resume restores the saved model,
Adam, EMA, data cursor and RNG; it does not repeat the conversion.

The optional bass branch has five streaming states and produces a fourteen-state
combined graph. The released model remains thirteen-state. The candidate system
has 8,018,252 parameters and 13,404,224 dense MACs per 128-sample hop, excluding
FFT and elementwise work. Its intended 256-sample latency is unchanged.
Architecture support and arithmetic counts do not establish improved separation
or M4/M4 Pro runtime performance.

The bounded comparison uses final EMA checkpoints, fixed released drums and
vocals, and physical residual Other. Qualification requires at least +0.1 dB in
bass full-band SDR, bass 20–250 Hz SDR and Other SDR, positive paired median
bass/Other gains, and retained SIR, absence and bass-only preservation limits.
Attribution to the added branch additionally requires +0.05 dB bass SDR over
matched continuation and the declared protection checks. These configurations
are experimental; they do not replace released weights.


### Drums longer-analysis comparison

`configs/drums-control-finetune.json` and
`configs/drums-long-analysis-finetune.json` define matched 8,000-update stages
starting from the same released 40,000-update drums EMA. Both use fresh Adam and
EMA, FP32, the same ordered data and augmentation, unchanged source-view
objectives, and a 3e-4 to 3e-5 cosine learning rate with 100-update warmup. The
candidate adds zero-initialized 4096-point trailing analysis projections. The
existing waveform branch, carrier and output timing are retained.

```sh
python train_streaming.py --config configs/drums-long-analysis-finetune.json \
  --manifest /path/to/train.json --validation-manifest /path/to/valid.json \
  --checkpoint /path/to/drums.pt --sha256 DRUMS_SHA256 --role ema \
  --output /path/to/new-drums-run
```

The bounded research comparison retains complete Adam/EMA/RNG checkpoints every
1,000 updates and scores fixed training and validation crops. Only final EMA
endpoints are eligible for selection. Bass and vocals stay fixed, and Other is
the physical mixture residual. Release gates require at least +0.1 dB each in
drums full-band SDR, drums 20–250 Hz SDR and Other SDR, positive paired drums/Other
track medians, and the existing SIR, absence and drums-only preservation limits.
Attribution to longer analysis additionally requires +0.05 dB drums SDR over
matched continuation with the same protection checks. No test set or automatic
budget extension is used. These recipes do not replace released weights.

When exporting a qualifying final endpoint with the released bass and vocals,
pass `--drums-updates 8000 --drums-parent-updates 40000` to
`scripts/export_independent_model.py`, keeping `--expected-updates 40000` for the
fixed stems. Export validates each stem's ancestry and interface independently.

## Bass continuation with residual Other supervision

`configs/bass-residual-other-finetune.json` starts an 8,000-update stage from
the released Bass EMA. Use the release Drums and Vocals EMA checkpoints as fixed
training inputs. The configuration authenticates their SHA-256 digests. Supply
their paths on both fresh and resumed invocations.

```sh
python train_streaming.py --config configs/bass-residual-other-finetune.json \
  --manifest data/train.json --validation-manifest data/valid.json \
  --checkpoint /path/to/bass.pt --sha256 a75137f77e4c9be4868fc6fe3320cf784410519ff6bf4baeb281342eb99524c7 \
  --drums-checkpoint /path/to/drums.pt --vocals-checkpoint /path/to/vocals.pt \
  --output runs/bass-residual-other
```

The ordinary loss includes the error of the physical residual against the
recorded, augmented Other reference. Each frozen peer processes the same
remixed crop as Bass, with its full warmup, carried state and aligned synthesis
flush. Compute Other as `mixture - ((drums + bass) + vocals)`. For each crop,
divide the mean squared stereo Other error by the scored mixture's mean square,
floored at `1e-6`. Average the 16 crop losses, multiply by
`7.2794127729967615`, and add to the ordinary Bass loss before grouped gradient
replay. Only Bass takes an optimizer step. The existing source-absent and
source-only views use their original objective.

The coefficient is 0.2 times the median ratio of the ordinary Bass parameter
gradient norm to the unweighted Other gradient norm at initialization. The
calibration used eight complete training groups with start addresses
`7993856 + 272*i` for `i=0..7`, the control augmentation, and zero optimizer
updates. The matched control is `configs/bass-control-finetune.json`.
Both use the continuation addresses 8,640,000 through 8,767,999.

Score the final EMA on the 14-track development panel with released Drums and
Vocals. Require at least +0.10 dB over the release for Bass, 20–250 Hz Bass,
and Other, positive paired Bass and Other medians, and higher Other than the
matched continuation. Apply the Bass SIR, source-absent leakage and solo Bass
preservation checks. The standard combined exporter accepts the final Bass
checkpoint with `--bass-updates 8000 --bass-parent-updates 40000`.

## Bass continuation with release-relative leakage and preservation penalties

`configs/bass-release-margin-finetune.json` adds one term to the residual
Other recipe. The released Bass EMA provides a reference output on the same
augmented input, including warmup and synthesis alignment. Supply its path
on both fresh and resumed runs. The example starts fresh Adam and EMA from
the released Bass weights.

```sh
python train_streaming.py --config configs/bass-release-margin-finetune.json \
  --manifest data/train.json --validation-manifest data/valid.json \
  --checkpoint /path/to/bass.pt --sha256 a75137f77e4c9be4868fc6fe3320cf784410519ff6bf4baeb281342eb99524c7 --role ema \
  --drums-checkpoint /path/to/drums.pt --vocals-checkpoint /path/to/vocals.pt \
  --bass-release-checkpoint /path/to/bass.pt --output /path/to/new-run
```

| Condition during training | Per-window degradation relative to the released Bass EMA | Tolerance |
|---|---|---:|
| Ordinary mix, Bass reference at or below −50 dBFS | Rise in Bass output dBFS | 0.50 dB |
| Bass removed, input above −50 dBFS | Rise in Bass output/input dB | 0.50 dB |
| Bass alone, reference above −50 dBFS | Loss of scale-dependent Bass SDR | 0.10 dB |

For each condition, average the positive excess above tolerance over all
eligible one-second stereo windows. An empty condition contributes zero.
The three means receive equal weight. The coefficient of their average is
`0.003270580966293747`. It was set on eight preselected training groups by
measuring gradients at the completed residual-Other EMA. The weight is 0.20
times the median ratio of the full existing objective's gradient norm to the
unit-weight penalty's gradient norm. All eight groups had a nonzero penalty
gradient. Only the student Bass model receives updates.

The matched comparison uses the same 8,000 updates and addressed crops as
`configs/bass-residual-other-finetune.json`. Each checkpoint authenticates the
frozen Bass reference and loss specification. The final EMA is evaluated on
ordinary mixes, Bass-removed inputs and isolated Bass with the original
development criteria. The frozen reference runs only during training.

## Released inventory and reconstruction limits

The [model card](../MODEL_CARD.md) identifies the three missing MUSDB songs,
MoisesDB, the duplicate exclusion, and the separate baseline lineage. The
[portable inventory](../configs/released-training-inventory.json) preserves all
301 selected recordings and prepared-file identities. The [reference Moises
converter](benchmark.md) is new code with an explicit conversion policy; the
historical raw conversion script is not available, so it is not a claim of
byte-identical data reproduction. Native training checkpoints are not distributed.

### Interpolating a Bass continuation with its parent

The released Bass specialist is the 40,000-update EMA. The
`configs/bass-release-margin-finetune.json` recipe initializes from that EMA
and trains for another 8,000 updates with fixed Drums and Vocals. Keep both
authenticated checkpoints to reproduce an interpolation of their Bass parameters.
The continuation checkpoint also contains its matching raw, Adam, EMA and RNG
recovery state.

With the ordinary independent export arguments, use the continuation for
`--bass-checkpoint` and `--bass-sha256`. Set `--expected-updates 40000`,
`--bass-updates 8000` and `--bass-parent-updates 40000`. Add
`--bass-blend-parent` with the release checkpoint path,
`--bass-blend-parent-sha256` with its digest and `--bass-trained-fraction` with
the selected fraction between 0 and 1. Zero reproduces the parent and one
reproduces the continuation. The exporter checks the continuation's
initialization fingerprint, interpolates each learned tensor in FP64 on CPU,
and rounds once to FP32. Both endpoints must have identical fixed buffers.

The graph runs a single Bass model. Its manifest records both source
checkpoints, the fraction and the reconstructed state fingerprint. The
interpolation performs zero further optimizer updates. Drums, Vocals and the
physical residual calculation use the same training and deployment definitions
as the source endpoints.

For training-only calibration, pass the training tracks, the released combined
model and a mapping of fractions to authenticated Bass models to
`stemgenrt.bass_interpolation_eval.score_bass_interpolation_track`. Include
the released model at fraction zero. The scorer streams each track from its
origin on the ordinary mix, isolated Bass and the mix with Bass removed.
Aggregate each fraction's tracks with `summarize_bass_tracks` and select
with `select_bass_fraction`. In this experiment the candidates are 0.125,
0.25, 0.5 and 0.75. SIR, solo projection level and solo SDR tolerate a loss
of 0.10 dB. Both leakage measures tolerate a rise of 0.50 dB. Among fractions
within all five limits, selection maximizes the smallest gain in Bass, Other
and low-Bass SDR. If every fraction exceeds a limit, the smallest worst excess
relative to its tolerance takes priority. Exact ties favor the smaller fraction.
Score only the selected setting on development.

### Bass-only preservation weight

`configs/bass-solo-margin-finetune.json` repeats the 8,000-update
release-margin continuation with `release_margin_solo_multiplier` set to 4.
Run the same training entry point with the same Bass release and fixed Drums
and Vocals checkpoints used by `configs/bass-release-margin-finetune.json`.
Each condition averages its excess over all its eligible one-second windows.
The objective sums the ordinary quiet excess, the Bass-removed excess and
four times the Bass-only SDR excess, then divides by three and applies
`release_margin_weight`. This preserves the absolute coefficients on both
leakage terms. A multiplier of one reproduces the original recipe.
The optimizer journal records the multiplier and each condition's unweighted
mean excess. Training snapshots bind the multiplier to the objective for
recovery and EMA evaluation.

### Lower-rate Bass release-margin continuation

`configs/bass-margin-low-lr-finetune.json` starts from the released
40,000-update Bass EMA and uses the equal-weight Bass release-margin objective.
Its 100-update warmup reaches a peak learning rate of 0.0001, followed by the
same cosine decay to 0.00001 at update 8,000. Both rates are one third of
those in `configs/bass-release-margin-finetune.json`. Use the same
training entry point, authenticated Bass release and fixed Drums and Vocals
checkpoints. The sampling addresses, optimizer settings other than the
learning rate, and EMA decay are identical. Recovery uses the configuration
stored with this training run.

### Sampling difficult isolated Bass

`configs/bass-hard-solo-finetune.json` uses the lower-rate continuation and
changes how the trainer chooses its single Bass-only source view. Odd updates
use the original batch index 15. On even updates the frozen released Bass EMA
renders all 16 isolated, augmented Bass crops from the beginning of their
warmup. Each crop needs at least two complete active one-second windows.
The trainer chooses the lowest mean bounded SDR below 50 dB. It prefers
index 15 in a tie, then the lowest index. If no crop qualifies, it uses 15.

The trainer uses the selected crop for both the existing auxiliary Bass-only
loss and its release margin. The 16 ordinary mixtures, Bass-removed view at
index 14, per-view weights and number of optimizer updates use the same
recipe. The journal records the selected index, window counts and release
scores for each screened group. Checkpoint recovery validates the sampler and
its release fingerprint. Inference uses the trained Bass specialist directly.

### Repeat the matched Bass sampling experiment

`configs/bass-low-lr-repeat.json` and `configs/bass-hard-solo-repeat.json`
start at crop address 8,768,000, immediately after the original 8,000-update
experiments. Both consume 128,000 new crop addresses from the same training
recordings. Use the released Bass EMA as the starting checkpoint and release
reference for each run, with the same frozen Drums and Vocals peers described
above. Give each run a separate output directory and a fresh optimizer.

The first recipe uses the fixed Bass-only view. The second uses the alternating
hard-solo sampler. Compare their final EMAs on the same evaluation excerpts
at the original levels. The journal records the crop addresses, audio hashes
before and after remixing, learning rates, and selected views.
