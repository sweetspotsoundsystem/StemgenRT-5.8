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
