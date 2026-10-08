# StemgenRT-5.8 training, evaluation and export

StemgenRT-5.8 has fixed audio geometry: stereo 44.1 kHz input, 1024-sample
analysis, 256-sample synthesis, a 128-sample hop and eight streaming states.
The model uses 32 attention frames. Its fixed geometry is recorded in model,
checkpoint and export metadata.
Install the appropriate PyTorch 2.8.0 build, then `pip install -e '.[training,onnx,test]'`.
The export dependencies are pinned because integer lowering verifies a specific
ONNX node inventory.
Training also needs `ffmpeg` on `PATH`, compiled with the `rubberband` filter.
Check it with `ffmpeg -hide_banner -h filter=rubberband`.

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

## Current recipe

`configs/current-training.json` contains the recipe that produced the frozen teacher-assisted EMA research baseline:

| Setting | Value |
| --- | --- |
| Context | 88,064 warmup + 176,512 scored samples |
| Batch | 16 independent addressed crops |
| Ordinary / auxiliary microbatch | 16 / 2 |
| Root weights | 0.5 MUSDB18-HQ / 0.5 prepared MoisesDB |
| Vocal activity probability | 0.85 |
| Pitch/tempo selection | 20%; expanded 300,672-sample context |
| Pitch / tempo | ±2 semitones; normal 5%, clamped to ±12% |
| Optimizer | Adam, one update after the ordinary and auxiliary groups |
| Learning rate | 100-update warmup to 3e-5; cosine to 3e-6 at update 2,000 |
| Gradient clipping / EMA | 5.0 / 0.995 |
| Precision | BF16 learned operations; FP32 synthesis, parameters and public states |
| Track sampling | Uniform within each corpus |
| Ordinary primary SDR coefficient | 0.4 |
| Teacher coefficient | 1.0, ordinary ground-truth-active windows only |
| Streaming geometry | 32 attention frames; eight states; no past filter |
| First absolute sample address | 4,132,000 |

This recipe starts from the verified parent EMA with fresh Adam and EMA
state. It records the 2,000-update schedule and original data address; reproducing
the sequence also requires the same ordered 301-track inventory and parent.
Starting it from the best endpoint would be a new continuation, not reproduction.
The generic `TrainingConfig` starts at address zero with teacher supervision
disabled; the supplied configuration enables the pinned teacher described below.
Portable recovery checkpoints are saved every 50 updates, retaining optimizer,
EMA, RNG and the next absolute sample address.

The selected source checkpoint scores **4.564402 dB** on the unchanged development
panel. The separately evaluated v0.6.2 deployment graph scores **4.564148 dB**
on that panel; neither score establishes a fresh held-out result or host
performance. The research baseline and historical v0.6.1 product baseline
remain frozen; see [provenance](provenance.md).

The auxiliary source views retain their original weights and whole-group
denominators. Source remixing is addressed in groups of 16. Warmup initializes
state without retaining its gradient graph; the scored region uses continuous
state and the model's physical delay. The selected recipe requires substantial
GPU memory; CPU checks do not qualify its production resource use.

Use a copied configuration to select different roots or a new sample address.
Root insertion order is part of sampling identity, and `data_start` must be a
multiple of 16. A small CPU experiment needs `device="cpu"`, `precision="fp32"`
and typically `workers=0`; the trainer still requires the full crop and batch
geometry. Reducing the stopping point does not shorten the configured schedule.

### Objective

The default `extra_ordinary_primary_sdr_weight=0.2` gives a total ordinary
primary negative-SDR coefficient of **0.4** (base 0.2 plus additional 0.2).
The absence coefficient is 0.1 and the relative raw-head anchor is 0.01.
Auxiliary source views retain their joint reduction and weights. The saved
configuration includes these choices; exact resume requires the same values.

For source-specific fine-tuning, set `target_source` in a copied training
configuration to `"drums"`, `"bass"`, `"vocals"`, or `"other"`. Omit it or use
`null` for the default joint objective. Source selection applies to waveform,
complex-STFT, raw-output anchors, active-window SDR, absence penalties, and the
optional teacher term. Each component uses the selected source's own reduction;
there is no division by four. The auxiliary instrumental and vocals-only inputs
retain their original weights and complete-group denominators.

This changes the training objective while retaining the four-output model.
Its internal source normalization and consistency correction couple the heads,
so a loss on one output can still update other internal heads. A specialist
therefore does not imply a cheaper one-output graph. Evaluate the selected
stem, leakage, and any residual stem formed from it before using the result.
The selected source is saved in the checkpoint configuration and cannot change
during exact resume; initialize a fresh run to change objectives.

For independently designed one-output bass and drums models, use
`model_family="specialist"` and the explicit configurations described in
[independent stems](independent-stems.md). Their auxiliary views remove or
isolate the named target. Their model, complete recovery and export interfaces
are separate from the historical four-output specialization described above.

### Teacher-assisted baseline

`configs/current-training.json` uses `teacher_coefficient=1.0` alongside the
ordinary ground-truth loss and auxiliary source views. Its saved EMA is the
frozen 4.564402 dB research baseline.

Install the optional training dependency and obtain the pinned teacher:

```bash
python -m pip install -e '.[training,teacher]'
mkdir -p models
curl --fail --location \
  https://dl.fbaipublicfiles.com/demucs/mdx_final/7d865c68-3d5dd56b.th \
  --output models/7d865c68-3d5dd56b.th
```

The provider verifies the checkpoint's 167,918,783 bytes and SHA-256
`3d5dd56b5bc986f136dff98655ded22b2b033f465ccec7a28640a6b15fd71ed6`,
the loaded model state, and the pinned Demucs source files before use. The
teacher extra pins Demucs commit `e976d93ecc3865e5757426930257e200846a520a`
and Julius 0.2.7. Downloaded teacher weights remain separate from this package.

Use the training command below. Set `teacher_checkpoint` in a copied JSON
config to use another local location for the same verified weights.
The CLI sets one CPU thread; library callers must call
`torch.set_num_threads(1)` before enabling the teacher.

The teacher runs on CPU in FP32, using each final augmented six-second mixture,
including the student's warmup context. It uses official per-context
normalization, no random shifts, and no split inference. Drums, Bass and Vocals
use the teacher's native heads; Other is mixture minus those three stems. Only
the scored suffix contributes to the student's teacher loss.

The additional waveform L1 term applies to ground-truth-active complete
one-second windows. Its scale is the maximum of ground-truth RMS, 0.1 times
mixture RMS, and 0.001; reduction uses full-batch active-window counts per stem.
Silent ground-truth windows, the partial final window, and auxiliary views
receive no direct teacher term. Gradients through residual Other can still
affect the native heads, so leakage must be checked after training.

Teacher construction preserves Python, NumPy and CPU PyTorch random state.
Portable checkpoints retain the coefficient, teacher identity and supervision
policy; exact resume rejects changes to them. The checkpoint file's local path
is excluded from that identity. A zero coefficient preserves the baseline
training path without loading teacher weights or Demucs. The teacher is never
included in streaming inference or export; it adds training work and memory,
and does not change the student's architecture or algorithmic delay.

## Start and resume

The released integer ONNX graph is sufficient for inference. It cannot recover
the original FP32 training weights. Supply a native current-model checkpoint
and its independently obtained SHA-256 to initialize a new Adam/EMA run:

```bash
python train_streaming.py --config configs/current-training.json \
  --manifest data/train.json --validation-manifest data/valid.json \
  --checkpoint models/current-ema.pt --sha256 EXPECTED_CHECKPOINT_SHA256 \
  --role ema --output runs/first
```

Use `--role raw` for a raw native checkpoint. Use `--scratch` in place of the
checkpoint arguments to explicitly start untrained. `--stop-after 50` ends at
update 50 while retaining the original 2,000-update learning-rate schedule.
Every run requires a new output directory. The installed `stemgenrt-train`
command has the same options.

The runner writes config/input identities, per-update metrics, `checkpoint.pt`
and a final `result.json`. Checkpoints include raw weights, Adam, EMA, RNG states,
the completed update and the next absolute sample address. Checkpoint writes
are atomic and use self-contained lossless compression. SIGINT/SIGTERM finishes
the current update and saves its complete endpoint.

Resume into a new directory with the checkpoint digest recorded in `result.json`:

```bash
python train_streaming.py --config configs/current-training.json \
  --manifest data/train.json --validation-manifest data/valid.json \
  --resume runs/first/checkpoint.pt --sha256 EXPECTED_RECOVERY_SHA256 \
  --output runs/resumed
```

Resume restores the optimizer and EMA instead of initializing them again. It
requires matching config, manifest identities, root order, precision, device
family and PyTorch build. Keep the same audio decoding, FFmpeg/Rubber Band and
GPU/software environment too: these external dependencies can change numerical
results and are not all captured by those identity checks. Historical XOR-packed
research recoveries need their archived decoder and parent; see
[provenance](provenance.md).

After an abrupt process or machine failure, the last metric row can be newer
than the last saved checkpoint. Resume from the completed step and data cursor
inside the checkpoint; later unsaved updates must be repeated. Keep the
original schedule horizon and preserve the previous run directory. A missing
`result.json` does not establish that training reached its requested endpoint.

When the previous journal contains completed updates beyond the checkpoint,
add `--replay-journal runs/first/metrics.jsonl --replay-sha256 EXPECTED_JOURNAL_SHA256`
to the resume command. The trainer compares every repeated update, including
losses, gradients, input hashes and weight hashes, before writing that row or
saving its checkpoint. Only elapsed-time and memory measurements are excluded.
A difference stops the continuation. The journal must cover a consecutive
sequence through the unsaved updates; an unterminated final line is ignored
as an uncommitted write, while its bytes remain part of the required digest.
Verification details are recorded in checkpoint metadata and `result.json`.
The original journal and checkpoint stay in their previous run directory.

### Changing the training dataset

Dataset expansion starts a new experiment. Build a new manifest, exclude the
held-out recordings and related versions, and update a copied configuration's
root weights. Initialize with `--checkpoint` and an explicit raw or EMA role;
this starts fresh Adam/EMA. `--resume` requires the original dataset and config
identities and will reject a changed manifest or root order.

For a data-only comparison, keep the starting weight role, model, objective and
optimizer-update budget fixed, and record the sampling weight of new material.
New artists and production conditions add diversity beyond the existing pitch,
tempo and remix augmentations. The current recipe already adds instrumental and
vocals-only supervision on every update. Check unwanted vocal output on
instrumental material alongside preservation of quiet real vocals and Other.
Keep evaluation recordings out of training and retain the existing physical
scoring intervals when comparing results. Changing training data does not
change the deployment graph or its algorithmic latency.

## Export

Export either checkpoint role with its eight-state interface:

```bash
python scripts/export_streaming_model.py \
  --checkpoint runs/first/checkpoint.pt --checkpoint-sha256 EXPECTED_RECOVERY_SHA256 \
  --role ema --variant fp32 --output models/candidate.onnx
```

`--variant integer` applies the maintained 17-product integer deployment
transforms. FP32 exports are compared with the native model; integer exports
use an independently reconstructed integer arithmetic reference. Both compare
audio and carried states before publishing the graph and its adjacent
`.verification.json`. Numerical verification is not separation-quality or host
timing qualification. New exports have their own hashes; they are not claimed
to reproduce the released file byte for byte.

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

```bash
python scripts/evaluate_streaming_model.py --manifest data/evaluation.json \
  --checkpoint runs/first/checkpoint.pt --sha256 EXPECTED_RECOVERY_SHA256 \
  --role ema --output runs/ema-scores.json
python scripts/evaluate_streaming_model.py --manifest data/evaluation.json \
  --onnx models/candidate.onnx --sha256 EXPECTED_ONNX_SHA256 \
  --output runs/onnx-scores.json
```

Rendering starts at sample zero, carries all state continuously, accounts for
the 128-sample output delay and uses actual future audio where available.
Only EOF is zero-flushed. Reports preserve the current per-stem metrics and
aggregation. Silent-only custom panels report unavailable primary scores as
`null`. This evaluator measures offline quality, not real-time callback timing.
