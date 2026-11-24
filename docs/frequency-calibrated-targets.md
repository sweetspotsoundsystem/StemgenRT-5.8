# Frequency-calibrated Bass and Other teaching targets

The released Drums and Vocals specialists and an offline teacher can assign the same sound to different stems. Let the discrepancy equal the physical mixture minus released Drums, native teacher Bass, released Vocals and native teacher Other. Add a fitted fraction of this discrepancy to teacher Bass in each of seven bands. Compute target Other by subtracting the resulting three stems from the physical mixture.

The calibration measures both reference errors on ordinary stereo recordings. It minimizes normalized Other error while allowing a 0.35 dB increase in normalized direct-teacher Bass error on the fitting artists. Each corpus and each dual-active artist within a corpus has equal weight. A scalar control fits one coefficient under the same constraint. The bands have boundaries at 85, 170, 340, 680, 1360 and 2720 Hz. A real Fourier transform of an even-reflected scored interval gives orthogonal contributions and avoids a jump between the last and first samples. It uses the complete excerpt and is intended for offline training labels.

Use the training and teacher extras and the pinned offline checkpoint supported by `stemgenrt.teacher`. The plan uses schema `frequency-calibrated-teacher-target-v1`. Record a training manifest, the three released EMA specialists and the teacher checkpoint, each with a path and SHA-256. The artist-disjoint `fit` and `transfer` lists each contain track ID (`id`), artist, name, root and 128-aligned sample offsets. Set `warmup_samples` to 88064, `scored_samples` to 176512 and supply the bootstrap `seed`, `source_bindings` and `input_bindings`. Optional `stop_files` pause execution between artists.

```bash
for stage in statistics fit transfer report; do
    python scripts/calibrate_teacher_projection.py "$stage" --plan "$PLAN" --plan-sha256 "$DIGEST" --output "$RUN" || exit 1
done
```

The statistics and transfer stages can distribute artists with `--shard` and `--shards`. The fit stage seals the two sets of coefficients before transfer. The report includes paired intervals against the release, direct Bass, power-weighted correction and fitted scalar correction. Its decision concerns the quality of an offline training target. Streaming student quality requires a separate matched training experiment.
