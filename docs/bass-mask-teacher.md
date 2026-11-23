# Training causal Bass output heads with an offline teacher

This experimental pilot trains two copies of a frozen Bass specialist on identical audio. The supervised control uses the reference stems. The student replaces half of the ordinary Bass and Other losses with offline teaching targets. Both arms train the spectral mask heads and share the original causal feature extractor, learning rate, data addresses, loss normalizers and auxiliary examples. Drums and Vocals stay fixed. At inference the existing graph emits Bass and computes Other from the stereo mixture and the three estimated instruments.

Install the training and teacher extras. The offline target requires the pinned MDX/Demucs checkpoint recorded in `stemgenrt.teacher`. The direct target takes its native Bass output. The power-weighted target adds a fraction of the mismatch between the mixture and the combination of released Drums/Vocals and native teacher Bass/Other. The fraction is the teacher's Bass energy divided by its combined Bass and Other energy, clamped between 0.05 and 0.95. Both targets use the complete scored training excerpt.

Provide a JSON plan using schema `paired-bass-mask-teacher-v1`. Identify the training manifest, frozen EMA models for drums/bass/vocals and teacher checkpoint with `path` and `sha256`. Set `target` to the preselected `direct` or `power` rule and record the qualification evidence alongside the plan. Each entry in the artist-disjoint `fit` and `transfer` lists has `id`, `root`, `name`, `artist` and 128-aligned sample `offsets`. The ordered `schedule` contains exactly one entry with `id` and `offset` per fitting crop. Set `steps` to its length, `lr` to 0.0003, `warmup_samples` to 88064 and `scored_samples` to 176512. Include an integer `seed` and maps of input and source paths to their hashes using `input_bindings` and `source_bindings`. Optional `stop_files` pause the run.

```bash
for stage in fit transfer report; do
    python scripts/train_mask_teacher.py "$stage" --plan "$PLAN" --plan-sha256 "$DIGEST" --output "$RUN" || exit 1
done
```

Training runs in single-thread deterministic CPU FP32. Each atomic `paired.pt` stores both sets of heads and Adam moments, their shared address cursor, RNG and backend settings, parent identities, losses and gradients. Repeating the command recovers both arms from the last committed step. The final `endpoint.json` seals their tensor and checkpoint hashes before held-artist evaluation. The score report compares each model against the release and the student against the same-budget supervised control. The absent-Bass metric is output level relative to the control and the boundary metric measures changes in error across adjacent hop endpoints.

To export either final model, supply the original three EMA checkpoints to `scripts/export_independent_model.py` and add `--bass-mask-pilot "$RUN/paired.pt" "$PAIRED_SHA256" student` or use `supervised`. The exporter verifies all parent identities, installs the selected raw mask weights, and creates the normal ONNX graph and native PyTorch waveform fixtures. Record model-quality and device-runtime measurements separately.
