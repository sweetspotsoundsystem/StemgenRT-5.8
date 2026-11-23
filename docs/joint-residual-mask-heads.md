# Training the three streaming output heads together

Drums, Bass and Vocals each have a dedicated causal specialist. The deployed Other stem equals the physical mixture minus all three emitted stems. This experiment trains the three spectral output heads together and measures which heads should receive the Other error during backpropagation.

Both arms start from the same three released EMA checkpoints. They use the same 44.1 kHz stereo crops, reference sources, optimizer, clipping, normalizers and five equally weighted losses. Four losses cover ordinary Drums, Bass, Vocals and Other. The fifth covers either isolated Bass or Bass output when the Bass reference is removed from the input. The `bass_only` control detaches Drums and Vocals only inside the calculation of the residual. The `all_stems` arm backpropagates that error through all three estimates. Every arm updates all three heads against their own labeled sources. The forward calculation, streaming state and deployed output geometry are identical in the two arms.

Use schema `joint-residual-mask-heads-v1` with the training manifest and three EMA checkpoint identities. Each has a path and SHA-256 checksum. Supply artist-disjoint `fit` and `transfer` track lists with `id`, `artist`, `name`, `root` and physical sample `offsets`. The `schedule` specifies the fitting track ID, offset and `auxiliary` value of `solo` or `absent` for each update. Set `steps` to the schedule length, `lr` to 0.0003 and `warmup_samples` and `scored_samples` to 88064 and 176512. Include the bootstrap and runtime `seed`, the input and source digest maps (`input_bindings` and `source_bindings`), and any optional `stop_files`. Audio hashes for all five files are checked on first access to each track.

```bash
for stage in fit transfer report; do
    python scripts/train_joint_residual_heads.py "$stage" --plan "$PLAN" --plan-sha256 "$DIGEST" --output "$RUN" || exit 1
done
```

The CPU FP32 trainer uses Adam and clips the head gradients of each specialist to a norm of one. Each update atomically commits both sets of spectral heads, complete Adam moments, the common address cursor, loss and gradient history, parent identities, RNG and backend settings. Repeating the fit command recovers the pair. After the final step, `endpoint.json` seals the states. Transfer scoring measures ordinary Bass and physical Other with a paired, stratified artist bootstrap, plus isolated and absent Bass, hop-boundary errors and active Drums/Vocals separately.

To export either arm through the existing ONNX and native-fixture workflow, supply the original three EMA checkpoints to `scripts/export_independent_model.py` and add `--joint-residual-heads "$RUN/paired.pt" "$PAIRED_SHA256" all_stems` or use `bass_only`. The export authenticates all parent weights and reports the separately trained tensors for each specialist. Device timing and song-quality qualification require their own measurements.
