# Experimental peer-conditioned Bass correction

This experiment fits a post-processing model to frozen Drums, Bass and Vocals specialists. Its inputs are their emitted stereo waveforms and the time-aligned mixture. Its outputs preserve Drums and Vocals exactly and recompute Other from the corrected Bass and physical mixture.

The model splits each estimated stem into four complementary bands using trailing means of 8, 64 and 512 samples. For each 128-sample output hop it measures the Bass versus Other power and the Drums versus Vocals power. Two channels share the same coefficients and gates. The readout keeps 511 samples of filter history and consumes no later output hops. CPU and plugin execution times require measurement on the target device.

## Fitting and evaluation

Use the training extras and a frozen JSON plan with the training manifest, sample rate, warmup and scored sample counts, three SHA-256-identified EMA checkpoints and two artist-disjoint lists named `fit` and `transfer`. Each entry identifies the track, artist, corpus and offsets in physical samples. The plan also records source and input file hashes, bootstrap seed, optional stop-file paths and the `peer-conditioned-bass-readout-v1` schema. The training manifest supplies per-file audio hashes. The runner verifies the mixture, Bass and Other recordings as it reads each artist.

Run the four stages in order with the same plan, digest and output directory.

```bash
for stage in statistics fit transfer report; do
    python scripts/train_bass_peer_readout.py "$stage" --plan "$PLAN" --plan-sha256 "$DIGEST" --output "$OUTPUT" || exit 1
done
```

The statistics and transfer stages accept `--shard` and `--shards` for disjoint artist ranges. Training records are restartable. Fitting writes the static control and the conditioned readout to `weights.json`. Both use the same ordinary Bass and Other loss terms and the same isolated and absent Bass examples. The transfer results use directly rendered FP32 output, one vote per eligible artist and stratified paired bootstrap intervals. The network initialization may already have seen these songs. The artist separation applies to fitting the correction.

For inference instantiate `stemgenrt.bass_peer_readout.PeerBassReadout` with the desired vector from `weights.json`. Pass consecutive float32 arrays to `process(mixture, stems)` with shapes `(2, samples)` and `(3, 2, samples)` in Drums, Bass, Vocals order. Each call must contain a whole number of aligned 128-sample output hops. Start a new instance at each stream reset.

The JSON keys used by the command are `schema`, `sample_rate`, `warmup_samples`, `scored_samples`, `bootstrap_seed`, `manifest`, `models`, `fit`, `transfer`, `source_bindings` and `input_bindings`. The manifest and each of the three model entries have `path` and `sha256` fields. A panel entry has `id`, `root`, `name`, `artist` and an `offsets` list of 128-aligned sample indices. Binding maps associate paths with SHA-256 digests. The pilot uses 88,064 warmup samples and 176,512 scored samples at 44.1 kHz.
