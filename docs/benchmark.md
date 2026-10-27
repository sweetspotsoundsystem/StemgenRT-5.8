# Full-song scoring and data preparation

Install `python -m pip install -e '.[streaming,benchmark]'`. Put each complete test
song in its own directory containing `mixture.wav`, `drums.wav`, `bass.wav`,
`vocals.wav` and `other.wav`. Supply already aligned, finite stereo 44.1 kHz audio.
Keep the declared test set separate from training/preparation selections.

```sh
python scripts/benchmark_full_songs.py --reference-root /data/test \
  --onnx models/independent.onnx \
  --sha256 0a49388dd2a47986ab9a161c8764c22853b4d601c188ac052eb530f076d513c7 \
  --output /results/new-full-song-run
```

Alternatively use `--estimate-root /aligned-estimates` in place of `--onnx` and
`--sha256`. This scores four existing stem WAVs per song. `--whole-song-only`
skips museval and must not be described as reproducing its score. One song at a
time is rendered; the complete four-source arrays and BSS Eval filters can
require substantial RAM. Inputs are hashed and versions and protocol recorded.
A completed `summary.json` appears only after every declared song succeeds.
No input or existing output directory is modified. Whole-song SDR includes every
sample; museval's undefined frames remain explicit with counts. The package's
`evaluation` development-window metric is a different protocol.

## MoisesDB reference conversion

Obtain MoisesDB under its [dataset terms](https://github.com/moises-ai/moises-db).
Install its official `moisesdb` package and record the exact environment. Supply
a JSON selection such as `[{"provider": "moisesdb_v0.1", "id": "actual-song-id"}]`.
Use the provider/id values in your downloaded dataset, not the example literally.

```sh
python scripts/prepare_moisesdb.py --data-path /data/moisesdb \
  --selection /data/training-selection.json --output /data/prepared-new
```

The official library decodes/resamples each source at 44.1 kHz and constructs its
taxonomy stems. Its internal source-group mixing trims to the shortest member.
Our wrapper maps drums/bass/vocals directly and every remaining taxonomy group,
including percussion, to Other. Missing groups are silence; mono duplicates to
stereo; tails between taxonomy stems are zero-padded. There is no level
normalization. Output is floating WAV, avoiding clipping and PCM quantization.
The summed mixture, output hashes and installed package versions are retained.
Use a locked environment for repeated conversions. Unknown taxonomy fails.

This is a **new documented reference pipeline**. The released endpoints used
prepared PCM16 WAVs; their original raw conversion script has not been recovered.
The tool does not claim to recreate those historical bytes. Compare any prepared
copy against `configs/released-training-inventory.json` before claiming exact
training reproduction. Its root placeholders must be replaced with local dataset
paths. The inventory and existing manifest/data tools preserve the historical
selected tracks and addressing, but cannot recover missing source audio or its
unknown conversion choices. Do not mix test songs into a new training manifest.

## Paired song intervals

`scripts/paired_song_intervals.py` takes the two frozen summary files with
`--previous`, `--current` and a new `--output` JSON path. The default artifact keys
are `int8` and `independent_fp32`. It requires identical unique song inventories,
uses 10,000 shared paired bootstrap draws (seed 20261025) and reports pointwise
95% percentile intervals. The script does not estimate training-seed variability
or correct multiple comparisons. Raw benchmark summaries accompany the private
manuscript and will be released with it; the scorer can produce new records now.
