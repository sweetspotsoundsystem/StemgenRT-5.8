# StemgenRT independent models

Maintainer: Axel Delafosse. Software 0.8.0 uses the byte-identical v0.7.0 weights.
The manuscript is a private draft; no public paper link or DOI is available yet.

## Intended use and interface

Research on streaming music separation and interactive stem playback at stereo
44.1 kHz. The graph consumes 128 samples and thirteen persistent FP32 states per
call. Drums, Bass and Vocals have separate learned parameters; Other is the
aligned mixture minus their estimates. See [architecture](docs/independent-stems.md).
There are 7,866,572 parameters and approximately 13.25 million dense MACs per hop.

The graph alignment is 128 samples. Model algorithmic latency under the
synthesis-frame convention is 256 samples (5.805 ms). Separately, the asynchronous
plugin reports 256 samples of PDC at a 128-sample host buffer. Including its queue
under the synthesis-frame convention gives 384 samples (8.707 ms). None is an
audio-interface round-trip latency. Runtime qualification is specific to each
machine, host, buffer and build. Windows remains experimental.

## Training inventory and provenance

Each model was trained from scratch for 40,000 updates, using the final parameter
EMA, with no teacher loss or inherited legacy weights. Sampling gives equal
probability to 83 MUSDB18-HQ songs and 218 prepared MoisesDB songs, then samples
uniformly within the chosen group. The MUSDB count excludes fourteen development
songs and three recordings missing from the local copy: Chris Durban – Celebrate,
Hop Along – Sister Cities, and Leaf – Wicked. These omissions were availability
issues, not a quality filter. The Moises inventory removed one exact duplicate:
Firefly – Up Close And Personal matched Firefly – The Long Way in all five WAVs.

[The portable prepared inventory](configs/released-training-inventory.json)
records song names, file hashes and crop activity metadata. Native training
checkpoints are not currently distributed; the published inference artifact is
ONNX. The historical raw Moises conversion script has not been recovered.
[The reference preparation tool](docs/benchmark.md) documents its own conversion
policy and does not claim byte-equivalent historical reconstruction.

Dataset credits: Z. Rafii et al., [MUSDB18-HQ](https://zenodo.org/records/3338373)
(2019); I. Pereira, F. Araújo, F. Korzeniowski and R. Vogl,
[MoisesDB](https://arxiv.org/abs/2307.15913) (2023).
Code is MIT; weights have [separate terms and unresolved third-party rights
questions](WEIGHTS-LICENSE.md). Dataset audio is not redistributed here.
Legacy v0.6.x baselines additionally used third-party audio with incompletely
documented provenance. Their plugin release attachments and model download link
have been withdrawn pending rights review. Historical baseline results and local
compatibility metadata remain; this does not erase copies in Git/LFS history.

## Frozen quality results

All 50 complete MUSDB18-HQ test songs are included. Museval 0.4.1 uses BSS Eval v4,
median frames then median songs; whole-song SDR averages the per-song MDX/Demucs
energy-ratio score. Both average four sources for the overall value. There is no
gain fit, delay search, clipping or development activity gate in the latter score.

| Source | Museval SDR | Whole-song SDR | Change from v0.6.2, paired 95% interval |
| --- | ---: | ---: | ---: |
| Drums | 4.893 | 5.431 | +0.183 [−0.028, +0.384] |
| Bass | 4.073 | 3.846 | +0.133 [−0.147, +0.433] |
| Vocals | 5.179 | 5.462 | +1.149 [+0.919, +1.393] |
| Other | 2.357 | 2.430 | −0.504 [−1.008, −0.095] |
| Overall | 4.125 | 4.292 | +0.240 [+0.012, +0.453] |

Intervals use 10,000 paired song bootstrap draws, seed 20261025, percentile
endpoints, with the same sampled songs for all sources. They describe song
variation for these two endpoints, not uncertainty over training runs or
simultaneous coverage of all comparisons. Vocals improve on 47/50 songs.
Drums and Bass intervals include zero. Other's median change is −0.200 dB;
the two largest declines account for about 55% of its net mean decline.
The comparator differs in architecture, capacity, precision and training history.
These results do not isolate a causal benefit of independent models.

Controlled absent/isolated-source comparisons are also confounded: the joint
comparator trained with vocal-absent and vocal-only views, while specialists
received views for their own target. Do not interpret Drums/Bass controlled-view
differences as a matched ablation. Clipping occurred in 67–89% of training
updates and 92–96% of the final 10,000 updates. These frequencies do not measure
each loss term's contribution to the gradient.

## Limits and reproducibility

Residual Other can deteriorate even when the learned targets improve. Listening
reports are informal. No blinded population study is claimed. The development
panel was repeatedly consulted and the test collection evaluated previously;
the frozen run does not make the whole research process benchmark-blind.

Use [the full-song scorer and preparation instructions](docs/benchmark.md),
[training/recovery recipes](docs/training.md) and the plugin's
[machine-specific runtime evidence](https://github.com/sweetspotsoundsystem/stemgen-rt/blob/main/docs/parallel-workers.md).
The exact graph SHA-256 is
`0a49388dd2a47986ab9a161c8764c22853b4d601c188ac052eb530f076d513c7`.
