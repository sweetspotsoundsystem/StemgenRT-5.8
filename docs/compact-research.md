# Compact causal separator experiment

`stemgenrt.compact.CompactSeparator` is an untrained experimental architecture.
It is not the released plugin model. The default constructor produces a single
stereo vocal estimate; `sources=("drums", "bass", "vocals", "other")` constructs
the joint training control.

Both use 1024-sample asymmetric analysis, 256-sample synthesis and a 128-sample
hop at 44.1 kHz. Three causal GRU layers of width 384 receive current-frame
normalized complex spectra and log magnitudes. One linear projection estimates
independent complex spectral masks. There is no source softmax, shared residual
correction or waveform decoder. The vocal variant has no unused source heads.
Persistent state consists of audio history, GRU hidden state and synthesis tail.

The graph alignment is 128 samples. Host hop accumulation contributes the other
128 samples to the intended 256-sample (5.805 ms) system delay. This geometry
does not establish that an implementation meets the CPU processing deadline.
`compute_budget()` reports dense learned MACs, with its exclusions listed;
measured native runtime and separation quality remain required.

The variants initialize the same backbone and vocal mask weights when given
the same seed. The joint control has a larger output projection, so it has
more parameters and MACs. Report those counts alongside training exposure.
Both variants return independent learned outputs. For a replacement comparison,
combine the candidate vocal estimate with the frozen baseline's drums and bass,
then calculate Other from the mixture residual. This assembly is an evaluation
policy, not an additional output of the vocal network.

Training uses the existing data augmentation and detached warmup pipeline.
Set `model_family` to `compact`, `compact_hidden_size` to `384`, and
`compact_layers` to `3` in the training configuration. Set `target_source` to
`vocals` for the single-output model; omit it for joint training. Start with
`--scratch`. Online teacher supervision is not supported for this experiment.
The training schedule must be set explicitly before evaluating quality.

Instrumental and vocals-only auxiliary inputs are constructed from all four
reference stems before selecting the vocal reference. The single-output loss
retains the existing example and active-window reductions, without padding the
network output to four stems. Training snapshots include the architecture,
raw parameters, Adam, EMA, RNG state and absolute data cursor. `--resume` checks
the original configuration and data identity. The compact checkpoint schema
is distinct from the released model's schema.

The released ONNX exporter and plugin runtime do not yet support this
architecture. Development quality and measured native runtime remain necessary
before deployment. The current bass/drums phase uses the completed first-seed
vocal model as its fixed anchor. Multi-seed repeatability is not established.

## Causal band candidate

`stemgenrt.banded.BandSeparator` preserves the same audio geometry while
retaining separate temporal state in 20 nonoverlapping frequency bands. Each
of two layers uses a shared width-96 temporal GRU across bands, followed by a
width-192 global causal GRU. Independent projections encode and decode each
band; equal-width projections run as batched matrix operations. Band boundaries
partition the existing FFT and do not increase its frequency resolution.

The single-output candidate and four-output control have the same backbone and
the same initial vocal masks under a matched seed. Mask output projections are
the only parameter difference. The recurrent core, current-frame normalization
and synthesis never use future audio or source-axis normalization.

| Default band model | Parameters | Dense MACs per hop | Persistent FP32 state |
| --- | ---: | ---: | ---: |
| Vocal | 2,720,484 | 4,805,568 | 25,088 bytes |
| Joint control | 3,317,616 | 5,396,544 | 28,160 bytes |

MAC counts exclude FFT, normalization, elementwise operations, copies and bias
additions. Three vocal-sized networks would total 14,416,704 dense MACs per hop
before that overhead. This is an arithmetic reference for the eventual system
budget. Independently designed bass and drums research candidates are described
in [independent stems](independent-stems.md), together with their matching
training, recovery, evaluation and export code.

For band training, set `model_family="banded"`, `band_width=96`,
`band_global_width=192`, `band_layers=2`, and `precision="fp32"`. Use
`target_source="vocals"` for the vocal candidate and omit it for the joint
control. The existing data, losses, grouped updates and detached warmup apply.
BF16 and online teacher supervision are rejected. The distinct
`stemgenrt-causal-bands-training-v1` schema restores raw weights, Adam, EMA,
RNG, data position and training history. A fresh stage records inherited updates
separately from updates in the current stage. The training schedule remains an
experiment decision, not an architecture default.

`stemgenrt.band_export.export_fp32(model, new_path)` exports the complete one-hop
FP32 graph, including analysis FFT, complex masking, inverse FFT and overlap
synthesis. Its inputs are one `[1,2,128]` audio chunk and four states: audio
history, local band hidden state, global hidden state and spectral numerator
tail. The outputs are one `[1,S,2,128]` chunk and the four updated states.
Keep these states between callbacks and account for the 128-sample graph
alignment. The host adds its own 128-sample accumulation delay.

The graph carries experimental metadata and uses its own four-state interface;
the released plugin loader does not accept it. CPU tests check streaming
causality, all impulse phases, detached warmup, single-output loss updates,
exact recovery and continuous ONNX/PyTorch waveform and state agreement.
These checks do not measure separation quality or M4/M4 Pro performance.
Experiment-specific quality evidence and training authorization are recorded
with each frozen research plan, separately from this portable interface.

During training, grouped band projections fold batch and time into one matrix
dimension before batched multiplication. Autograd then accumulates each band's
weight gradient directly, avoiding a separate weight matrix for every frame.
The parameter shapes, full-group loss reduction and streaming states stay the
same. Evaluation and export retain the original one-hop calculation. Output
and gradient tests allow floating-point reduction roundoff between the two
layouts; exact checkpoint continuation uses the same bound training source.

## Experimental integer export

`stemgenrt.band_integer.export_int8(model, fp32_path, new_path,
expected_fp32_sha256=...)` converts every learned matrix, including packed band
projections and GRU input/hidden products. It authenticates the complete FP32
graph and maps its matrices back to native weights. The default geometry has
27 converted matrices; biases and other initializers remain byte-exact.

Weights use symmetric S8 values in `[-64,64]` with one scalar scale per matrix
initializer. A packed band group shares this scale, while retaining independent
weights. Activations use one dynamic U8 scale over the complete one-hop matrix
input. Scalar weight zero points avoid the unsupported batched per-channel
zero-point shape in the pinned runtime.

The analysis FFT and operations preceding activation quantizers use FP64.
Integer products/dequantization, output masking, inverse synthesis, public
audio and all persistent state tensors use FP32. This explicit boundary
reduces numerical branch differences during repeated dynamic quantization.
`make_integer_reference` reconstructs weight bytes with NumPy and executes
independent int32 products without using ONNX Runtime as an arithmetic oracle.
Tests compare continuous audio and every state, including stronger synthetic
mask weights, and reject altered graph/source identities or quantized weights.

This integer graph is a distinct numerical model. Agreement with its integer
reference does not establish agreement with the original FP32 separator or
separation quality. Any trained checkpoint needs a separate development quality
comparison and native M4/M4 Pro runtime check for this variant.
