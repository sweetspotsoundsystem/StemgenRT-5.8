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
architecture. A successful development comparison and a repeat experiment are
prerequisites for further deployment work.
