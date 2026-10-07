# Independent stem native worker probe

This example prepares a self-contained packet for physical Apple M4 and M4 Pro
machines. It measures complete FP32 graphs, input/output/state copies, and the
combined physical residual Other on one paced inference worker. The probe does
not measure a plugin queue, audio driver or DAW callback.

The selected geometries are the default vocal/joint band models and independent
bass/drums specialists. Bass retains 3968 samples of analysis history. Drums has
a fifth waveform state. All emit one 128-sample-delayed hop at 44.1 kHz; adding
the host's 128-sample accumulation gives the intended 256-sample system delay.

## Prepare a packet

Work from this source checkout with its training and ONNX dependencies and the
`build` package installed.
Build a source distribution, and obtain the official ONNX Runtime **1.26.0
osx-arm64 SDK**. The builder copies its headers, libraries and license notices as
regular files. It checks the archive's SHA256 and the actual model, trainer,
data, loss, recovery, export and configuration bytes against this checkout.

Provide complete native EMA checkpoint paths and their SHA256 values:

```sh
python -m build
python examples/native_independent/prepare.py \
  --output /allocated/new-native-packet \
  --sdk-root /path/to/onnxruntime-osx-arm64-1.26.0 \
  --training-source dist/stemgenrt-0.6.2.tar.gz \
  --training-source-sha256 SOURCE_ARCHIVE_SHA256 \
  --vocals-checkpoint /path/to/vocals/checkpoint.pt --vocals-sha256 VOCAL_SHA256 \
  --bass-checkpoint /path/to/bass/checkpoint.pt --bass-sha256 BASS_SHA256 \
  --drums-checkpoint /path/to/drums/checkpoint.pt --drums-sha256 DRUMS_SHA256 \
  --joint-checkpoint /path/to/joint/checkpoint.pt --joint-sha256 JOINT_SHA256 \
  --expected-updates 40000
```

The joint control is optional. `--expected-updates` checks the current stage's
endpoint when supplied. With a frozen research experiment, authenticate its
completion proof and follow its checkpoint-selection policy before exporting.

For an architecture-only probe, replace the bass/drums checkpoint arguments
with `--scratch-specialists`. It constructs their default untrained geometries
with the declared seed and labels them as untrained in the packet. Such a packet
cannot qualify the final trained candidate.

The builder exports each complete graph and generates 192 hops of independent
native PyTorch waveform/state references using synthetic audio only. The combined
reference checks that each stem equals its separate model, Other uses the delayed
physical input, and the next physical-history copy is exact. It includes the
matching training archive and hashes every packet file. No corpus is opened.

## Run on each Mac

Copy the prepared directory to a physical M4 or M4 Pro. Use native arm64 Python
3.9 or later and Apple's command-line developer tools. No additional Python
packages or model downloads are needed on the target machine.

```sh
cd /path/to/new-native-packet
python3 run_macos.py --output results-m4 \
  --power-notes "AC power; Low Power Mode off; other workloads closed"
```

Describe the actual conditions. Use a different output directory on M4 Pro.
Existing results are preserved. Idle sleep is prevented only during the command.
The runner creates a result ZIP, or a diagnostic ZIP if a preflight or build fails.

With a joint control, five cases run in five rotating rounds: vocal, bass, drums,
joint and combined DBV with residual Other. Each trial has 1024 paced warmup hops
and 10,000 measured hops, for 50,000 measured hops per case across five segments.
Without a joint control, four cases rotate through four rounds and 40,000 measured
hops per case. Neither is one continuous endurance run.

The CPU provider uses sequential execution, one intra/inter-op thread, no spinning,
KleidiAI disabled, flushed denormals, preallocated tensors and user-interactive
QoS. Absolute arrivals and deadlines use one Mach-derived clock. Late calls keep
their original deadlines; no hops are dropped or reset.

Every case must pass waveform and state preflight before timing. Combined preflight
also checks the exact FP32 residual calculation and delayed physical-history copy.
The analyzer independently recomputes statistics and deadline counts from raw CSVs,
checks package/runtime identities, and requires repeatable final outputs and states.
Linux checks can validate numerical behavior; they are rejected as Mac evidence.

Read loop cost and completion latency separately: completion includes wake-up delay.
Use actual trained models, development quality, listening, both target Macs and
plugin integration checks before recommending deployment.
