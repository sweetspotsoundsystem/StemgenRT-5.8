# Independent stem native worker probe

This example prepares a self-contained packet for physical Apple M4 and M4 Pro
machines. It measures complete FP32 graphs, input/output/state copies, and the
combined physical residual Other on one paced inference worker. It compares
three separate sessions with a single combined graph containing all 13 released states or 14 with the experimental bass waveform branch. The probe does
not measure a plugin queue, audio driver or DAW callback.

The selected geometries are the default vocal/joint band models and independent
bass/drums specialists. Bass retains 3968 samples of analysis history. Drums supports either 896 or 3968 history samples and has
a fifth waveform state. The optional bass waveform branch adds its own fifth state; the probe validates this declared interface as well. All emit one 128-sample-delayed hop at 44.1 kHz; adding
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
  --training-source dist/stemgenrt-0.7.0.tar.gz \
  --training-source-sha256 SOURCE_ARCHIVE_SHA256 \
  --vocals-checkpoint /path/to/vocals/checkpoint.pt --vocals-sha256 VOCAL_SHA256 \
  --bass-checkpoint /path/to/bass/checkpoint.pt --bass-sha256 BASS_SHA256 \
  --drums-checkpoint /path/to/drums/checkpoint.pt --drums-sha256 DRUMS_SHA256 \
  --joint-checkpoint /path/to/joint/checkpoint.pt --joint-sha256 JOINT_SHA256 \
  --expected-updates 40000
```

The joint control is optional. `--expected-updates` checks the current stage's
endpoint when supplied. Authenticate the selected checkpoint identities before exporting.

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

With a joint control, six cases run in six rotating rounds: vocal, bass, drums,
joint, three-session DBV with host residual, and one combined graph with residual.
Each trial has 1024 paced warmup hops and 10,000 measured hops, for 60,000
measured hops per case across six segments. Without a joint control, five cases
rotate through five rounds and 50,000 measured hops per case. Neither is one continuous endurance run.

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

## Continuous endurance protocol

A packet can additionally declare `endurance_order` (each case exactly once),
`endurance_warmup_hops`, `endurance_measured_hops`, and `input_period_hops`.
After the balanced short trials, the runner performs one uninterrupted paced
trial per case. The fixed synthetic input repeats, while every model state
continues across input boundaries. The analyzer checks the input period,
exposure, every raw timing sample, final finite states, and pinned runtime.
Endurance results appear separately from the shorter trials; both retain the
original arrival deadlines even when the worker falls behind.

## Compare scheduling with an unchanged graph

`prepare_scheduling.py` takes an authenticated existing packet, its manifest
SHA256, an explicit model key and graph SHA256, and this checkout's matching
training source archive. It copies the selected graph and its independent
PyTorch reference without exporting new weights:

```sh
python examples/native_independent/prepare_scheduling.py \
  --source-packet /path/to/verified-packet \
  --source-manifest-sha256 MANIFEST_SHA256 \
  --model-key released-fp32 --model-sha256 GRAPH_SHA256 \
  --training-source dist/stemgenrt-0.7.0.tar.gz \
  --training-source-sha256 SOURCE_ARCHIVE_SHA256 \
  --output /allocated/new-scheduling-packet
```

The resulting packet uses all six permutations of QoS, Mach real-time and
standalone audio-workgroup scheduling, followed by five minutes of continuous
state per condition. The Mach computation estimate is fixed at 2.7 ms per
128-sample hop before the run. Initial and final policy readbacks must match;
workgroup interval counts must cover every warmup and measured hop. Timing
keeps the original arrival clock and includes interval bookkeeping. Endpoints
must be byte-identical across these same-graph conditions.

A separate 80-hop ORT operator profile runs after every timing trial. Its node
durations include instrumentation and warmup, so they identify work to investigate
without measuring deadline performance. The analyzer rejects instrumented runs
from the timing results and independently recomputes operator summaries.

The standalone workgroup does not represent an AU host workgroup. Use the plugin's
paced callback comparison and physical DAW tests to evaluate that complete path.
Production scheduling remains unchanged until fresh target-machine evidence
supports a change. Apple documents the soft real-time behavior in its
[Mach scheduling guide](https://developer.apple.com/library/archive/documentation/Darwin/Conceptual/KernelProgramming/scheduler/scheduler.html)
and host collaboration in [Understanding audio workgroups](https://developer.apple.com/documentation/audiotoolbox/understanding-audio-workgroups).
