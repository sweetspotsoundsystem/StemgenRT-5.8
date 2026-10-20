# Independent-model deployment experiments

Alternative execution graphs can preserve the independent models' training
weights and complete 13-state interface. Conversion creates a new candidate and
a transformation report. The released graph is preserved.

The accompanying model definitions, training recipes, objectives, augmentation,
checkpoint recovery and evaluation remain in this source distribution. These
execution experiments use the same learned models and training recipe. Each
candidate needs numerical, quality and physical Mac timing checks before use.

## Produce a candidate

Use the existing ONNX development dependencies and an authenticated original
FP32 independent-model export:

```sh
python scripts/optimize_independent_model.py \
  --source /path/to/independent.onnx \
  --sha256 EXPECTED_SOURCE_SHA256 \
  --variant layout-fp32 \
  --output /path/to/new-candidate.onnx
```

| Variant | Transformation | Numerical status |
| --- | --- | --- |
| `layout-fp32` | Specialize fixed layouts while retaining the original recurrence | Same FP32 operations; runtime qualification remains required |
| `fused-fp32` | Replace twelve explicit recurrences with standard ONNX GRU operations | Same learned recurrence; FP32 rounding can differ |
| `fused-layout-fp32` | Also collapse redundant reshape paths under the fixed tensor dimensions | Same arithmetic as the fused candidate, subject to runtime optimization |
| `fused-layout-global-int8` | Also quantize twelve global input/output projection matrices | Approximate output; requires quality and preservation evaluation |

The default `layout-fp32` variant retains every original recurrent operation.
It proves the six gate-slice shapes in each cell from the validated recurrence
and the public hidden-state dimensions, then specializes only fixed layouts.

For optional GRU fusion, the converter checks the exact exported recurrence before replacing it.
PyTorch's reset/update/candidate gate order becomes ONNX's
update/reset/candidate order. Both bias vectors are retained separately, with
`linear_before_reset=1`. Public names, shapes and types are preserved. Layout
simplification uses statically inferred shapes. A transpose becomes a reshape
only when the order of all non-singleton axes is unchanged.

The integer variant uses dynamic unsigned activations and symmetric signed
weights in [-64, 64], with one weight scale per matrix. It selects the two global
input and two global output projections in each specialist. Recurrent weights,
band encoders and heads, waveform processing, FFTs and synthesis stay FP32.
The range bounds avoid signed-pair saturation and 32-bit dot-product overflow
for these matrices. They do not establish quality or cross-platform trajectory
agreement.

Conversion marks every result `quality_measured=false` and
`native_host_qualified=false`. It does not change download URLs, packaged models,
training endpoints, default inference or plugin assets.

## Validate and measure

Use independent native PyTorch references for every output/state, covering
silence, transients, nonzero state, reset and EOF. Keep state continuous through
long music excerpts. Check the literal residual arithmetic and alignment.
For integer candidates, distinguish agreement with an integer reference from
the intended difference against FP32.

Evaluate ordinary development mixtures, target-absent rejection and isolated
target preservation. Good mean SDR can coexist with worse isolated-target
reconstruction. Keep test-set results out of candidate selection.

Operator profiling and uninstrumented timing use separate sessions. Python
timings include interpreter and allocation overhead. The
[native independent probe](../examples/native_independent/README.md) provides a
pinned runtime, preallocated buffers, one-thread paced measurements and native
waveform/state preflight. Compare variants with balanced run order, retain raw
samples, and count every missed 128-sample deadline. Separate compute time from
arrival-to-completion latency. Several shorter segments with intervening resets
do not constitute one continuous endurance run.

Run the same authenticated packet on physical M4 and M4 Pro machines. Linux
timing is diagnostic only. The complete plugin callback/worker path and installed
AU/VST3 behavior need their own qualification after quality and worker checks.

References: [ONNX GRU](https://onnx.ai/onnx/operators/onnx__GRU.html),
[ONNX Runtime profiling](https://onnxruntime.ai/docs/performance/tune-performance/profiling-tools.html),
[ONNX Runtime quantization](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html).
