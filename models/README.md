# Streaming model interface

StemgenRT 0.7.0 uses the independent Drums/Bass/Vocals FP32 graph with residual
Other. Download it with `python scripts/download_streaming_model.py`.
The immutable URL, SHA-256 and all state shapes are in
[`stemgenrt/streaming_models.json`](../stemgenrt/streaming_models.json).

- File: `models/independent.onnx`, 31,946,241 bytes, no external tensors.
- SHA-256: `0a49388dd2a47986ab9a161c8764c22853b4d601c188ac052eb530f076d513c7`.
- Input: `audio_chunk`, float32 `[1,2,128]`, stereo 44.1 kHz.
- Output: `separated_chunk`, float32 `[1,4,2,128]`, Drums/Bass/Vocals/Other.
- State: five Drums, four Bass, four Vocals; 14 inputs and 14 outputs total.

| State suffix | Drums | Bass | Vocals |
| --- | --- | --- | --- |
| `audio_history` | `[1,2,896]` | `[1,2,3968]` | `[1,2,896]` |
| `local_hidden` | `[2,20,96]` | `[2,17,80]` | `[2,20,96]` |
| `global_hidden` | `[2,1,192]` | `[2,1,160]` | `[2,1,192]` |
| `spectral_numerator_tail` | `[1,1,2,128]` | `[1,1,2,128]` | `[1,1,2,128]` |
| `waveform_numerator_tail` | `[1,1,2,128]` | — | — |

Prefix each input state with `drums_`, `bass_` or `vocals_`. Each returned state
adds `next_` before that full name and has the same shape. Initialize all states
to zero, retain every returned state unchanged, and reset on a seek or input gap.
The first call returns invalid pre-roll; subsequent calls emit the preceding
input hop. Pad a partial final hop once, flush with exactly one zero hop, and
trim to the original sample count. `StreamingSeparator` handles this lifecycle.

Other is the delayed physical input minus `((Drums + Bass) + Vocals)`. This
reconstructs the mixture within FP32 rounding; it does not guarantee perfect
source recovery. Do not normalize, clip or rescale input or recurrent states.
Use ONNX Runtime 1.26.0 CPU with the wrapper's one-thread settings.

## Local legacy copies and custom exports

The v0.6.x plugin release attachments and the v0.6.2 model download have been
withdrawn pending provenance and rights review. The downloader refuses
`--release v0.6.2`. Its checksum and eight-state interface remain in the registry
for identifying existing local copies and documenting the historical baseline;
they are not a download offer or a grant of rights. See the
[weights notice](../WEIGHTS-LICENSE.md).

For an existing local copy, pass its recorded SHA-256 explicitly to
`StreamingSeparator`. Its eight-state native class remains `StemgenRT58`.
Custom exports also require an explicit SHA-256 and an exact supported
13-state or legacy eight-state interface. States cannot be carried between
models. Downloading preserves existing files and refuses a different model at
the destination; the new filename avoids overwriting older downloads.
