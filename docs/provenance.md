# Provenance and migration

StemgenRT-5.8 is a divergent fork of
[Phil Wang's HS-TasNet implementation](https://github.com/lucidrains/HS-TasNet),
with ancestry in [HS-TasNet](https://arxiv.org/abs/2402.17701). The Python
package is `StemgenRT`, the import is `stemgenrt`, and the training command is
`stemgenrt-train`. Use the upstream repository for the original `hs_tasnet` API.

## Version 0.7.0

The default ONNX model now combines independently trained Drums, Bass and
Vocals, with physical-residual Other. Its 13 states are incompatible with the
previous joint model's eight states. Model files and host adapters must match.
The graph uses the final 40,000-update parameter EMA from each source model.
Model hashes and complete deployment metadata are distributed with the plug-in.

The download script and `StreamingSeparator` select the new default as documented in
[the model interface](../models/README.md). Downloads of the previous v0.6.2 graph
have been withdrawn pending provenance and rights review. Its checksum and state
contract remain for identifying existing local copies and the historical baseline;
the downloader refuses `--release v0.6.2`. See the [weights notice](../WEIGHTS-LICENSE.md).
`StemgenRT58`, its training/recovery support and integer exporter remain
available for existing native checkpoints. They do not construct the new model.

The plug-in's established StemgenRT name, AU/VST3 identity and parameter layout
are retained. Loading the new release in an existing session uses the new
separation model. The temporary StemgenRT Independent test build had separate
host IDs; projects using that test build must replace its instance explicitly.

The ONNX graph retains its original `stemgenrt.experimental` metadata key and
export schema so the already verified bytes and provenance stay unchanged.
That historical schema name does not select the GitHub release channel.

## Source and reproducibility

The release includes model definitions, training recipes, objectives, data
augmentation, optimizer/EMA/RNG recovery, evaluation and export. Source archives
exclude datasets, training checkpoints, draft papers, private experiment
records and machine-specific source-comparison manifests.

Training constructors initialize untrained weights. Reproduction requires the
same ordered, authenticated audio manifests and software environment; complete
resume additionally requires native recovery checkpoints. An inference ONNX
file does not contain the original optimizer, random streams or data cursor.
Tests verify numerical and lifecycle behavior; physical host timing needs
separate measurement for each deployed graph.
