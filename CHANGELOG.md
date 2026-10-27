# Changelog

## 0.8.0 (unreleased)

- Add the exact full-song/museval scorer, paired song intervals and a documented
  MoisesDB reference preparation tool, with an explicit historical conversion gap.
- Publish a portable prepared training inventory, model card and software citation;
  distinguish code, weights and training-data terms. No weights changed.
- Use the immutable checksummed release asset and disable ORT telemetry.


- Add exact independent graph partitions for the matching three-worker plugin,
  retaining full model, training, data, objectives, recovery, evaluation and
  export support. The released model weights and thirteen-state ABI are unchanged.
- Reject parents whose residual arithmetic or audio-history update cannot be
  reproduced by the partition host. Bind the parent identity to the bytes
  actually loaded and reject a source replaced during conversion.
- Retain the optional carrier/GRU export experiments as diagnostics, disabled
  by default. These candidates do not replace the released model.
- Update runtime documentation with M4 sustained, M1 Pro quick and Windows
  results, including the remaining Windows playback underruns.
