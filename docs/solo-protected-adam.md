# Isolated Bass protection during Adam training

This paired experiment starts two full streaming Bass specialists from the same equal-weight average of two sampling schedules on two training streams. Each update uses an ordinary example from each corpus. Its Bass and physical Other errors each receive 40% weight. Bass-only and Bass-absent examples made from the same tracks each receive 10%. Each error is normalized against the starting model on the same sample. Drums and Vocals come from the released specialists. Other is the physical mixture minus the sum of the three estimates.

A third artist supplies a separate, audible, isolated Bass reference at each update. Both arms measure the gradient of its relative squared error before Adam acts. The protected arm projects the proposed parameter displacement onto the half-space with a nonpositive directional derivative. The plain arm takes the original displacement. The constraint applies to that training example under a first-order approximation. The held-artist comparison measures the actual effect on separated audio.

Run the following stages against an authenticated plan and the corresponding local audio and checkpoints. The run records which three artists and physical offsets belong to every update. Each model uses fresh Adam at 0.00001 and EMA decay 0.98. A checkpoint contains raw and EMA parameters, Adam moments, RNG streams, the next data address and the projection measurements. Repeating an interrupted stage on the same runtime resumes at its saved address.

    export CUBLAS_WORKSPACE_CONFIG=:4096:8
    python scripts/train_solo_protected_adam.py plain --device cuda --plan "$PLAN" --plan-sha256 "$SHA" --output "$OUT"
    python scripts/train_solo_protected_adam.py protected --device cuda --plan "$PLAN" --plan-sha256 "$SHA" --output "$OUT"
    python scripts/train_solo_protected_adam.py transfer --device cpu --plan "$PLAN" --plan-sha256 "$SHA" --output "$OUT"
    python scripts/train_solo_protected_adam.py report --device cpu --plan "$PLAN" --plan-sha256 "$SHA" --output "$OUT"

Transfer scores final EMAs alongside the four-parent starting point and released Bass, using identical clips and frozen peer stems. It measures ordinary Bass and Other, isolated Bass, Bass-absent output and hop-boundary error. Confidence intervals resample artists within each corpus. This pilot reuses a previously benchmarked artist panel from the training corpora.

The standard independent exporter accepts the selected arm's checkpoint and SHA-256 along with the original Drums and Vocals endpoints. For the 192-update experiment, use the Bass flags `--bass-updates 192 --bass-parent-updates 48000`. The exported network retains the 128-sample hop and 13 recurrent states.
