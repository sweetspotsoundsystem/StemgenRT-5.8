"""Calibrate the frequency distribution of offline Bass/Other target inconsistency."""
import numpy as np

EDGES_HZ = (85., 170., 340., 680., 1360., 2720.)
BANDS = len(EDGES_HZ) + 1
RATE = 44100
BASS_BUDGET_DB = .35
RIDGE = .01


def spectrum(x):
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] != 2 or x.shape[1] < 2 or not np.isfinite(x).all():
        raise ValueError("Require finite stereo samples")
    return np.fft.rfft(np.concatenate((x, x[:, ::-1]), axis=-1), axis=-1)


def frequency_bands(samples):
    bins = np.fft.rfftfreq(2*samples, d=1./RATE)
    return np.searchsorted(EDGES_HZ, bins, side="right")


def inner_products(left, right, bands):
    """Inner products on the original interval using the orthogonal even extension."""
    if left.shape != right.shape or left.shape != (2, len(bands)):
        raise ValueError("Spectrum shapes differ")
    products = np.real(left * np.conj(right)).sum(axis=0)
    products[1:-1] *= 2
    return np.bincount(bands, weights=products/(4*(len(bands)-1)), minlength=BANDS)


def projected_bass(bass, mismatch, weights):
    """Construct an offline FP32 target with independent gains on seven bands."""
    w = np.asarray(weights, dtype=np.float64)
    if (w.shape != (BANDS,) or not np.isfinite(w).all() or np.any((w < 0) | (w > 1))
            or bass.shape != mismatch.shape or bass.dtype != np.float32):
        raise ValueError("Invalid offline projection gains or target geometry")
    n = bass.shape[-1]
    filtered = np.fft.irfft(spectrum(mismatch) * w[frequency_bands(n)], n=2*n, axis=-1)[:, :n]
    return bass + filtered.astype(np.float32)


def clip_statistics(mismatch, bass_error, other_error):
    """The desired corrections are truth minus direct Bass and direct Other minus truth."""
    x = spectrum(mismatch)
    bands = frequency_bands(mismatch.shape[-1])
    g = inner_products(x, x, bands)
    cross = np.stack([inner_products(x, spectrum(target), bands) for target in (bass_error, other_error)])
    return g, cross


def fit_projection(records, *, scalar):
    """Minimize normalized Other error with a bound on the increase in direct Bass error."""
    roots = sorted({r["root"] for r in records})
    groups = [[r for r in records if r["root"] == s and min(r["active"]) > 0] for s in roots]
    if len(groups) != 2 or any(not group for group in groups):
        raise ValueError("Both calibration corpora require dual-active artists")
    gram = np.zeros((2, BANDS))
    cross = np.zeros_like(gram)
    base = np.zeros(2)
    for group in groups:
        for r in group:
            error = np.asarray(r["direct_error"], dtype=float)
            den = np.maximum(error, max(1e-4*r["mixture_energy"], 1e-12)) * (2*len(group))
            gram += np.asarray(r["gram"])[None] / den[:,None]
            cross += np.asarray(r["cross"]) / den[:,None]
            base += error / den
    if not np.isfinite([*gram.flat, *cross.flat, *base]).all() or np.any(gram < 0):
        raise ValueError("Invalid calibration statistics")
    if scalar:
        gram, cross = gram.sum(1, keepdims=True), cross.sum(1, keepdims=True)
    maximum_change = base[0] * (10**(BASS_BUDGET_DB/10)-1)

    def weights(multiplier):
        return np.clip((cross[1]+multiplier*cross[0]) / (gram[1]+RIDGE/gram.shape[1]+multiplier*gram[0]), 0., 1.)

    def change(w):
        return (gram*w*w-2*cross*w).sum(1)

    w = weights(0.)
    multiplier = 0.
    if base[0] == 0.:
        if np.any(cross[0] != 0.):
            raise ValueError("Zero direct Bass error has nonzero target covariance")
        w = np.where(gram[0] > 0., 0., w)
    elif change(w)[0] > maximum_change:
        lo, hi = 0., 1.
        for _ in range(160):
            if change(weights(hi))[0] <= maximum_change:
                break
            hi *= 2.
        else:
            raise ValueError("Bass safety constraint has no numerical solution")
        for _ in range(80):
            mid = (lo+hi)*.5
            if change(weights(mid))[0] > maximum_change:
                lo = mid
            else:
                hi = mid
        multiplier, w = hi, weights(hi)
    delta = change(w)
    if delta[0] > maximum_change+1e-12:
        raise ValueError("Calibration exceeds the Bass error budget")
    return {"weights": np.broadcast_to(w, (BANDS,)).tolist(), "dual_multiplier": multiplier,
            "baseline_normalized_errors": base.tolist(), "change_in_normalized_errors": delta.tolist(),
            "allowed_bass_error_increase": float(maximum_change), "eligible_by_corpus": dict(zip(roots,map(len,groups),strict=True))}
