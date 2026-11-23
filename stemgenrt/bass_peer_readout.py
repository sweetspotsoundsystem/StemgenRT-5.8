"""Fit and apply a causal Bass correction using the emitted independent stems."""

import numpy as np


HOP = 128
WINDOWS = (8, 64, 512)
N_FEATURES = 48
SCHEMA = "peer-conditioned-bass-readout-v1"
LOSS_WEIGHTS = (0.4, 0.4, 0.1, 0.1)


class PeerFeatures:
    """Four complementary causal bands from Bass, physical Other, Drums and Vocals."""

    def __init__(self):
        self.history = np.zeros((4, 2, WINDOWS[-1] - 1), dtype=np.float32)

    def design(self, mixture, stems):
        """Return a sample-major, interleaved-stereo design matrix for complete output hops."""
        mix, stems = np.asarray(mixture), np.asarray(stems)
        if (mix.ndim != 2 or mix.shape[0] != 2 or mix.shape[1] == 0 or mix.shape[1] % HOP
                or stems.shape != (3, *mix.shape) or mix.dtype != np.float32 or stems.dtype != np.float32):
            raise ValueError("Supply float32 stereo output hops and aligned Drums, Bass and Vocals")
        if not np.isfinite(mix).all() or not np.isfinite(stems).all():
            raise ValueError("Audio must be finite")
        drums, bass, vocals = stems
        other = mix - ((drums + bass) + vocals)
        carriers = np.stack((bass, other, drums, vocals))
        history = self.history.shape[-1]
        signal = np.concatenate((self.history, carriers), axis=-1)
        prefix = np.pad(np.cumsum(signal, axis=-1, dtype=np.float64), ((0, 0), (0, 0), (1, 0)))
        end = prefix[..., history + 1:]
        low = [(end - prefix[..., history + 1 - width:prefix.shape[-1] - width]) / width
               for width in WINDOWS]
        bands = np.stack((low[2], low[1] - low[2], low[0] - low[1], carriers - low[0])).astype(np.float32)
        # Each gate uses both channels from this already emitted hop. Band sums reconstruct each carrier.
        blocks = bands.reshape(4, 4, 2, -1, HOP)
        power = np.square(blocks, dtype=np.float64).mean(axis=(2, 4))
        b, o, d, v = (power[:, i] for i in range(4))
        gates = np.stack((np.ones_like(b), (b - o) / (b + o + 1e-10),
                          (d - v) / (b + o + d + v + 1e-10)), axis=-1).astype(np.float32)
        # Time, stereo, band, carrier, gate. The static-control columns are indices 0,3,6,...
        design = (blocks.transpose(3, 4, 2, 0, 1)[..., None]
                  * gates.transpose(1, 0, 2)[:, None, None, :, None, :])
        self.history = signal[..., -history:].copy()
        return design.reshape(-1, N_FEATURES)


def validate_weights(weights):
    values = np.asarray(weights, dtype=np.float64)
    if (values.shape != (N_FEATURES,) or not np.isfinite(values).all()
            or np.abs(values.reshape(4, 4, 3)).sum(axis=-1).max() > 0.25000000001):
        raise ValueError("Expected 48 finite coefficients with bounded per-carrier gains")
    return values


def apply_design(mixture, stems, design, weights):
    """Apply an aligned design and recompute physical Other in deployed float32 order."""
    delta = (design @ validate_weights(weights)).reshape(-1, 2).T.astype(np.float32)
    drums, bass, vocals = stems
    corrected = bass + delta
    other = mixture - ((drums + corrected) + vocals)
    return np.stack((drums, corrected, vocals, other))


class PeerBassReadout:
    """Stream a fitted model after the three aligned specialist outputs, in 128-sample hops."""

    def __init__(self, weights):
        self.weights = validate_weights(weights).copy()
        self.features = PeerFeatures()

    def process(self, mixture, stems):
        design = self.features.design(mixture, stems)
        return apply_design(mixture, stems, design, self.weights)


def fit_readout(artists, *, conditional):
    """Fit per-artist normalized joint error using ordinary, Bass-only and Bass-absent examples.

    Each artist supplies four unnormalized sets of Gram matrices, design-target products,
    baseline squared errors and mixture energies in the order of LOSS_WEIGHTS. The first
    two targets share the ordinary design. Targets are the desired *additive* Bass correction.
    """
    if not artists:
        raise ValueError("Need at least one fitting artist")
    cols = np.arange(N_FEATURES) if conditional else np.arange(0, N_FEATURES, 3)
    gram = np.zeros((len(cols), len(cols)), dtype=np.float64)
    cross = np.zeros(len(cols), dtype=np.float64)
    for item in artists:
        for i, weight in enumerate(LOSS_WEIGHTS):
            scale = weight / max(float(item["error"][i]), 1e-4 * float(item["mixture"][i]), 1e-12)
            gram += scale * item["gram"][i][np.ix_(cols, cols)]
            cross += scale * item["cross"][i][cols]
    ridge = max(0.02 * gram.trace() / len(cols), 1e-12)
    coefficients = np.zeros(N_FEATURES, dtype=np.float64)
    coefficients[cols] = np.linalg.solve(gram + ridge * np.eye(len(cols)), cross)
    largest = np.abs(coefficients.reshape(4, 4, 3)).sum(axis=-1).max()
    coefficients /= max(1.0, largest / 0.25)
    return validate_weights(coefficients)
