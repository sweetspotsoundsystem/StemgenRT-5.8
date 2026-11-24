"""The peer-conditioned correction preserves stream timing and fits the registered loss."""
import numpy as np
import pytest

from stemgenrt.bass_peer_readout import N_FEATURES, PeerBassReadout, PeerFeatures, fit_readout


def audio(n=2048):
    rng = np.random.default_rng(173)
    x = rng.normal(0, .1, (4, 2, n)).astype(np.float32)
    return x[0], x[1:]


def test_band_sum_stereo_exchange_and_streaming_state():
    mix, stems = audio()
    design = PeerFeatures().design(mix, stems)
    bands = design.reshape(-1, 2, 4, 4, 3)
    carriers = bands[..., 0].sum(axis=2).transpose(2, 1, 0)
    expected = np.stack((stems[1], mix - ((stems[0] + stems[1]) + stems[2]), stems[0], stems[2]))
    np.testing.assert_allclose(carriers, expected, rtol=1e-6, atol=2e-7)
    w = np.linspace(-.05, .05, N_FEATURES)
    full = PeerBassReadout(w).process(mix, stems)
    chunks = PeerBassReadout(w)
    actual = np.concatenate([chunks.process(mix[:, i:i+128], stems[..., i:i+128])
                             for i in range(0, mix.shape[-1], 128)], axis=-1)
    np.testing.assert_allclose(actual, full, rtol=1e-6, atol=2e-7)
    np.testing.assert_allclose(PeerBassReadout(w).process(mix[::-1], stems[:, ::-1])[:, ::-1],
                               full, rtol=1e-6, atol=2e-7)
    np.testing.assert_array_equal(full[[0, 2]], stems[[0, 2]])
    np.testing.assert_array_equal(full[3], mix - ((stems[0] + full[1]) + stems[2]))
    identity = PeerBassReadout(np.zeros(N_FEATURES)).process(mix, stems)
    np.testing.assert_array_equal(identity[:3], stems)


def test_features_have_no_dependency_on_later_hops():
    mix, stems = audio()
    a = PeerFeatures().design(mix, stems)
    mix[:, 512:] *= -2
    stems[..., 512:] *= 3
    b = PeerFeatures().design(mix, stems)
    np.testing.assert_array_equal(a[:2*512], b[:2*512])


def test_fit_uses_targets_and_static_control_has_no_gate_coefficients():
    rng = np.random.default_rng(91)
    x = rng.normal(size=(500, N_FEATURES))
    truth = .1*x[:, 1] - .06*x[:, 3]
    artist = {"gram": np.stack([x.T @ x]*4), "cross": np.stack([x.T @ truth]*4),
              "error": np.repeat(truth @ truth, 4), "mixture": np.ones(4)}
    full = fit_readout([artist], conditional=True)
    static = fit_readout([artist], conditional=False)
    assert np.mean((x @ full - truth)**2) < .01*np.mean((x @ static - truth)**2)
    np.testing.assert_array_equal(static.reshape(-1, 3)[:, 1:], 0)
    with pytest.raises(ValueError):
        PeerBassReadout(np.ones(N_FEATURES))
    with pytest.raises(ValueError):
        PeerBassReadout(np.full(N_FEATURES, np.nan))
    with pytest.raises(ValueError):
        fit_readout([], conditional=False)
