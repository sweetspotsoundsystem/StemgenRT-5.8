import numpy as np
import pytest
from stemgenrt import teacher_frequency_projection as projection


def test_orthogonal_bands_reproduce_stereo_error_and_constant_projections():
    rng=np.random.default_rng(21)
    raw=rng.normal(size=(2,507)).astype(np.float32)
    bass=rng.normal(size=raw.shape).astype(np.float32)
    want=rng.normal(size=raw.shape)
    gram,cross=projection.clip_statistics(raw,want,-want)
    np.testing.assert_allclose(gram.sum(),np.square(raw,dtype=np.float64).sum(),rtol=1e-12)
    np.testing.assert_allclose(cross[0].sum(),(raw*want).sum(),rtol=1e-12)
    w=np.array([.1,.2,.9,.8,.05,.97,.1])
    pred=projection.projected_bass(bass,raw,w)
    measured=np.square((pred.astype(float)-bass)-want).sum()
    predicted=np.square(want).sum()+(w*w*gram-2*w*cross[0]).sum()
    np.testing.assert_allclose(measured,predicted,rtol=1e-7)
    assert np.array_equal(projection.projected_bass(bass,raw,np.zeros(7)),bass)
    np.testing.assert_array_equal(projection.projected_bass(bass,raw,np.full(7,.5)),bass+.5*raw)
    with pytest.raises(ValueError):
        projection.projected_bass(bass,raw,np.ones(6))


def test_constrained_fitting_spends_its_budget_and_preserves_band_identity():
    g=np.array([100.,150.,80.,100.,100.,80.,80.])
    needed=np.array([0.,0.,.1,.3,.8,.9,1.])
    rows=[{"root":k,"active":[1,1],"mixture_energy":1000.,"gram":g,
           "cross":np.stack((np.zeros(7),g*needed)),"direct_error":[200.,600.]} for k in ("a","b")]
    all_bands=projection.fit_projection(rows,scalar=False)
    one=projection.fit_projection(rows,scalar=True)
    for result in (all_bands,one):
        assert result["dual_multiplier"]>0
        np.testing.assert_allclose(result["change_in_normalized_errors"][0],result["allowed_bass_error_increase"],rtol=1e-9)
    assert np.std(one["weights"])<1e-15 and all_bands["weights"][0]==0
    assert all_bands["change_in_normalized_errors"][1]<one["change_in_normalized_errors"][1]


def test_degenerate_silent_gram_has_finite_zero_correction():
    rows=[{"root":k,"active":[1,1],"mixture_energy":0.,"gram":np.zeros(7),
           "cross":np.zeros((2,7)),"direct_error":[0.,0.]} for k in ("a","b")]
    for scalar in (False,True):
        assert projection.fit_projection(rows,scalar=scalar)["weights"]==[0.]*7
    perfect=[{**r,"gram":np.ones(7),"cross":np.stack((np.zeros(7),np.ones(7))),
              "direct_error":[0.,10.]} for r in rows]
    assert projection.fit_projection(perfect,scalar=False)["weights"]==[0.]*7
    with pytest.raises(ValueError):
        projection.fit_projection(rows[:1],scalar=False)
