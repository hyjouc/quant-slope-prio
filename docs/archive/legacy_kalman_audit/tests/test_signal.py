import numpy as np
from src.slope_signal import standardize, signal

def test_signal_direction_boundaries_and_invalid():
    np.testing.assert_equal(signal([1.,.5,0.,-.5,-1.,np.nan,np.inf],.5),[1.,0.,0.,0.,-1.,np.nan,np.nan])

def test_standardization_prior96_sample_no_demean():
    v=np.arange(110,dtype=float)+10
    s=standardize(v)
    assert np.isnan(s[:96]).all()
    for i in range(96,len(v)):
        np.testing.assert_allclose(s[i],v[i]/np.std(v[i-96:i],ddof=1),rtol=1e-12)
    assert np.isnan(standardize(np.ones(110))).all()
