import numpy as np
from src.kalman_filter import kalman_slope
from src.slope_signal import standardize

def test_future_perturbation_and_prefix():
    close=100*np.exp(np.sin(np.arange(600)/40)*.1)
    modified=close.copy();modified[350:]*=2
    full=kalman_slope(close);other=kalman_slope(modified)
    np.testing.assert_array_equal(full[:350],other[:350])
    np.testing.assert_array_equal(full[:350],kalman_slope(close[:350]))
    np.testing.assert_allclose(standardize(full)[:350],standardize(other)[:350],equal_nan=True,rtol=0,atol=0)

def test_current_excluded_from_std():
    v=np.arange(220,dtype=float);changed=v.copy();changed[150]+=500
    before=standardize(v);after=standardize(changed)
    np.testing.assert_allclose(after[150]/before[150],changed[150]/v[150])
