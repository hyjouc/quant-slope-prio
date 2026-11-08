import numpy as np
from src.kalman_filter import kalman_slope

def test_constant_price_and_initialization():
    np.testing.assert_equal(kalman_slope(np.full(400,100.)),np.zeros(400))
    assert kalman_slope([100.,101.])[0]==0.
