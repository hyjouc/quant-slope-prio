import numpy as np
import pytest
from src.slope_estimators import estimate,N_GRID

@pytest.mark.parametrize('method',['A_endpoint','B_ols'])
@pytest.mark.parametrize('n',N_GRID)
def test_log_linear_exact_formula_and_window(n,method):
    y=.003*np.arange(200)+4
    beta=estimate(np.exp(y),n,method)
    assert np.isnan(beta[:n-1]).all()
    np.testing.assert_allclose(beta[n-1:],.003,atol=1e-15,rtol=0)

def test_endpoints_versus_full_ols_distinct_formula():
    y=np.array([1.,3.,2.,6.,4.,3.,7.,5.])
    np.testing.assert_allclose(estimate(np.exp(y),8,'A_endpoint')[-1],(y[-1]-y[0])/7)
    x=np.arange(8);expected=np.sum((x-x.mean())*(y-y.mean()))/np.sum((x-x.mean())**2)
    np.testing.assert_allclose(estimate(np.exp(y),8,'B_ols')[-1],expected)
    assert expected!=(y[-1]-y[0])/7

@pytest.mark.parametrize('method',['A_endpoint','B_ols'])
def test_invalid_inside_window_is_invalid_and_recovers(method):
    close=np.full(30,100.);close[12]=np.nan
    beta=estimate(close,8,method)
    assert np.isnan(beta[12:20]).all()
    assert beta[20]==0.
