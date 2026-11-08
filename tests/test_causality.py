import numpy as np
import pandas as pd
import pytest
from src.slope_estimators import estimate,N_GRID

@pytest.mark.parametrize('method',['A_endpoint','B_ols'])
@pytest.mark.parametrize('n',N_GRID)
def test_future_perturbation_and_exact_prefix(n,method):
    close=np.exp(4+.01*np.sin(np.arange(700)/11))
    changed=close.copy();changed[400:]*=1.8
    full=estimate(close,n,method)
    np.testing.assert_allclose(full[:400],estimate(close[:400],n,method),equal_nan=True,rtol=0,atol=0)
    np.testing.assert_allclose(full[:400],estimate(changed,n,method)[:400],equal_nan=True,rtol=0,atol=0)

def test_year_boundary_uses_preceding_history():
    date=pd.date_range('2020-12-31 12:00',periods=110,freq='15min',tz='UTC')
    price=np.exp(4+.002*np.arange(110))
    boundary=int(date.searchsorted(pd.Timestamp('2021-01-01',tz='UTC')))
    for method in ['A_endpoint','B_ols']:
        beta=estimate(price,32,method)
        assert np.isfinite(beta[boundary])
        np.testing.assert_allclose(beta[boundary],.002,atol=1e-15)
