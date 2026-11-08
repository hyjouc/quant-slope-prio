"""Log-price change per bar. Exactly one parameter N; no normalization/smoothing."""
import numpy as np

N_GRID=(8,16,32,64,96)

def estimate(close,n,method):
    if not isinstance(n,(int,np.integer)) or n<2:raise ValueError('N must be an integer >=2')
    close=np.asarray(close,dtype=float)
    if close.ndim!=1:raise ValueError('Expected one-dimensional closes')
    y=np.full(len(close),np.nan)
    valid=np.isfinite(close)&(close>0)
    y[valid]=np.log(close[valid])
    out=np.full(len(y),np.nan)
    if method not in ('A_endpoint','B_ols'):raise ValueError('Unknown estimator')
    if len(y)<n:return out
    windows=np.lib.stride_tricks.sliding_window_view(y,n)
    good=np.isfinite(windows).all(axis=1)
    if method=='A_endpoint':beta=(windows[:,-1]-windows[:,0])/(n-1)
    else:
        x=np.arange(n,dtype=float)-(n-1)/2
        # Subtract window first value for numerical stability; sum(x)=0 so the
        # estimator is algebraically identical to centered OLS. Per-window sum
        # keeps prefix results independent of input length / BLAS dispatch.
        beta=np.sum((windows-windows[:,:1])*x,axis=1)/np.dot(x,x)
    out[n-1:]=np.where(good,beta,np.nan)
    return out
