"""EWLS log-price slope, fixed 64-bar window; sole new research parameter H."""
import numpy as np

N=64

def weights(half_life):
    if half_life is None or np.isposinf(half_life):return np.ones(N)
    if not np.isfinite(half_life) or half_life<=0:raise ValueError('H must be positive or +infinity')
    return np.exp2(-(N-1-np.arange(N,dtype=float))/float(half_life))

def ewls_log(y,half_life):
    y=np.asarray(y,dtype=float)
    if y.ndim!=1:raise ValueError('Expected one-dimensional log price')
    out=np.full(len(y),np.nan)
    if len(y)<N:return out
    w=weights(half_life);sw=w.sum();x=np.arange(N,dtype=float);xb=np.sum(w*x)/sw
    dx=x-xb;den=np.sum(w*dx*dx)
    win=np.lib.stride_tricks.sliding_window_view(y,N)
    good=np.isfinite(win).all(axis=1)
    z=win-win[:,:1]
    if half_life is None or np.isposinf(half_life):
        # Equal-weight centered regression. Sum(dx)==0 exactly here, so the y
        # centering cancels algebraically. Same independent per-window sum as S0,
        # without calling S0 estimator: permits an independent exact equality gate.
        beta=np.sum(z*dx,axis=1)/den
    else:
        yb=np.sum(z*w,axis=1)/sw
        beta=np.sum((z-yb[:,None])*w*dx,axis=1)/den
    out[N-1:]=np.where(good,beta,np.nan)
    return out

def estimate_ewls(close,half_life):
    close=np.asarray(close,dtype=float);y=np.full(len(close),np.nan)
    good=np.isfinite(close)&(close>0);y[good]=np.log(close[good])
    return ewls_log(y,half_life)
