import numpy as np
import pandas as pd

def standardize(v):
    v = pd.Series(v, dtype=float)
    sd = v.shift(1).rolling(96, min_periods=96).std(ddof=1)
    return (v / sd.where(sd > 0)).to_numpy()

def signal(slope, theta):
    if not np.isfinite(theta) or theta < 0:
        raise ValueError('theta must be finite and nonnegative')
    s = np.asarray(slope, dtype=float)
    return np.where(np.isfinite(s), np.where(s > theta, 1., np.where(s < -theta, -1., 0.)), np.nan)
