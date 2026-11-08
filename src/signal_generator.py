"""Strict sign; exact zero carries previous signal; invalid means no order."""
import numpy as np

def generate_signals(beta):
    out=np.full(len(beta),np.nan);previous=0
    for i,b in enumerate(beta):
        if not np.isfinite(b):continue
        if b>0:previous=1
        elif b<0:previous=-1
        out[i]=previous
    return out
