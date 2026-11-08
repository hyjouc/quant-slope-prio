"""Exact V18 forward Joseph filter; first observation updates the initialized prior."""
import numpy as np

def kalman_slope(close):
    y = 100 * np.log(np.asarray(close, dtype=float))
    if len(y) == 0 or not np.isfinite(y).all():
        raise ValueError('Need finite positive closes')
    dt = .25; R = .01
    F = np.array([[1.,dt,.5*dt**2],[0.,1.,dt],[0.,0.,1.]])
    H = np.array([[1.,0.,0.]])
    Q = 1e-7 * np.array([[dt**5/20,dt**4/8,dt**3/6],
                         [dt**4/8,dt**3/3,dt**2/2],
                         [dt**3/6,dt**2/2,dt]])
    x = np.array([y[0],0.,0.]); P = np.diag([R,1.,.1]); I = np.eye(3)
    v = np.empty(len(y))
    for i, obs in enumerate(y):
        if i:
            x = F @ x; P = F @ P @ F.T + Q
        innovation = obs - (H @ x)[0]
        innovation_var = (H @ P @ H.T)[0,0] + R
        K = (P @ H.T / innovation_var).ravel()
        x = x + K * innovation
        residual = I - np.outer(K, H.ravel())
        P = residual @ P @ residual.T + np.outer(K,K) * R
        v[i] = x[1]
    return v
