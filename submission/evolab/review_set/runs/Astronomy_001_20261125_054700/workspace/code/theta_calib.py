"""Calibrate the mock-chain correlation matrix so that the implied scatter of 100*theta_* (CAMB)
matches the CMB precision, while keeping marginal sigmas fixed (unit diagonal).
Iterates: C <- normalise( C - C g g^T C / (g^T C g + s^2) ), g = d(100θ*)/d(z_i) in sigma units."""
import numpy as np
from cosmo import background

SIG_THETA = 0.0003  # Planck/ACT-level precision on 100 θ*

def theta_grad(model, params):
    names = list(params)
    p0 = {k: v[0] for k, v in params.items()}
    g = np.zeros(len(names))
    for i, k in enumerate(names):
        if k in ('sigma8', 'ns', 'ln10As', 'tau'):
            continue  # do not affect background θ*
        h = 0.3 * params[k][1]
        pp = dict(p0); pp[k] += h; pm = dict(p0); pm[k] -= h
        if k == 'f_EDE': pm[k] = max(pm[k], 1e-4)
        tp = background(pp, model)[1]['thetastar']; tm = background(pm, model)[1]['thetastar']
        g[i] = (tp - tm) / (pp[k] - pm[k]) * params[k][1]
    return g

def calibrate(C, g, s=SIG_THETA, alpha=0.05, maxit=5000, eig_floor=1e-3):
    """Damped conditioning; stops once sqrt(g^T C g) <= s. Keeps C positive definite."""
    for _ in range(maxit):
        if np.sqrt(g @ C @ g) <= s: break
        Cg = C @ g
        C = C - alpha * np.outer(Cg, Cg) / (g @ Cg + s**2)
        w, V = np.linalg.eigh(C); w = np.clip(w, eig_floor, None); C = V @ np.diag(w) @ V.T
        dd = np.sqrt(np.diag(C)); C = C / np.outer(dd, dd)
    return C
