"""Locate where the (near-tight) VOS-I discrete Lyapunov ratio is largest, to separate
rounding from genuine violation."""
import json, numpy as np
from common import *
A, b, xt, meta = load_data()
R = json.load(open(f'{OUT}/reference.json'))
n = A.shape[1]; x0r = np.random.default_rng(0).standard_normal(n)
res = {}
for mu in [1.0, 0.1]:
    P = Problem(A, b, lam=R['ref']['lam'], mu=mu, smax=R['facts']['smax'])
    xs = reference(P, K=5000); P.set_ref(xs)
    _, E, al = vos1(P, x0r, 400 if mu == 1 else 1200, gamma0=mu, lyap=True)
    ratio = (1 + al) * E[1:] / E[:-1]
    for thr in [1e-12, 1e-8, 1e-6, 1e-4]:
        m = E[:-1] > thr * E[0]
        j = np.argmax(np.where(m, ratio, -np.inf))
        res[f'{mu}_{thr:.0e}'] = dict(max_ratio_minus_1=float(ratio[j] - 1), at_k=int(j), E_rel=float(E[j] / E[0]))
        print(mu, thr, res[f'{mu}_{thr:.0e}'], flush=True)
    res[f'{mu}_E0'] = float(E[0])
json.dump(res, open(f'{OUT}/vos1_tightness.json', 'w'), indent=1)
