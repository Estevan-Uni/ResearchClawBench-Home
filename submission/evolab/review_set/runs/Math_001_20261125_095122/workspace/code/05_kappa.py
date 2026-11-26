"""kappa-sweep: iterations to reach relative gap 1e-8 as a function of kappa = L/mu,
ridge (lam=0) and elastic net (lam=0.01 lam_max), random x0 (seed 1)."""
import json, numpy as np
from common import *

A, b, xt, meta = load_data()
R = json.load(open(f'{OUT}/reference.json'))
smax = R['facts']['smax']; lam0 = R['ref']['lam']
n = A.shape[1]
x0 = np.random.default_rng(1).standard_normal(n)
TOL = 1e-8


class Stop(Exception):
    pass


def iters_to_tol(P, fn, Kmax):
    g0 = P.gap(x0)
    hit = {}
    def rec(k, x):
        if P.gap(x) <= TOL * g0:
            hit['k'] = k; raise Stop
    try:
        fn(P, x0, Kmax, rec)
    except Stop:
        return hit['k']
    return None


out = {}
for kind, mus in [('ridge', [1.0, 0.3, 0.1, 0.03, 0.01, 0.003, 0.001]), ('elastic', [1.0, 0.1, 0.01, 0.001])]:
    for mu in mus:
        lam = 0.0 if kind == 'ridge' else lam0
        P = Problem(A, b, lam=lam, mu=mu, smax=smax)
        xs = reference(P, K=6000); P.set_ref(xs)
        kap = P.L / mu
        r = dict(kappa=kap)
        # PG: run if the predicted count (worst mode, ln(1/tol)/(2 mu/L)) is affordable
        pred_pg = np.log(1 / TOL) / (-2 * np.log(1 - mu / P.L))
        r['pg_pred'] = float(pred_pg)
        r['PG'] = iters_to_tol(P, pg, 12000) if (pred_pg < 10000 or kind == 'elastic') else None
        r['FISTA'] = iters_to_tol(P, lambda P, x, K, rc: fista(P, x, K, rc), 20000)
        r['VOS-I'] = iters_to_tol(P, lambda P, x, K, rc: vos1(P, x, K, rc, gamma0=mu), 20000)
        r['VOS-II'] = iters_to_tol(P, lambda P, x, K, rc: vos2(P, x, K, rc, gamma0=mu), 20000)
        out[f'{kind}_{mu}'] = r
        print(kind, mu, r, flush=True)
        json.dump(out, open(f'{OUT}/kappa.json', 'w'), indent=1)
print('done')
