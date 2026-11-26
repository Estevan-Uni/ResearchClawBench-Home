"""Robustness of the headline comparison: other lambda values and a random starting point."""
import json, numpy as np
from common import *

A, b, xt, meta = load_data()
R = json.load(open(f'{OUT}/reference.json'))
smax = R['facts']['smax']; lmax = R['facts']['lam_max']
n = A.shape[1]; K = 1000
out = {}
for frac, x0name in [(0.005, 'zero'), (0.01, 'zero'), (0.02, 'zero'), (0.05, 'zero'), (0.1, 'zero'), (0.01, 'random')]:
    x0 = np.zeros(n) if x0name == 'zero' else np.random.default_rng(2).standard_normal(n)
    P = Problem(A, b, lam=frac * lmax, smax=smax)
    xs = reference(P, K=4000); P.set_ref(xs)
    S = xs != 0; ev = np.linalg.eigvalsh(A[:, S].T @ A[:, S])
    g0 = P.gap(x0)
    r = dict(nnz=int(S.sum()), muS=float(ev.min()), kkt=float(max(P.kkt(xs)[:2])))
    adm = ADMMSolver(P)
    for nm, fn in [('PG', lambda P, rc: pg(P, x0, K, rc)), ('FISTA', lambda P, rc: fista(P, x0, K, rc)),
                   ('FISTA+grad-restart', lambda P, rc: fista(P, x0, K, rc, restart='grad')),
                   ('VOS-AG-II', lambda P, rc: vos2(P, x0, K, rc)),
                   ('VOS-AG-II+restart', lambda P, rc: vos2(P, x0, K, rc, restart='fun')),
                   ('ADMM(beta=sqrt(muS LS))', lambda P, rc: adm.run(np.sqrt(ev.min() * ev.max()), K, x0=x0, rec=rc))]:
        rec = Recorder(P, None, full=False); fn(P, rec); h = rec.out()
        r[nm] = dict(it_1e6=first_hit(h['k'], h['gap'] / g0, 1e-6), it_1e10=first_hit(h['k'], h['gap'] / g0, 1e-10))
    out[f'{frac}_{x0name}'] = r
    print(frac, x0name, r, flush=True)
json.dump(out, open(f'{OUT}/lasso_robustness.json', 'w'), indent=1)
print('done')
