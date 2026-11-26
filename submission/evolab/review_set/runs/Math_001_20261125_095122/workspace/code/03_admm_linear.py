"""ADMM Lyapunov function V_k = beta||u-u*||^2 + beta||z-z*||^2 and its linear rate vs beta;
comparison with the spectral radius of the ADMM (Douglas-Rachford) operator linearised on the
identified support, and with predicted local rates of PG / accelerated methods."""
import json, pickle, numpy as np, scipy.linalg as sla
from common import *

A, b, xt, meta = load_data()
R = json.load(open(f'{OUT}/reference.json'))
lam = R['ref']['lam']; smax = R['facts']['smax']
xs = np.load(f'{OUT}/xstar.npy')
n = A.shape[1]; m = A.shape[0]
P = Problem(A, b, lam=lam, smax=smax); P.set_ref(xs)
S = xs != 0; sgn = np.sign(xs)
grs = A.T @ (A @ xs - b)
AAt = A @ A.T
adm = ADMMSolver(P, AAt)
D = S.astype(float)

betas = np.logspace(-1, 2.5, 15)
res = {}
K = 1500
for be in betas:
    us = -grs / be; zs = xs
    P.mv = 0
    z, u, V = adm.run(be, K, lyap=True, zs=zs, us=us)
    V = np.array(V)
    mono = bool(np.all(np.diff(V) <= 1e-12 * V[0]))
    # empirical linear rate: fit in window V in [1e-22, 1e-6] * V0 (asymptotic regime)
    w = np.nonzero((V > 1e-22 * V[0]) & (V < 1e-6 * V[0]))[0]
    rate = float(np.exp(np.polyfit(w, np.log(V[w]), 1)[0])) if len(w) > 5 else np.nan
    d = dict(monotone=mono, max_increase=float(np.max(np.diff(V)) / V[0]), rate_V=rate, nfit=int(len(w)),
             V=V[::5].tolist())
    # linearised operator on the identified support: w+ = T w, T = (I-D) + (2D-I) beta M
    if 0.9 <= be <= 200:
        cf = sla.cho_factor(AAt + be * np.eye(m))
        bM = np.eye(n) - A.T @ sla.cho_solve(cf, A)          # beta (A^T A + beta I)^{-1}
        Tm = (2 * D - 1)[:, None] * bM
        Tm[np.diag_indices(n)] += (1 - D)
        ev = np.linalg.eigvals(Tm)
        d['rho_T'] = float(np.max(np.abs(ev)))
    res[float(be)] = d
    print(f'beta {be:8.3f}  monotone {mono}  rate_V {rate:.4f}  rho_T^2 {d.get("rho_T", np.nan) ** 2:.4f}', flush=True)

# predicted local rates for prox-gradient type methods on the support
ev = np.linalg.eigvalsh(A[:, S].T @ A[:, S])
L = P.L; muS, LS = ev.min(), ev.max()
rho_pg = float(np.max(np.abs(1 - ev / L)))
q = muS / L; bet = (1 - np.sqrt(q)) / (1 + np.sqrt(q))
rr = []
for lmb in ev:
    c = 1 - lmb / L
    rr.append(np.max(np.abs(np.roots([1, -(1 + bet) * c, bet * c]))))
rho_nest = float(max(rr))
# ADMM best possible for quadratic with spectrum [muS, LS] (Ghadimi et al.-type optimum)
pred = dict(rho_pg=rho_pg, gap_rate_pg=rho_pg ** 2, rho_nest_muS=rho_nest, gap_rate_nest_muS=rho_nest ** 2,
            one_minus_sqrt_q=float(1 - np.sqrt(q)), muS=float(muS), LS=float(LS), L=float(L),
            sqrt_muS_LS=float(np.sqrt(muS * LS)))
print(pred)
json.dump(dict(res={str(k): v for k, v in res.items()}, pred=pred), open(f'{OUT}/admm_linear.json', 'w'))
print('done')
