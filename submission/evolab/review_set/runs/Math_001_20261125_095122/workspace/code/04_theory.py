"""Numerical verification of the continuous and discrete strong-Lyapunov theory,
and the algebraic equivalence  VOS-AG-II  <->  Nesterov / FISTA."""
import json, pickle, time, numpy as np
from scipy.integrate import solve_ivp
from common import *

A, b, xt, meta = load_data()
R = json.load(open(f'{OUT}/reference.json'))
smax = R['facts']['smax']; lam0 = R['ref']['lam']
n = A.shape[1]
out = {}

# ------------------------------------------------------------------
# A. Continuous AG flow on ridge, random x0 (row-space pitfall avoided)
# ------------------------------------------------------------------
rng = np.random.default_rng(0)
x0r = rng.standard_normal(n)
cont = {}
for mu in [1.0, 0.1]:
    P = Problem(A, b, lam=0.0, mu=mu, smax=smax)
    xs = reference(P); P.set_ref(xs); fs = P.Fs
    gfun = lambda x: A.T @ (A @ x - b) + mu * x
    g0 = 10.0

    def rhs(t, z):
        x, y, gam = z[:n], z[n:2 * n], z[-1]
        return np.concatenate([y - x, (mu * (x - y) - gfun(x)) / gam, [mu - gam]])

    def rhs_gf(t, x):
        return -gfun(x)

    T = 30.0; tt = np.linspace(0, T, 601)
    t0 = time.time()
    sol = solve_ivp(rhs, (0, T), np.concatenate([x0r, x0r, [g0]]), method='RK45', rtol=1e-10, atol=1e-12, t_eval=tt)
    nfev = sol.nfev
    solg = solve_ivp(rhs_gf, (0, T), x0r, method='RK45', rtol=1e-10, atol=1e-12, t_eval=tt)
    E, Ed, gapx, gapgf, gamt = [], [], [], [], []
    for j in range(len(sol.t)):
        x, y, gam = sol.y[:n, j], sol.y[n:2 * n, j], sol.y[-1, j]
        gx = gfun(x)
        e = P.gap(x) + 0.5 * gam * np.sum((y - xs) ** 2)
        # analytic time derivative of E along the flow (no finite differences)
        de = gx @ (y - x) + 0.5 * (mu - gam) * np.sum((y - xs) ** 2) + (mu * (x - y) - gx) @ (y - xs)
        E.append(e); Ed.append(de + e + 0.5 * mu * np.sum((x - y) ** 2)); gapx.append(P.gap(x)); gamt.append(gam)
    for j in range(len(solg.t)):
        gapgf.append(P.gap(solg.y[:, j]))
    E, Ed = np.array(E), np.array(Ed)
    mask = E > 1e-12 * E[0]
    cont[mu] = dict(t=sol.t.tolist(), E=E.tolist(), gap=gapx, gap_gf=gapgf, gam=gamt,
                    max_logE_plus_t=float(np.max(np.log(E[mask] / E[0]) + sol.t[mask])),
                    max_rel_resid=float(np.max(Ed[mask] / E[mask])), nfev=int(nfev),
                    tmask=float(sol.t[mask][-1]), time=time.time() - t0)
    print('cont mu', mu, cont[mu]['max_logE_plus_t'], cont[mu]['max_rel_resid'], nfev, time.time() - t0, flush=True)
out['continuous'] = {str(k): v for k, v in cont.items()}

# ------------------------------------------------------------------
# B. Discrete Lyapunov:  (1+a_k) E_{k+1} <= E_k  on elastic net
# ------------------------------------------------------------------
disc = {}
for mu, K in [(1.0, 400), (0.1, 1200), (0.01, 3000)]:
    P = Problem(A, b, lam=lam0, mu=mu, smax=smax)
    xs = reference(P, K=5000); P.set_ref(xs)
    on, off, _ = P.kkt(xs)
    d = dict(kkt=max(on, off), nnz=int(np.count_nonzero(xs)))
    for g0name in ['mu', 'L']:
        g0 = mu if g0name == 'mu' else P.L
        for name, fn in [('VOS-I', vos1), ('VOS-II', vos2)]:
            P.mv = 0
            _, E, al = fn(P, x0r, K, gamma0=g0, lyap=True)
            msk = E[:-1] > 1e-12 * E[0]
            ratio = (1 + al) * E[1:] / E[:-1]
            d[f'{name}_g0={g0name}'] = dict(max_ratio=float(ratio[msk].max()), min_ratio=float(ratio[msk].min()),
                                            max_margin=float(((1 + al) * E[1:] - E[:-1])[msk].max() / E[0]),
                                            nsteps=int(msk.sum()), alpha_last=float(al[-1]),
                                            rate_last=float(1 / (1 + al[-1])), E=E[:: max(1, K // 400)].tolist(),
                                            ratio=ratio[msk].tolist())
            print(mu, g0name, name, d[f'{name}_g0={g0name}']['max_ratio'], d[f'{name}_g0={g0name}']['min_ratio'], flush=True)
    d['one_minus_sqrtq'] = float(1 - np.sqrt(mu / P.L))
    d['L'] = P.L
    disc[mu] = d
    print('rate check', mu, d['VOS-II_g0=mu']['rate_last'], d['one_minus_sqrtq'], flush=True)
out['discrete'] = {str(k): v for k, v in disc.items()}

# ------------------------------------------------------------------
# C. Equivalences
# ------------------------------------------------------------------
eq = {}
mu = 0.1
P = Problem(A, b, lam=lam0, mu=mu, smax=smax)
X1 = vos2(P, x0r, 300, gamma0=mu, keep=True)
X2 = nesterov_const(P, x0r, 300, mu)
rd = [np.linalg.norm(a - c) / np.linalg.norm(c) for a, c in zip(X1, X2)]
eq['EN_mu0.1_vos2_vs_nesterov_maxrel'] = float(max(rd))
print('VOS-II(g0=mu) vs Nesterov const momentum:', max(rd), flush=True)
# mu = 0 Lasso: VOS-II(g0=L) vs FISTA
P = Problem(A, b, lam=lam0, mu=0.0, smax=smax)
xs0 = np.load(f'{OUT}/xstar.npy')
X1 = vos2(P, np.zeros(n), 400, keep=True)
Xf = []
fista(P, np.zeros(n), 400, rec=lambda k, x: Xf.append(x.copy()))
rd = np.array([np.linalg.norm(a - c) / max(np.linalg.norm(c), 1e-300) for a, c in zip(X1, Xf)])
eq['lasso_vos2_vs_fista_reldiff'] = rd.tolist()
print('VOS-II vs FISTA rel diff at k=1,2,5,10,50,100,300:', rd[[1, 2, 5, 10, 50, 100, 300]], flush=True)
# rescaled comparison: distance of the two sequences relative to distance to x*
dd = np.array([np.linalg.norm(a - c) / max(np.linalg.norm(c - xs0), 1e-300) for a, c in zip(X1, Xf)])
eq['lasso_vos2_vs_fista_diff_over_err'] = dd.tolist()
out['equivalence'] = eq

# ------------------------------------------------------------------
# D. Overshoot on elastic net: FISTA (mu-agnostic) vs VOS-II(g0=mu) vs restart
# ------------------------------------------------------------------
ov = {}
for mu in [0.1, 0.01]:
    P = Problem(A, b, lam=lam0, mu=mu, smax=smax)
    xs = reference(P, K=5000); P.set_ref(xs)
    for name, fn in [('PG', lambda P, r: pg(P, x0r, 600, r)),
                     ('FISTA', lambda P, r: fista(P, x0r, 600, r)),
                     ('FISTA+grad-restart', lambda P, r: fista(P, x0r, 600, r, restart='grad')),
                     ('VOS-AG-II (g0=mu)', lambda P, r: vos2(P, x0r, 600, r, gamma0=mu)),
                     ('VOS-AG-II (g0=L)', lambda P, r: vos2(P, x0r, 600, r, gamma0=P.L)),
                     ('VOS-AG-I (g0=mu)', lambda P, r: vos1(P, x0r, 600, r, gamma0=mu))]:
        rec = Recorder(P, None, full=False); fn(P, rec); h = rec.out()
        g = h['gap']; inc = np.sum(np.diff(h['F']) > 1e-12 * abs(P.Fs))
        ov[f'{mu}_{name}'] = dict(gap=g.tolist(), n_increase=int(inc))
        print('overshoot', mu, name, inc, g[-1], flush=True)
out['overshoot'] = ov
json.dump(out, open(f'{OUT}/theory.json', 'w'))
print('done')
