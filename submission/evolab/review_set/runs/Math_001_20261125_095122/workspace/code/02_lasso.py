"""Headline Lasso experiment: VOS accelerated methods vs proximal gradient (and ADMM)."""
import json, time, pickle, numpy as np
from common import *

A, b, xt, meta = load_data()
R = json.load(open(f'{OUT}/reference.json'))
lam = R['ref']['lam']; smax = R['facts']['smax']
xs = np.load(f'{OUT}/xstar.npy')
n = A.shape[1]; x0 = np.zeros(n); K = 800
muS, LS = R['ref']['mu_S'], R['ref']['L_S']


def newP():
    P = Problem(A, b, lam=lam, smax=smax); P.set_ref(xs); return P


H = {}; T = {}; NR = {}
def run(name, fn):
    P = newP(); rec = Recorder(P, xt); t = time.time()
    fn(P, rec); T[name] = time.time() - t
    H[name] = rec.out(); NR[name] = getattr(P, 'nrestart', None)
    g = H[name]['gap']
    print(f'{name:28s} final gap {g[-1]:.2e}  mv {H[name]["mv"][-1]}  t {T[name]:.1f}s  restarts {NR[name]}', flush=True)

run('PG', lambda P, r: pg(P, x0, K, r))
run('FISTA', lambda P, r: fista(P, x0, K, r))
run('FISTA+grad-restart', lambda P, r: fista(P, x0, K, r, restart='grad'))
run('VOS-AG-I', lambda P, r: vos1(P, x0, K, r))
run('VOS-AG-I+restart', lambda P, r: vos1(P, x0, K, r, restart='fun'))
run('VOS-AG-II', lambda P, r: vos2(P, x0, K, r))
run('VOS-AG-II+restart', lambda P, r: vos2(P, x0, K, r, restart='fun'))
run('VOS-AG-II monotone', lambda P, r: vos2(P, x0, K, r, monotone=True))
run('VOS-AG-II(mu=muS)+restart', lambda P, r: vos2(P, x0, K, r, mu=muS, restart='fun'))

# ---- ADMM: beta sweep (gap only) then full diagnostics at best beta ----
P0 = newP(); adm = ADMMSolver(P0)
betas = np.logspace(-1, 2.5, 15)
sweep = {}
for be in betas:
    P = newP(); adm.P = P; rec = Recorder(P, None, full=False)
    adm.run(be, K, rec=rec); h = rec.out()
    rel = h['gap'] / (R['facts']['F0'] - P.Fs)
    sweep[float(be)] = dict(it_1e6=first_hit(h['k'], rel, 1e-6), it_1e10=first_hit(h['k'], rel, 1e-10),
                            final_gap=float(h['gap'][-1]), gap=h['gap'].tolist())
    print('beta', be, sweep[float(be)]['it_1e6'], sweep[float(be)]['it_1e10'], flush=True)
ok = [be for be in sweep if sweep[be]['it_1e10'] is not None]
best_beta = min(ok, key=lambda be: sweep[be]['it_1e10'])
print('best beta', best_beta, 'sqrt(muS LS)', np.sqrt(muS * LS))
for name, fast in [('ADMM', False), ('Fast ADMM (restart)', True)]:
    P = newP(); adm.P = P; rec = Recorder(P, xt); t = time.time()
    adm.run(best_beta, K, rec=rec, fast=fast); T[name] = time.time() - t
    H[name] = rec.out(); print(name, H[name]['gap'][-1], flush=True)
# ADMM at the curvature-predicted beta
P = newP(); adm.P = P; rec = Recorder(P, xt)
adm.run(np.sqrt(muS * LS), K, rec=rec); H['ADMM (beta=sqrt(muS LS))'] = rec.out()

pickle.dump(dict(H=H, T=T, NR=NR, sweep=sweep, best_beta=best_beta, betas=betas.tolist()),
            open(f'{OUT}/lasso_hist.pkl', 'wb'))
print('done')
