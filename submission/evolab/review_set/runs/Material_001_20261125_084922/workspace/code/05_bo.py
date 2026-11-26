"""Autonomous optimisation of synthesis temperature T (°C) and time t (min) with Gaussian-process
Bayesian optimisation (numpy/scipy GP, ARD-RBF, EI and UCB), vs random / LHS / grid baselines.

Simulated experiment (assumed form; the file only gives bounds, optimum, noise and n_iter):
  yield(T,t) = exp(-½[((T-T*)/50)^2 + ((t-t*)/5)^2])  (fraction, max 1 = 100 %)
  observation = yield + N(0, 0.1^2)                    (noise sd 0.1 from the file = 10 % points)
Budget: 5 LHS initial + 20 sequential = 25 experiments; the file's n_iter=10 is marked."""
import os, sys, json
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm, wilcoxon
from common import load_dataset, save_json, style, IMG, OUT, C

d = load_dataset()
LO = np.array([d['T_range'][0], d['t_range'][0]]); HI = np.array([d['T_range'][1], d['t_range'][1]])
OPT0 = np.array([d['T_opt'], d['t_opt']]); NOISE0 = d['noise']; NITER = d['n_iter']
WID = np.array([50.0, 5.0]); N_INIT, N_SEQ = 5, 20; NSEED = int(os.environ.get('NSEED', 20)); LS_MIN = 0.1


def yield_true(X, opt):
    return np.exp(-0.5 * (((X - opt) / WID) ** 2).sum(-1))


def lhs(n, rng):
    u = (np.stack([rng.permutation(n) for _ in range(2)], 1) + rng.random((n, 2))) / n
    return LO + u * (HI - LO)


class GP:
    """ARD-RBF GP on unit-scaled inputs, standardised outputs; hyper-parameters by max marginal likelihood."""
    def fit(self, U, y, rng):
        self.U = U; self.m, self.s = y.mean(), y.std() + 1e-9; self.y = (y - self.m) / self.s
        bounds = [(np.log(LS_MIN), np.log(2.0))] * 2 + [(np.log(0.05), np.log(5.0)), (np.log(1e-3), np.log(1.0))]
        best = None
        for _ in range(3):
            x0 = [rng.uniform(*b) for b in bounds]
            r = minimize(self.nll, x0, bounds=bounds, method='L-BFGS-B')
            if best is None or r.fun < best.fun: best = r
        self.th = best.x; self._prep(); return self

    def K(self, A, B, th):
        ls, sf = np.exp(th[:2]), np.exp(th[2])
        return sf * np.exp(-0.5 * (((A[:, None] - B[None]) / ls) ** 2).sum(-1))

    def nll(self, th):
        K = self.K(self.U, self.U, th) + (np.exp(th[3]) + 1e-8) * np.eye(len(self.U))
        try: L = np.linalg.cholesky(K)
        except np.linalg.LinAlgError: return 1e10
        a = np.linalg.solve(L.T, np.linalg.solve(L, self.y))
        return 0.5 * self.y @ a + np.log(np.diag(L)).sum()

    def _prep(self):
        K = self.K(self.U, self.U, self.th) + (np.exp(self.th[3]) + 1e-8) * np.eye(len(self.U))
        self.L = np.linalg.cholesky(K); self.a = np.linalg.solve(self.L.T, np.linalg.solve(self.L, self.y))

    def predict(self, Us):
        Ks = self.K(Us, self.U, self.th); mu = Ks @ self.a
        v = np.linalg.solve(self.L, Ks.T); var = np.maximum(np.exp(self.th[2]) - (v ** 2).sum(0), 1e-12)
        return mu * self.s + self.m, np.sqrt(var) * self.s


def run(method, seed, opt, noise):
    rng = np.random.default_rng(1000 * seed + 7)
    obs = lambda X: yield_true(X, opt) + rng.normal(0, noise, len(X))
    N = N_INIT + N_SEQ
    if method == 'Random':
        X = LO + rng.random((N, 2)) * (HI - LO)
    elif method == 'LHS':
        X = lhs(N, rng)
    elif method == 'Grid 5x5':
        g = np.stack(np.meshgrid(np.linspace(LO[0], HI[0], 5), np.linspace(LO[1], HI[1], 5)), -1).reshape(-1, 2)
        X = g[rng.permutation(25)]
    else:
        X = lhs(N_INIT, rng)
    Y = obs(X)
    rec = []; gp = None
    if method.startswith('GP'):
        for k in range(N_INIT, N + 1):
            U = (X - LO) / (HI - LO); gp = GP().fit(U, Y, rng)
            mu_obs, _ = gp.predict(U); rec.append(X[np.argmax(mu_obs)])    # recommend max posterior mean
            if k == N: break
            Uc = rng.random((3000, 2)); mu, sd = gp.predict(Uc)
            if method == 'GP-EI':
                imp = mu - mu_obs.max(); zz = imp / sd; acq = imp * norm.cdf(zz) + sd * norm.pdf(zz)
            else:
                acq = mu + 2.0 * sd
            xn = LO + Uc[np.argmax(acq)] * (HI - LO)
            X = np.vstack([X, xn]); Y = np.r_[Y, obs(xn[None])]
    else:
        for k in range(N_INIT, N + 1):
            rec.append(X[:k][np.argmax(Y[:k])])                              # best observed
    rec = np.array(rec)     # recommendations after n = 5..25 experiments
    return dict(X=X, Y=Y, rec=rec, rec_yield=yield_true(rec, opt), gp=gp)


METHODS = ['GP-EI', 'GP-UCB', 'Random', 'LHS', 'Grid 5x5']
SCEN = {'nominal (optimum 350 °C/20 min, noise 0.1)': ('fixed', NOISE0),
        'randomised optimum, noise 0.1': ('random', NOISE0),
        'randomised optimum, noise 0.02': ('random', 0.02)}
if __name__ == '__main__':
    out = {}; store = {}
    for sname, (mode, noise) in SCEN.items():
        out[sname] = {}; store[sname] = {}
        for m in METHODS:
            ys, recs, dists, hit_iter = [], [], [], []
            for s in range(NSEED):
                orng = np.random.default_rng(555 + s)
                opt = OPT0 if mode == 'fixed' else LO + (0.1 + 0.8 * orng.random(2)) * (HI - LO)
                r = run(m, s, opt, noise)
                ys.append(r['rec_yield']); recs.append(r['rec'][-1] - opt)
                ok = np.where(r['rec_yield'] >= 0.95)[0]
                hit_iter.append(int(N_INIT + ok[0]) if len(ok) else None)
                if s == 0: store[sname][m] = r
            ys = np.array(ys); recs = np.array(recs)
            out[sname][m] = dict(curve_median=np.median(ys, 0).tolist(), curve_q25=np.percentile(ys, 25, 0).tolist(),
                                 curve_q75=np.percentile(ys, 75, 0).tolist(), final_yield=ys[:, -1].tolist(),
                                 yield_at_10=ys[:, 10 - N_INIT].tolist(), final_offset=recs.tolist(),
                                 frac_ge95_at25=float(np.mean(ys[:, -1] >= 0.95)), frac_ge95_at10=float(np.mean(ys[:, 10 - N_INIT] >= 0.95)),
                                 first_n_ge95=hit_iter, median_abs_dT=float(np.median(np.abs(recs[:, 0]))), median_abs_dt=float(np.median(np.abs(recs[:, 1]))))
            print(sname, m, f"final median yield {np.median(ys[:, -1]):.3f}  ≥95%: {out[sname][m]['frac_ge95_at25']:.2f}  |dT| {out[sname][m]['median_abs_dT']:.1f}  |dt| {out[sname][m]['median_abs_dt']:.2f}", flush=True)
        # Wilcoxon paired tests on final true yield
        out[sname]['wilcoxon'] = {}
        for a in ['GP-EI', 'GP-UCB']:
            for b in ['Random', 'LHS', 'Grid 5x5']:
                x, y = np.array(out[sname][a]['final_yield']), np.array(out[sname][b]['final_yield'])
                try: pval = float(wilcoxon(x, y, alternative='greater').pvalue)
                except ValueError: pval = None
                out[sname]['wilcoxon'][f'{a} > {b}'] = pval
        print(out[sname]['wilcoxon'])
    # nominal seed-0 GP-EI trace for the report
    r = store[list(SCEN)[0]]['GP-EI']
    out['nominal_seed0_GP-EI'] = dict(X=r['X'].tolist(), Y=r['Y'].tolist(), rec_final=r['rec'][-1].tolist(),
                                      rec_yield=r['rec_yield'].tolist(), lengthscales_unit=np.exp(r['gp'].th[:2]).tolist())
    out['setup'] = dict(n_seeds=NSEED, n_init=N_INIT, n_seq=N_SEQ, widths=WID.tolist(), ls_min=LS_MIN, candidates=3000, ucb_beta=2.0)
    save_json(out, 'bo_results.json')

    plt = style()
    n = np.arange(N_INIT, N_INIT + N_SEQ + 1)
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2), sharey=True)
    for k, sname in enumerate(SCEN):
        for m, cc in zip(METHODS, C):
            o = out[sname][m]
            ax[k].plot(n, 100 * np.array(o['curve_median']), c=cc, label=m, lw=2 if m.startswith('GP') else 1.2)
            ax[k].fill_between(n, 100 * np.array(o['curve_q25']), 100 * np.array(o['curve_q75']), color=cc, alpha=0.12)
        ax[k].axvline(NITER, c='k', ls=':', lw=1); ax[k].text(NITER + 0.2, 8, 'n_iter=10\n(dataset)', fontsize=7)
        ax[k].axhline(95, c='grey', ls='--', lw=0.8)
        ax[k].set(xlabel='number of experiments', title=f'({"abc"[k]}) {sname}', ylim=(0, 102))
    ax[0].set_ylabel('true yield at recommended (T,t) (%)'); ax[0].legend(fontsize=8, loc='lower right')
    fig.tight_layout(); fig.savefig(os.path.join(IMG, 'fig7_bo_convergence.png'))

    # contour of nominal run
    r = store[list(SCEN)[0]]['GP-EI']
    TT, tt = np.meshgrid(np.linspace(LO[0], HI[0], 120), np.linspace(LO[1], HI[1], 120))
    G = np.c_[TT.ravel(), tt.ravel()]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    cs = ax[0].contourf(TT, tt, 100 * yield_true(G, OPT0).reshape(TT.shape), 20, cmap='viridis'); fig.colorbar(cs, ax=ax[0], label='true yield (%)')
    mu, sd = r['gp'].predict((G - LO) / (HI - LO))
    cs2 = ax[1].contourf(TT, tt, 100 * mu.reshape(TT.shape), 20, cmap='viridis'); fig.colorbar(cs2, ax=ax[1], label='GP posterior mean (%)')
    for a_ in ax:
        a_.scatter(*r['X'][:N_INIT].T, c='w', edgecolor='k', s=40, marker='s', label='initial LHS (5)')
        sc = a_.scatter(*r['X'][N_INIT:].T, c=np.arange(N_SEQ), cmap='autumn', edgecolor='k', s=40, label='BO queries (20)')
        a_.scatter(*OPT0, marker='+', c='k', s=200, lw=2, label='true optimum (350 °C, 20 min)')
        a_.scatter(*r['rec'][-1], marker='*', c='cyan', edgecolor='k', s=300, label=f"recommended ({r['rec'][-1][0]:.0f} °C, {r['rec'][-1][1]:.1f} min)")
        a_.set(xlabel='temperature T (°C)', ylabel='time t (min)')
    ax[0].set_title('(a) True yield surface and GP-EI samples (seed 0)'); ax[1].set_title('(b) GP posterior mean after 25 experiments')
    ax[0].legend(fontsize=7, loc='upper left'); fig.colorbar(sc, ax=ax[1], label='BO iteration', pad=0.12)
    fig.tight_layout(); fig.savefig(os.path.join(IMG, 'fig8_bo_contour.png'))

    # final offsets scatter
    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    for k, sname in enumerate(SCEN):
        for m, cc in zip(METHODS, C):
            off = np.array(out[sname][m]['final_offset']); ax[k].scatter(*off.T, c=cc, s=15, alpha=0.7, label=m)
        ax[k].add_patch(plt.matplotlib.patches.Ellipse((0, 0), 2 * 50 * np.sqrt(2 * np.log(1 / 0.95)) , 2 * 5 * np.sqrt(2 * np.log(1 / 0.95)), fill=False, ls='--', color='k'))
        ax[k].axhline(0, c='k', lw=0.5); ax[k].axvline(0, c='k', lw=0.5)
        ax[k].set(xlabel='ΔT = T_rec − T* (°C)', ylabel='Δt = t_rec − t* (min)', title=f'({"abc"[k]}) {sname}', xlim=(-160, 160), ylim=(-12, 12))
    ax[0].legend(fontsize=7); fig.tight_layout(); fig.savefig(os.path.join(IMG, 'fig9_bo_offsets.png'))
