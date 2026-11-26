"""Step 3: MMGA -- genetic algorithm on the ANN meta-model identifying {R_n, R_p, k_n, h}.
Synthetic-truth benchmark: truths are simulated with a REFINED PyBaMM discretisation (avoids the inverse crime)
and corrupted with Gaussian noise (2 mV, 0.1 K).  Top-K distinct GA candidates are re-simulated with the
physics model and the best physics fit is retained.  A physics-only GA baseline is run for cost comparison.
"""
import sys, os, json, time, pickle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from multiprocessing import Pool
from ecat_model import ECAT, CORE, NOMINAL, unit_to_phys, phys_to_unit, features, TAU
from ga import ga

O = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'outputs')
SIG_V, SIG_T, SIG_Q = 0.010, 0.5, 0.005     # normalisation of the objective: 10 mV, 0.5 K, 0.5 % capacity
NOISE_V, NOISE_T = 0.002, 0.1
POP, GENS, SEEDS, TOPK = 200, 100, 5, 10
PGA_POP, PGA_GENS = 40, 25


def objective(V, T, te, tgt):
    Vt, Tt, tt = tgt
    return (np.mean((V - Vt) ** 2, -1) / SIG_V ** 2 + np.mean((T - Tt) ** 2, -1) / SIG_T ** 2
            + ((te - tt) / tt) ** 2 / SIG_Q ** 2)


_m = None
def init():
    global _m
    _m = ECAT(); _m.run([NOMINAL[n] for n in CORE], CORE)

def work(u):
    x = unit_to_phys(u[None], CORE)[0]
    r = _m.run(x, CORE)
    if r is None:
        return None
    return features(r) + ((r['t'][::5], r['V'][::5], r['T'][::5]),)


def phys_J(pool, U, tgt):
    out = pool.map(work, list(U))
    J = np.array([objective(o[0], o[1], o[2], tgt) if o is not None else 1e6 for o in out])
    return J, out


def distinct_topk(P, F, k, tol=0.03):
    order = np.argsort(F); sel = []
    for i in order:
        if all(np.max(np.abs(P[i] - P[j])) > tol for j in sel):
            sel.append(i)
        if len(sel) == k:
            break
    return P[sel], F[sel]


def curve_metrics(o, raw_truth):
    """RMSE of identified physics curve vs the noise-free truth on the common time window (absolute time)"""
    tt, Vt, Tt = raw_truth; t, V, T = o[3]
    tm = np.linspace(0, min(t[-1], tt[-1]), 1500)
    dV = np.interp(tm, t, V) - np.interp(tm, tt, Vt); dT = np.interp(tm, t, T) - np.interp(tm, tt, Tt)
    rV, rT = np.sqrt(np.mean(dV ** 2)), np.sqrt(np.mean(dT ** 2))
    # combined V-T RMSE: residuals of each signal normalised by the range of the true signal, pooled
    comb = np.sqrt(0.5 * (np.mean((dV / np.ptp(Vt)) ** 2) + np.mean((dT / np.ptp(Tt)) ** 2)))
    return dict(rmse_V_mV=float(1e3 * rV), rmse_T_K=float(rT), rmse_VT_combined_pct=float(100 * comb),
                cap_err_pct=float(100 * (t[-1] - tt[-1]) / tt[-1]))


if __name__ == '__main__':
    A = pickle.load(open(os.path.join(O, 'ann_core.pkl'), 'rb')); mm = A['mm']
    ref = ECAT(var_pts=dict(x_n=40, x_s=40, x_p=40, r_n=60, r_p=60), rtol=1e-9, atol=1e-11, dt_out=1.0)
    rng = np.random.default_rng(123)
    U_true = np.vstack([phys_to_unit(np.array([[NOMINAL[n] for n in CORE]]), CORE),
                        0.12 + 0.76 * rng.random((4, 4))])
    X_true = unit_to_phys(U_true, CORE)
    results = []; curves = {}
    pool = Pool(2, initializer=init)
    for c, (u_t, x_t) in enumerate(zip(U_true, X_true)):
        r = ref.run(x_t, CORE)
        Vn = r['V'] + NOISE_V * rng.standard_normal(len(r['V'])); Tn = r['T'] + NOISE_T * rng.standard_normal(len(r['T']))
        tgt = features(dict(t=r['t'], V=Vn, T=Tn))
        tgt_clean = features(r)
        t0 = time.time()
        fobj = lambda U: objective(*mm.predict(U), tgt)
        per_seed, allP, allF, hists = [], [], [], []
        for s in range(SEEDS):
            P, F, h = ga(fobj, 4, POP, GENS, seed=s)
            per_seed.append(dict(seed=s, u=P[0].tolist(), x=unit_to_phys(P[0][None], CORE)[0].tolist(), J_ann=float(F[0])))
            allP.append(P); allF.append(F); hists.append(h)
        t_ga = time.time() - t0
        Pc, Fc = distinct_topk(np.vstack(allP), np.concatenate(allF), TOPK)
        t0 = time.time(); Jp, outs = phys_J(pool, Pc, tgt); t_ver = time.time() - t0
        b = int(np.argmin(Jp)); u_id = Pc[b]; x_id = unit_to_phys(u_id[None], CORE)[0]
        Jtrue_floor = float(objective(*tgt_clean, tgt))   # J of the exact truth (noise floor + discretisation)
        cm = curve_metrics(outs[b], (r['t'], r['V'], r['T']))
        seedX = np.array([p['x'] for p in per_seed])
        res = dict(case=c, u_true=u_t.tolist(), x_true=x_t.tolist(), u_id=u_id.tolist(), x_id=x_id.tolist(),
                   rel_err_pct={n: float(100 * (x_id[j] - x_t[j]) / x_t[j]) for j, n in enumerate(CORE)},
                   seed_rel_err_pct={n: (100 * (seedX[:, j] - x_t[j]) / x_t[j]).tolist() for j, n in enumerate(CORE)},
                   seed_cv_pct={n: float(100 * seedX[:, j].std() / seedX[:, j].mean()) for j, n in enumerate(CORE)},
                   J_ann_best=float(Fc[b]), J_phys_best=float(Jp[b]), J_ann_topk=Fc.tolist(), J_phys_topk=Jp.tolist(),
                   J_truth_floor=Jtrue_floor, curve=cm, per_seed=per_seed,
                   time_ga_s=t_ga, time_verify_s=t_ver, n_ann_evals=int(SEEDS * hists[0]['n_eval'][-1]),
                   n_phys_sims_online=int(TOPK), hist_best=[h['best'] for h in hists], hist_median=[h['median'] for h in hists])
        results.append(res)
        curves[c] = dict(truth=(r['t'], r['V'], r['T']), noisy=(r['t'], Vn, Tn), ident=outs[b][3])
        print(c, 'true', np.round(x_t, 8), 'id', np.round(x_id, 8), res['rel_err_pct'], cm, 'Jann %.3f Jphys %.3f floor %.3f' % (Fc[b], Jp[b], Jtrue_floor), flush=True)
        json.dump(results, open(os.path.join(O, 'step3_synthetic.json'), 'w'), indent=1)
        pickle.dump(curves, open(os.path.join(O, 'step3_curves.pkl'), 'wb'))

    # ---------------- physics-only GA baseline (cases 0 and 1) ----------------
    base = []
    for c in [0, 1]:
        x_t = X_true[c]; r = ref.run(x_t, CORE)
        rng2 = np.random.default_rng(1000 + c)
        Vn = r['V'] + NOISE_V * rng2.standard_normal(len(r['V'])); Tn = r['T'] + NOISE_T * rng2.standard_normal(len(r['T']))
        tgt = features(dict(t=r['t'], V=Vn, T=Tn))
        t0 = time.time()
        P, F, h = ga(lambda U: phys_J(pool, U, tgt)[0], 4, PGA_POP, PGA_GENS, seed=0)
        tp = time.time() - t0
        x_id = unit_to_phys(P[0][None], CORE)[0]
        # MMGA on the same target for a like-for-like comparison
        t0 = time.time()
        allP, allF = [], []
        for s in range(SEEDS):
            Pm, Fm, hm = ga(lambda U: objective(*mm.predict(U), tgt), 4, POP, GENS, seed=s); allP.append(Pm); allF.append(Fm)
        Pc, Fc = distinct_topk(np.vstack(allP), np.concatenate(allF), TOPK)
        Jp, outs = phys_J(pool, Pc, tgt); tm = time.time() - t0
        bm = int(np.argmin(Jp)); xm = unit_to_phys(Pc[bm][None], CORE)[0]
        base.append(dict(case=c, physGA=dict(pop=PGA_POP, gens=PGA_GENS, n_sims=int(h['n_eval'][-1]), time_s=tp,
                                             J_best=float(F[0]), x_id=x_id.tolist(),
                                             rel_err_pct={n: float(100 * (x_id[j] - x_t[j]) / x_t[j]) for j, n in enumerate(CORE)},
                                             hist_best=h['best'], hist_neval=h['n_eval']),
                         MMGA=dict(n_sims_online=TOPK, time_s=tm, J_phys=float(Jp[bm]), x_id=xm.tolist(),
                                   rel_err_pct={n: float(100 * (xm[j] - x_t[j]) / x_t[j]) for j, n in enumerate(CORE)})))
        print('baseline', c, base[-1]['physGA']['J_best'], base[-1]['physGA']['time_s'], base[-1]['MMGA']['J_phys'], tm, flush=True)
        json.dump(base, open(os.path.join(O, 'step3_physga_baseline.json'), 'w'), indent=1)
    pool.close()
