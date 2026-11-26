"""Optional extension: dynamic-load generalisation on the Oxford Artemis urban drive cycle (Kokam 740 mAh, 40 degC).
The Ai2020-based ECAT cell is current-density scaled (I_model = I_data * 2.28/0.74); the drive current is
averaged on 10 s windows (piecewise-linear Interpolant) to keep the solve at ~5 s.
  db   : LHS database (R_n, R_p, k_n, LAM_p, R_f, LLI; h fixed) -> V(t), T(t) on a fixed 10 s grid
  fit  : ANN (4-layer FC, Adam) + GA on the FIRST 60 % of the drive cycle; physics verification;
         prediction of the remaining 40 % (unseen dynamic data) with the identified parameters."""
import sys, os, json, time, pickle, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from multiprocessing import Pool
from scipy.stats import qmc
from ecat_model import ECAT, unit_to_phys, CAP_NOM

O = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'outputs')
N = ['R_n', 'R_p', 'k_n', 'LAM_p', 'R_f', 'LLI']
SCALE = CAP_NOM / 0.74
E = pickle.load(open(os.path.join(O, 'exp_data.pkl'), 'rb'))['Oxford']['dc']
T_DATA = E['t'] - E['t'][0]
DT = 10.0
TK = np.arange(0, T_DATA[-1] + DT, DT)
IK = np.array([(-E['I'] * SCALE)[(T_DATA >= a) & (T_DATA < a + DT)].mean() if np.any((T_DATA >= a) & (T_DATA < a + DT)) else 0.0 for a in TK])
TG = np.arange(0, T_DATA[-1] - 1, DT)            # fixed output grid
SPLIT = int(0.6 * len(TG))                          # identification window = first 60 %
T_AMB = 273.15 + float(E['T'][0])

_m = None
def init():
    global _m
    _m = ECAT(I_app=1.0, T_amb=T_AMB, V_cut=2.5, current=(TK, IK), rtol=1e-4, atol=1e-6, fixed=dict(h=35.0))

def work(x):
    t0 = time.time(); r = _m.run(x, N, t_eval=np.arange(0, T_DATA[-1] - 1, 2.0)); dt = time.time() - t0
    if r is None or r['t'][-1] < TG[-1] - 1:
        return None, dt
    return (np.interp(TG, r['t'], r['V']), np.interp(TG, r['t'], r['T']) - r['T'][0]), dt


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('mode'); ap.add_argument('--n', type=int, default=800); a = ap.parse_args()
    if a.mode == 'db':
        U = qmc.LatinHypercube(d=len(N), seed=21).random(a.n); X = unit_to_phys(U, N)
        t0 = time.time()
        with Pool(2, initializer=init) as pool:
            res = pool.map(work, list(X), chunksize=4)
        ok = np.array([r[0] is not None for r in res])
        V = np.array([r[0][0] if r[0] else np.full(len(TG), np.nan) for r in res]); T = np.array([r[0][1] if r[0] else np.full(len(TG), np.nan) for r in res])
        np.savez(os.path.join(O, 'db_oxford.npz'), U=U, X=X, V=V, T=T, ok=ok, tg=TG)
        info = dict(n_attempted=a.n, n_success=int(ok.sum()), n_failed=int((~ok).sum()), wall_time_s=time.time() - t0,
                    mean_time_per_sim_s=float(np.mean([r[1] for r in res])), names=N, scale=SCALE, T_amb=T_AMB)
        json.dump(info, open(os.path.join(O, 'db_oxford.json'), 'w'), indent=2); print(info)
    else:
        from sklearn.neural_network import MLPRegressor
        from ga import ga
        from step3_identify import distinct_topk
        d = np.load(os.path.join(O, 'db_oxford.npz')); ok = d['ok']
        U, V, T = d['U'][ok], d['V'][ok], d['T'][ok]
        Y = np.hstack([V, T]); mu, sd = Y.mean(0), Y.std(0) + 1e-9
        rng = np.random.default_rng(0); idx = rng.permutation(len(U)); te, tr = idx[:80], idx[80:]
        net = MLPRegressor(hidden_layer_sizes=(192, 192, 192), activation='tanh', solver='adam', learning_rate_init=2e-3, batch_size=32,
                           random_state=0, alpha=1e-6, max_iter=1)
        hist = dict(train=[], test=[])
        EP = 1500
        for ep in range(EP):
            net.partial_fit(U[tr], (Y[tr] - mu) / sd)
            net._optimizer.learning_rate = 2e-5 + 0.5 * (2e-3 - 2e-5) * (1 + np.cos(np.pi * (ep + 1) / EP))
            if ep % 10 == 0 or ep == EP - 1:
                hist['train'].append(float(np.mean((net.predict(U[tr]) - (Y[tr] - mu) / sd) ** 2)))
                hist['test'].append(float(np.mean((net.predict(U[te]) - (Y[te] - mu) / sd) ** 2)))
        pred = lambda Uq: net.predict(np.atleast_2d(Uq)) * sd + mu
        Pt = pred(U[te]); nV = V.shape[1]
        ann_m = dict(n_train=int(len(tr)), n_test=int(len(te)), test_rmse_V_mV=float(1e3 * np.sqrt(np.mean((Pt[:, :nV] - V[te]) ** 2))),
                     test_rmse_T_K=float(np.sqrt(np.mean((Pt[:, nV:] - T[te]) ** 2))), mse_initial=hist['test'][0], mse_final=hist['test'][-1])
        # measured signals averaged on the same 10 s windows as the input current (consistent comparison)
        win = lambda y: np.array([y[(T_DATA >= a) & (T_DATA < a + DT)].mean() for a in TG])
        Vd = win(E['V']); Td = win(E['T']) - E['T'][0]
        Vd_inst = np.interp(TG, T_DATA, E['V'])
        w = slice(0, SPLIT)
        fobj = lambda Uq: np.mean((pred(Uq)[:, :nV][:, w] - Vd[w]) ** 2, 1) / 0.01 ** 2
        allP, allF = [], []
        for s in range(5):
            P, F, h = ga(fobj, len(N), 200, 100, seed=s); allP.append(P); allF.append(F)
        Pc, Fc = distinct_topk(np.vstack(allP), np.concatenate(allF), 8)
        X = unit_to_phys(Pc, N)
        with Pool(2, initializer=init) as pool:
            outs = pool.map(work, list(X))
        Jp = np.array([np.mean((o[0][0][w] - Vd[w]) ** 2) / 0.01 ** 2 if o[0] else 1e9 for o in outs]); b = int(np.argmin(Jp))
        Vm, Tm = outs[b][0]
        r = lambda e: float(1e3 * np.sqrt(np.mean(e ** 2)))
        res = dict(ann=ann_m, x_id=X[b].tolist(), names=N, J_ann=float(Fc[b]), J_phys=float(Jp[b]),
                   rmse_V_ident_window_mV=r(Vm[:SPLIT] - Vd[:SPLIT]), rmse_V_prediction_window_mV=r(Vm[SPLIT:] - Vd[SPLIT:]),
                   max_V_prediction_window_mV=float(1e3 * np.abs(Vm[SPLIT:] - Vd[SPLIT:]).max()),
                   rmse_dT_full_K=float(np.sqrt(np.mean((Tm - Td) ** 2))),
                   rmse_V_prediction_vs_instantaneous_mV=r(Vm[SPLIT:] - Vd_inst[SPLIT:]), db_success=f'{int(ok.sum())}/{len(ok)}', t_split_s=float(TG[SPLIT]), hist=hist)
        json.dump(res, open(os.path.join(O, 'oxford_mmga.json'), 'w'), indent=1)
        pickle.dump(dict(tg=TG, Vd=Vd, Td=Td, Vm=Vm, Tm=Tm, split=SPLIT, t=T_DATA, I=E['I']), open(os.path.join(O, 'oxford_curves.pkl'), 'wb'))
        print({k: v for k, v in res.items() if k != 'hist'})
