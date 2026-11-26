"""Real-data application of MMGA to NASA PCoE B0005 (2 A ~ 1C CC discharge, 24 degC, V and T measured).
ECAT inputs (8): R_n, R_p, k_n, h (core) + LAM_p, R_f, LLI (aging state) + SOCp (initial positive stoichiometry).
Run with ECAT_BOUNDS=real (wider box).
(a) cycle 0: full 6-D MMGA;  (b) aging tracking: cycles 0..167 with R_n, R_p, h frozen at cycle-0 values and
(k_n, LAM_p, R_f, LLI) re-identified, plus an unconstrained 6-D run for comparison."""
import sys, os, json, time, pickle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from multiprocessing import Pool
from ecat_model import ECAT, unit_to_phys, phys_to_unit, features, TAU
from ga import ga
from step3_identify import objective, distinct_topk

O = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'outputs')
N6 = ['R_n', 'R_p', 'k_n', 'h', 'LAM_p', 'R_f', 'LLI', 'SOCp']   # 8 inputs (name kept for brevity)
ND = len(N6)
TRACK = [2, 4, 5, 6]   # re-identified over life: k_n, LAM_p, R_f, LLI (R_n, R_p, h, SOCp frozen at cycle 0)
I_APP, T_AMB, V_CUT = 2.0, 297.15, 2.7
CYCLES = [0, 10, 25, 40, 55, 70, 85, 100, 115, 130, 145, 160, 167]

_m = None
def init():
    global _m
    _m = ECAT(I_app=I_APP, T_amb=T_AMB, V_cut=V_CUT); _m.run([5e-6, 3e-6, 1, 10, 0.8, 0.0, 1.0, 1.0], N6)

def work(u):
    x = unit_to_phys(u[None], N6)[0]; r = _m.run(x, N6)
    if r is None:
        return None
    return features(r) + ((r['t'], r['V'], r['T']),)


def target(c):
    t, V, T = c['t'], c['V'], c['T'] + 273.15
    k = np.where(V < V_CUT)[0]
    if len(k):   # truncate at the 2.7 V cut-off (interpolated crossing)
        k = k[0]; f = (V[k - 1] - V_CUT) / (V[k - 1] - V[k]); te = t[k - 1] + f * (t[k] - t[k - 1])
        Tk = T[k - 1] + f * (T[k] - T[k - 1])
        t = np.append(t[:k], te); V = np.append(V[:k], V_CUT); T = np.append(T[:k], Tk)
    return dict(t=t, V=V, T=T)


def fit_metrics(o, tg):
    t, V, T = o[3]; tm = np.linspace(0, min(t[-1], tg['t'][-1]), 1500)
    dV = np.interp(tm, t, V) - np.interp(tm, tg['t'], tg['V'])
    dT = (np.interp(tm, t, T) - T[0]) - (np.interp(tm, tg['t'], tg['T']) - tg['T'][0])
    return dict(rmse_V_mV=float(1e3 * np.sqrt(np.mean(dV ** 2))), max_V_mV=float(1e3 * np.abs(dV).max()),
                rmse_dT_K=float(np.sqrt(np.mean(dT ** 2))), cap_model_Ah=float(t[-1] * I_APP / 3600),
                cap_meas_Ah=float(tg['t'][-1] * I_APP / 3600),
                cap_err_pct=float(100 * (t[-1] - tg['t'][-1]) / tg['t'][-1]),
                rmse_VT_combined_pct=float(100 * np.sqrt(0.5 * (np.mean((dV / np.ptp(tg['V'])) ** 2) + np.mean((dT / np.ptp(tg['T'])) ** 2)))))


def feasibility(db):
    """random-forest classifier of simulation feasibility (initial OCV above the upper cut-off -> failure);
    used as a penalty so that the GA does not exploit ANN extrapolation into infeasible regions"""
    from sklearn.ensemble import RandomForestClassifier
    d = np.load(os.path.join(O, f'db_{db}.npz'))
    return RandomForestClassifier(200, random_state=0, n_jobs=1).fit(d['U'], d['ok'])


CLF = None
def mmga(mm, tgt_f, pool, free, u_fixed, seeds=5, pop=200, gens=100, topk=10):
    free = np.array(free)
    def lift(Uf):
        U = np.tile(u_fixed, (len(Uf), 1)); U[:, free] = Uf; return U
    def fobj(Uf):
        U = lift(Uf); J = objective(*mm.predict(U), tgt_f)
        return J if CLF is None else J + 1e3 * CLF.predict_proba(U)[:, 0]
    allP, allF, H, best_seed = [], [], [], []
    for s in range(seeds):
        P, F, h = ga(fobj, len(free), pop, gens, seed=s); allP.append(P); allF.append(F); H.append(h['best'])
        best_seed.append(unit_to_phys(lift(P[:1]), N6)[0].tolist())
    Pc, Fc = distinct_topk(np.vstack(allP), np.concatenate(allF), topk)
    outs = pool.map(work, list(lift(Pc)))
    Jp = np.array([objective(o[0], o[1], o[2], tgt_f) if o is not None else 1e9 for o in outs])
    b = int(np.argmin(Jp))
    return lift(Pc[b:b + 1])[0], outs[b], float(Fc[b]), float(Jp[b]), H, best_seed


if __name__ == '__main__':
    E = pickle.load(open(os.path.join(O, 'exp_data.pkl'), 'rb'))['NASA']['B0005']
    mm = pickle.load(open(os.path.join(O, 'ann_nasa8.pkl'), 'rb'))['mm']
    CLF = feasibility('nasa8')
    pool = Pool(2, initializer=init)
    res = dict(names=N6, cycles=[]); curves = {}
    u_fix = None
    for cyc in CYCLES:
        tg = target(E[cyc]); tgt_f = features(tg)
        t0 = time.time()
        u6, o6, Ja6, Jp6, H6, bs6 = mmga(mm, tgt_f, pool, range(ND), np.zeros(ND))
        entry = dict(cycle=cyc, cap_meas_Ah=E[cyc]['cap'], free6=dict(x=unit_to_phys(u6[None], N6)[0].tolist(), u=u6.tolist(),
                     J_ann=Ja6, J_phys=Jp6, fit=fit_metrics(o6, tg), seed_best=bs6, hist=H6))
        if cyc == 0:
            u_fix = u6.copy()
        u3, o3, Ja3, Jp3, H3, bs3 = mmga(mm, tgt_f, pool, TRACK, u_fix)
        entry['track3'] = dict(x=unit_to_phys(u3[None], N6)[0].tolist(), u=u3.tolist(), J_ann=Ja3, J_phys=Jp3,
                               fit=fit_metrics(o3, tg), seed_best=bs3)
        entry['time_s'] = time.time() - t0
        res['cycles'].append(entry)
        curves[cyc] = dict(exp=(tg['t'], tg['V'], tg['T']), free6=o6[3], track3=o3[3])
        print(cyc, 'free6', np.round(entry['free6']['x'], 7), entry['free6']['fit'], '\n   track3', np.round(entry['track3']['x'], 7), entry['track3']['fit'], flush=True)
        json.dump(res, open(os.path.join(O, 'nasa8_mmga.json'), 'w'), indent=1)
        pickle.dump(curves, open(os.path.join(O, 'nasa8_curves.pkl'), 'wb'))
    pool.close()
