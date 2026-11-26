"""MMGA applied to CALCE CS2_36 (1.1 A = 1C CC discharge, voltage only).  The Ai2020 cell is used with the same
current density (1C of its nominal 2.28 Ah), i.e. CS2 is represented as an area-scaled copy (scale 2.28/1.1).
Temperature is not recorded, so h is fixed (35 W/m2/K) and only V(tau) and t_end enter the objective;
the model still outputs T (reported as a prediction).  Identified: R_n, R_p, k_n, LAM_p, R_f, LLI, SOCp."""
import sys, os, json, time, pickle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from multiprocessing import Pool
from ecat_model import ECAT, unit_to_phys, features, CAP_NOM
from ga import ga
from step3_identify import distinct_topk, SIG_V, SIG_Q
from step5_nasa import feasibility

O = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'outputs')
N5 = ['R_n', 'R_p', 'k_n', 'LAM_p', 'R_f', 'LLI', 'SOCp']   # run with ECAT_BOUNDS=real
V_CUT = 2.7; SCALE = CAP_NOM / 1.1
CYCLES = [0, 5, 10]

def objV(V, T, te, tgt):
    return np.mean((V - tgt[0]) ** 2, -1) / SIG_V ** 2 + ((te - tgt[2]) / tgt[2]) ** 2 / SIG_Q ** 2

_m = None
def init():
    global _m
    _m = ECAT(I_app=CAP_NOM, T_amb=298.15, V_cut=V_CUT, fixed=dict(h=35.0)); _m.run([5e-6, 3e-6, 1, 0.7, 0.0, 1.0, 1.0], N5)

def work(u):
    r = _m.run(unit_to_phys(u[None], N5)[0], N5)
    return None if r is None else features(r) + ((r['t'], r['V'], r['T']),)

if __name__ == '__main__':
    E = pickle.load(open(os.path.join(O, 'exp_data.pkl'), 'rb'))['CS2_36']
    mm = pickle.load(open(os.path.join(O, 'ann_cs2_7.pkl'), 'rb'))['mm']
    clf = feasibility('cs2_7')
    pool = Pool(2, initializer=init); res = []; curves = {}
    for cyc in CYCLES:
        c = E[cyc]; t, V = c['t'], c['V']
        k = np.where(V < V_CUT)[0]
        if len(k):
            k = k[0]; f = (V[k - 1] - V_CUT) / (V[k - 1] - V[k]); te = t[k - 1] + f * (t[k] - t[k - 1])
            t = np.append(t[:k], te); V = np.append(V[:k], V_CUT)
        tgt = features(dict(t=t, V=V, T=np.zeros_like(t)))
        allP, allF = [], []
        for s in range(5):
            P, F, h = ga(lambda U: objV(*mm.predict(U), tgt) + 1e3 * clf.predict_proba(U)[:, 0], len(N5), 200, 100, seed=s); allP.append(P); allF.append(F)
        Pc, Fc = distinct_topk(np.vstack(allP), np.concatenate(allF), 10)
        outs = pool.map(work, list(Pc))
        Jp = np.array([objV(o[0], o[1], o[2], tgt) if o is not None else 1e9 for o in outs]); b = int(np.argmin(Jp))
        tm_, Vm, Tm = outs[b][3]; tt = np.linspace(0, min(tm_[-1], t[-1]), 1000)
        dV = np.interp(tt, tm_, Vm) - np.interp(tt, t, V)
        x = unit_to_phys(Pc[b][None], N5)[0]
        e = dict(cycle=cyc, x=x.tolist(), R_f_CS2_equivalent_Ohm=float(x[4] * SCALE), J_ann=float(Fc[b]), J_phys=float(Jp[b]),
                 rmse_V_mV=float(1e3 * np.sqrt(np.mean(dV ** 2))), max_V_mV=float(1e3 * np.abs(dV).max()),
                 cap_meas_Ah=float(t[-1] * 1.1 / 3600), cap_model_Ah=float(tm_[-1] * 1.1 / 3600),
                 cap_err_pct=float(100 * (tm_[-1] - t[-1]) / t[-1]), predicted_dT_K=float(Tm.max() - Tm[0]))
        res.append(e); curves[cyc] = dict(exp=(t, V), model=(tm_, Vm, Tm))
        print(e, flush=True)
        json.dump(res, open(os.path.join(O, 'cs2_7_mmga.json'), 'w'), indent=1)
        pickle.dump(curves, open(os.path.join(O, 'cs2_7_curves.pkl'), 'wb'))
    pool.close()
