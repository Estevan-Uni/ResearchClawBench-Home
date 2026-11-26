"""Model-structure test on NASA B0005 cycle 0: direct physics Nelder-Mead fits with different sets of free
stoichiometry parameters (is the 48 mV residual a parameter-set or a structural limitation?)."""
import sys, os, json, pickle; sys.path.insert(0, 'code')
import numpy as np
from scipy.optimize import minimize
from ecat_model import ECAT, unit_to_phys, phys_to_unit, features
from step5_nasa import target, fit_metrics
from step3_identify import objective
E = pickle.load(open('outputs/exp_data.pkl', 'rb'))['NASA']['B0005']
tg = target(E[0]); tf = features(tg)
m = ECAT(I_app=2.0, T_amb=297.15, V_cut=2.7)
out = {}
for name, N, x0 in [('6p', ['R_n','R_p','k_n','h','LAM_p','R_f'], [1e-5,2.75e-6,0.262,10.8,0.727,0.0269]),
                    ('8p', ['R_n','R_p','k_n','h','LAM_p','R_f','LLI','SOCp'], [1e-5,2.75e-6,0.262,10.8,0.727,0.0269,1.0,1.0])]:
    u0 = np.clip(phys_to_unit(np.array([x0]), N)[0], 0.02, 0.98); cnt = [0]
    def f(u):
        cnt[0] += 1
        r = m.run(unit_to_phys(np.clip(u, 0, 1)[None], N)[0], N)
        return 1e6 if r is None else float(objective(*features(r), tf)) + 1e3 * np.sum(np.clip(np.abs(u - 0.5) - 0.5, 0, None))
    best = (f(u0), u0)
    for st in range(2):
        rr = minimize(f, best[1], method='Nelder-Mead', options=dict(maxfev=250, xatol=1e-3, fatol=1e-3, initial_simplex=None))
        if rr.fun < best[0]: best = (rr.fun, rr.x)
    x = unit_to_phys(np.clip(best[1], 0, 1)[None], N)[0]; r = m.run(x, N)
    fm = fit_metrics(features(r) + ((r['t'], r['V'], r['T']),), tg)
    out[name] = dict(names=N, x=x.tolist(), J=best[0], n_sims=cnt[0], fit=fm); print(name, out[name], flush=True)
json.dump(out, open('outputs/nasa_structure_test.json', 'w'), indent=1)
