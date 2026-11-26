"""Numerical convergence check of the PyBaMM ECAT model: default mesh/tolerances (used for the databases)
vs. a refined reference (3x radial and through-thickness points, tighter tolerances, 0.5 s output)."""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from ecat_model import ECAT, CORE, NOMINAL, BOUNDS

O = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'outputs')
std = ECAT()
ref = ECAT(var_pts=dict(x_n=40, x_s=40, x_p=40, r_n=60, r_p=60), rtol=1e-9, atol=1e-11, dt_out=0.5)
cases = {'nominal': [NOMINAL[n] for n in CORE],
         'lower_corner': [BOUNDS[n][0] for n in CORE], 'upper_corner': [BOUNDS[n][1] for n in CORE],
         'large_Rn_small_k': [BOUNDS['R_n'][1], BOUNDS['R_p'][1], BOUNDS['k_n'][0], 10.0]}
out = {}
for k, x in cases.items():
    a = std.run(x, CORE); b = ref.run(x, CORE)
    tt = np.linspace(0, min(a['t'][-1], b['t'][-1]) * 0.995, 2000)
    dV = np.interp(tt, a['t'], a['V']) - np.interp(tt, b['t'], b['V'])
    dT = np.interp(tt, a['t'], a['T']) - np.interp(tt, b['t'], b['T'])
    out[k] = dict(x=x, V_rmse_mV=float(1e3 * np.sqrt(np.mean(dV ** 2))), V_max_mV=float(1e3 * np.abs(dV).max()),
                  T_rmse_K=float(np.sqrt(np.mean(dT ** 2))), T_max_K=float(np.abs(dT).max()),
                  tend_std_s=float(a['t'][-1]), tend_ref_s=float(b['t'][-1]),
                  tend_rel_err_pct=float(100 * (a['t'][-1] - b['t'][-1]) / b['t'][-1]))
    print(k, out[k], flush=True)
json.dump(out, open(os.path.join(O, 'convergence_check.json'), 'w'), indent=2)
