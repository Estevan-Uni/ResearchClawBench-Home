"""Step 1: Latin-hypercube sampling of the ECAT parameter space + PyBaMM simulation of a 1C CC discharge
(voltage AND temperature).  Usage:
    python step1_lhs.py --n 20 --seed 0 --tag baseline20 [--names R_n,R_p,k_n,h] [--I 2.28 --Tamb 298.15 --Vcut 3.0]
Writes outputs/db_<tag>.npz and outputs/db_<tag>.json (timings, failures)."""
import sys, os, json, time, argparse, pickle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from multiprocessing import Pool
from scipy.stats import qmc

ap = argparse.ArgumentParser()
ap.add_argument('--n', type=int, default=20); ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--tag', default='baseline20'); ap.add_argument('--names', default='R_n,R_p,k_n,h')
ap.add_argument('--I', type=float, default=None); ap.add_argument('--Tamb', type=float, default=298.15)
ap.add_argument('--Vcut', type=float, default=3.0); ap.add_argument('--fixed', default='')
ap.add_argument('--procs', type=int, default=2); ap.add_argument('--raw', type=int, default=0)
a = ap.parse_args()
NAMES = a.names.split(',')
FIXED = json.loads(a.fixed) if a.fixed else {}
O = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'outputs')

_m = None
def init():
    global _m
    from ecat_model import ECAT
    _m = ECAT(I_app=a.I, T_amb=a.Tamb, V_cut=a.Vcut, fixed=FIXED)
    _m.run([_m.fixed[n] for n in NAMES], NAMES)  # warm-up (discretisation / compilation)

def work(x):
    from ecat_model import features
    t0 = time.time(); r = _m.run(x, NAMES); dt = time.time() - t0
    if r is None:
        return None, dt, None
    V, T, te = features(r)
    raw = (r['t'][::5], r['V'][::5], r['T'][::5]) if a.raw else None
    return (V, T, te), dt, raw

if __name__ == '__main__':
    from ecat_model import unit_to_phys, TAU
    U = qmc.LatinHypercube(d=len(NAMES), seed=a.seed).random(a.n)
    X = unit_to_phys(U, NAMES)
    t0 = time.time()
    with Pool(a.procs, initializer=init) as pool:
        res = pool.map(work, list(X), chunksize=max(1, a.n // (20 * a.procs)))
    wall = time.time() - t0
    ok = np.array([r[0] is not None for r in res]); times = np.array([r[1] for r in res])
    nT = len(TAU)
    Vf = np.array([r[0][0] if r[0] is not None else np.full(nT, np.nan) for r in res])
    Tf = np.array([r[0][1] if r[0] is not None else np.full(nT, np.nan) for r in res])
    te = np.array([r[0][2] if r[0] is not None else np.nan for r in res])
    np.savez(os.path.join(O, f'db_{a.tag}.npz'), U=U, X=X, V=Vf, T=Tf, tend=te, ok=ok, times=times, tau=TAU,
             names=np.array(NAMES))
    if a.raw:
        pickle.dump([r[2] for r in res], open(os.path.join(O, f'db_{a.tag}_raw.pkl'), 'wb'))
    info = dict(tag=a.tag, names=NAMES, n_attempted=int(a.n), n_success=int(ok.sum()), n_failed=int((~ok).sum()),
                wall_time_s=wall, mean_time_per_sim_s=float(times.mean()), median_time_per_sim_s=float(np.median(times)),
                procs=a.procs, I_app=a.I, T_amb=a.Tamb, V_cut=a.Vcut, fixed=FIXED,
                tend_range_s=[float(np.nanmin(te)), float(np.nanmax(te))],
                dT_max_range_K=[float(np.nanmin(np.nanmax(Tf, 1))), float(np.nanmax(np.nanmax(Tf, 1)))])
    json.dump(info, open(os.path.join(O, f'db_{a.tag}.json'), 'w'), indent=2)
    print(json.dumps(info, indent=2))
