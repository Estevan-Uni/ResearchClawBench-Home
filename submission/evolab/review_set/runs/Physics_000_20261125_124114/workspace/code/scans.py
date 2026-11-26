"""Single-step mismatch scans and stability maps.
For a core (path with joint-optimised LJ diameters) and candidate outer shells,
E(delta) is computed by relaxing core+shell with outer diameter d_last*(1+delta).
Modes:
  LJ    : grid delta in [0, 0.36] step 0.02 for all candidates (stability map) +
          bounded Brent refinement of argmin for each candidate.
  Morse : Morse rho=10, grid formula+-0.12 step 0.04, refine (NN steps only)
  harm  : harmonic contact network (fixed contact set from the ideal geometry at
          the formula delta*), same grid/refine.
Reconstructions (outer atom angular shift > 3 deg on relaxation) are flagged.
usage: python scans.py MODE core1 core2 ...   (core names below)"""
import sys, os, json, time, numpy as np
from scipy.optimize import minimize_scalar
import icoshell as ic
from joint_opt import optimise

CORES = {
    'c00': ([(0, 0)], [(0, 1), (1, 1)]),
    'c01': ([(0, 0), (0, 1)], [(0, 2), (1, 1), (1, 2)]),
    'c11': ([(0, 0), (0, 1), (1, 1)], [(1, 2), (0, 2), (2, 2)]),
    'c02': ([(0, 0), (0, 1), (0, 2)], [(0, 3), (1, 2), (2, 2)]),
    'c12': ([(0, 0), (0, 1), (1, 1), (1, 2)], [(1, 3), (2, 2), (0, 3), (2, 3)]),
    'c03': ([(0, 0), (0, 1), (0, 2), (0, 3)], [(0, 4), (1, 3), (2, 3)]),
    'c22': ([(0, 0), (0, 1), (1, 1), (1, 2), (2, 2)], [(2, 3), (1, 3), (0, 4)]),
    'c03b': ([(0, 0), (0, 1), (0, 2), (0, 3)], [(2, 2)]),
    'c11x': ([(0, 0), (0, 1), (1, 1)], [(0, 2)]),
    'c13': ([(0, 0), (0, 1), (1, 1), (1, 2), (1, 3)], [(1, 4), (2, 3), (0, 4)]),
}
EB_LJ = -8.61   # fcc LJ cohesive energy per atom (full LJ, units of eps)


def core_diams(path, cache):
    key = str(path)
    if key not in cache:
        if len(path) == 1:
            cache[key] = [1.0]
        elif len(path) == 2:
            cache[key] = [1.0, float(ic.vertex_contact_diams(path)[1])]
            r = optimise(path)
            cache[key] = r['diams']
        else:
            cache[key] = optimise(path)['diams']
    return cache[key]


def evaluate(path, diams, kind, pairs_ref=None):
    X, D, I = ic.build_cluster(path, diams)
    if kind == 'harm':
        pot = ic.PairPot(D, 'harm', pairs=pairs_ref)
    elif kind == 'Morse':
        pot = ic.PairPot(D, 'Morse', rho=10.0)
    else:
        pot = ic.PairPot(D, 'LJ')
    E, X1 = ic.relax(X, pot)
    outer = I == I.max()
    shift = ic.max_angular_shift(X, X1, outer)
    return float(E), float(shift), len(X)


def scan(core, kind, cache):
    path, cands = CORES[core]
    dc = core_diams(path, cache)
    Tc = ic.T_of(*path[-1])
    res = {}
    for c in cands:
        nn = ic.step_type(path[-1], c) != 'non-NN'
        if kind != 'LJ' and not nn:
            continue
        p = path + [c]
        dform = ic.delta_star(Tc, ic.T_of(*c))
        pairs_ref = None
        if kind == 'harm':
            X, D, I = ic.build_cluster(p, dc + [dc[-1] * (1 + dform)])
            pairs_ref = ic.contact_pairs(X, D, tol=1.2)
        if kind == 'LJ':
            grid = np.round(np.arange(0.0, 0.3601, 0.02), 3)
            if core == 'c00':
                grid = np.round(np.arange(-0.1, 0.4601, 0.02), 3)
            if nn and dform > 0.32:
                grid = np.round(np.arange(0.0, dform + 0.1601, 0.02), 3)
        else:
            grid = np.round(dform + np.arange(-0.12, 0.1201, 0.04), 4)
        Es, sh = [], []
        for d in grid:
            E, s, N = evaluate(p, dc + [dc[-1] * (1 + d)], kind, pairs_ref)
            Es.append(E); sh.append(s)
        Es = np.array(Es); sh = np.array(sh)
        ok = sh < 3.0
        Eok = np.where(ok, Es, np.inf)
        i = int(np.argmin(Eok))
        lo, hi = grid[max(i - 1, 0)], grid[min(i + 1, len(grid) - 1)]
        f = lambda d: evaluate(p, dc + [dc[-1] * (1 + d)], kind, pairs_ref)[0]
        if hi > lo:
            r = minimize_scalar(f, bounds=(lo, hi), method='bounded', options={'xatol': 2e-4})
            dopt, Eopt = float(r.x), float(r.fun)
        else:
            dopt, Eopt = float(grid[i]), float(Es[i])
        Eopt_full, sopt, N = evaluate(p, dc + [dc[-1] * (1 + dopt)], kind, pairs_ref)
        res[str(c)] = dict(shell=c, step=ic.step_type(path[-1], c), nn=nn, grid=grid.tolist(),
                           E=Es.tolist(), shift=sh.tolist(), delta_opt=dopt, E_opt=Eopt,
                           shift_opt=sopt, delta_formula=float(dform), N=N,
                           edge_of_grid=bool(i == 0 or i == len(grid) - 1),
                           Delta=((Es - EB_LJ * N) / N ** (2 / 3)).tolist() if kind == 'LJ' else None,
                           cos_theta=ic.step_cos(path[-1], c), T=Tc, Tp=ic.T_of(*c))
        print(core, kind, c, 'formula %.4f  num %.4f  shift %.2f' % (dform, dopt, sopt), flush=True)
    return dict(core_path=path, core_diams=dc, results=res)


if __name__ == '__main__':
    kind = sys.argv[1]
    fn = '../outputs/scans_%s.json' % kind
    out = json.load(open(fn)) if os.path.exists(fn) else {}
    cfn = '../outputs/core_diams_cache.json'
    cache = json.load(open(cfn)) if os.path.exists(cfn) else {}
    for core in sys.argv[2:]:
        if core in out:
            continue
        t = time.time()
        out[core] = scan(core, kind, cache)
        out[core]['wall'] = time.time() - t
        json.dump(out, open(fn, 'w'))
        json.dump(cache, open(cfn, 'w'))
