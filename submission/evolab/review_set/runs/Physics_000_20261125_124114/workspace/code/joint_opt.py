"""Joint Nelder-Mead optimisation of all shell diameters (log-space) of a whole
path, LJ (equal eps).  Central atom d0=1 fixed.  Saves ../outputs/joint_seeds.json"""
import json, sys, os, numpy as np, time
from scipy.optimize import minimize
import icoshell as ic

PATHS = {
    'p_011': [(0, 0), (0, 1), (1, 1)],
    'p_01112': [(0, 0), (0, 1), (1, 1), (1, 2)],
    'p_0111213': [(0, 0), (0, 1), (1, 1), (1, 2), (1, 3)],
    'p_010203': [(0, 0), (0, 1), (0, 2), (0, 3)],
    'p_01021213': [(0, 0), (0, 1), (0, 2), (1, 2), (1, 3)],
    'p_zigzag22': [(0, 0), (0, 1), (1, 1), (1, 2), (2, 2)],
    'p_0102031323': [(0, 0), (0, 1), (0, 2), (0, 3), (1, 3), (2, 3)],
}


def optimise(path, x0=None, kind='LJ'):
    d_vc = ic.vertex_contact_diams(path)
    if x0 is None:
        x0 = np.log(d_vc[1:])
    cache = {}

    def f(x):
        d = np.concatenate([[1.0], np.exp(x)])
        return ic.cluster_energy(path, d, kind=kind)
    r = minimize(f, x0, method='Nelder-Mead',
                 options={'xatol': 1e-4, 'fatol': 1e-6, 'maxfev': 400, 'initial_simplex':
                          np.vstack([x0] + [x0 + 0.02 * np.eye(len(x0))[i] for i in range(len(x0))])})
    d = np.concatenate([[1.0], np.exp(r.x)])
    deltas = d[1:] / d[:-1] - 1
    return dict(path=path, diams=d.tolist(), deltas=deltas.tolist(), E=float(r.fun),
                d_vertex_contact=d_vc.tolist(),
                deltas_vertex_contact=(d_vc[1:] / d_vc[:-1] - 1).tolist(), nfev=r.nfev)


if __name__ == '__main__':
    fn = '../outputs/joint_seeds.json'
    J = json.load(open(fn)) if os.path.exists(fn) else {}
    names = sys.argv[1:] or list(PATHS)
    for n in names:
        if n in J:
            continue
        t = time.time()
        path = PATHS[n]
        # warm start from shorter path if available
        x0 = None
        for m, v in J.items():
            if v['path'] == [list(p) for p in path[:-1]] or [tuple(p) for p in v['path']] == path[:-1]:
                dvc = ic.vertex_contact_diams(path)
                x0 = np.log(np.concatenate([v['diams'][1:], [v['diams'][-1] * dvc[-1] / dvc[-2]]]))
        J[n] = optimise(path, x0)
        J[n]['wall'] = time.time() - t
        json.dump(J, open(fn, 'w'), indent=1)
        print(n, J[n]['deltas'], J[n]['deltas_vertex_contact'], '%.0fs' % J[n]['wall'], flush=True)
