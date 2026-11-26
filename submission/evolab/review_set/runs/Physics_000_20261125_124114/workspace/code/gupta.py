"""Gupta / SMATB (second-moment tight-binding) energies for Ni, Cu, Ag core@shell
icosahedral clusters, with a smooth cosine taper (2.3 -> 2.6 r0; reproduces the fcc cohesive energies, Ni 4.45, Cu 3.55, Ag 2.96 eV) instead of a
hard cutoff.  Pure-metal parameters: Cleri & Rosato, PRB 48, 22 (1993).
Cross parameters, two sets (sensitivity test):
  'avg' : arithmetic mean of the pure-metal A, xi, p, q, r0
  'lit' : Ag-Ni / Ag-Cu values as recalled from the Ferrando-group literature
          (Rossi et al. PRL 93, 105503 (2004); Baletto et al.) -- provenance could
          NOT be verified offline, used only as a plausible stronger-binding variant;
          Cu-Ni 'lit' uses geometric-mean A, xi and arithmetic p, q, r0.
usage: python gupta.py"""
import json, numpy as np, itertools, time
from scipy.optimize import minimize
import icoshell as ic

PURE = {  # A(eV), xi(eV), p, q, r0(A)
    'Ni': (0.0376, 1.070, 16.999, 1.189, 2.4916),
    'Cu': (0.0855, 1.224, 10.960, 2.278, 2.5562),
    'Ag': (0.1028, 1.178, 10.928, 3.139, 2.8890),
}
ECOH = {'Ni': 4.447, 'Cu': 3.545, 'Ag': 2.960}   # model fcc bulk energies with this taper
LIT = {('Ag', 'Ni'): (0.0960, 1.3402, 11.3480, 2.3050, 2.6900),
       ('Ag', 'Cu'): (0.0980, 1.2274, 10.7000, 2.8050, 2.7225)}


def cross(a, b, mode):
    if a == b:
        return PURE[a]
    key = tuple(sorted((a, b)))
    if mode == 'lit' and key in LIT:
        return LIT[key]
    pa, pb = np.array(PURE[a]), np.array(PURE[b])
    m = 0.5 * (pa + pb)
    if mode == 'lit':
        m[0] = np.sqrt(pa[0] * pb[0]); m[1] = np.sqrt(pa[1] * pb[1])
    return tuple(m)


class Gupta:
    def __init__(self, species, mode='avg'):
        self.sp = list(species)
        n = len(species)
        self.i, self.j = np.triu_indices(n, 1)
        P = np.array([cross(self.sp[a], self.sp[b], mode) for a, b in zip(self.i, self.j)])
        self.A, self.xi, self.p, self.q, self.r0 = P.T
        self.n = n

    def eg(self, x):
        X = x.reshape(-1, 3)
        dv = X[self.i] - X[self.j]
        r = np.sqrt((dv * dv).sum(1))
        s = r / self.r0
        c1, c2 = 2.3, 2.6
        t = np.clip((s - c1) / (c2 - c1), 0, 1)
        fc = 0.5 * (1 + np.cos(np.pi * t))
        dfc = np.where((t > 0) & (t < 1), -0.5 * np.pi * np.sin(np.pi * t) / (c2 - c1) / self.r0, 0.0)
        er = self.A * np.exp(-self.p * (s - 1))
        eb = self.xi ** 2 * np.exp(-2 * self.q * (s - 1))
        rep = er * fc; bnd = eb * fc
        drep = er * (-self.p / self.r0) * fc + er * dfc
        dbnd = eb * (-2 * self.q / self.r0) * fc + eb * dfc
        n = self.n
        Rho = np.bincount(self.i, bnd, n) + np.bincount(self.j, bnd, n)
        Rep = np.bincount(self.i, rep, n) + np.bincount(self.j, rep, n)
        sq = np.sqrt(np.maximum(Rho, 1e-12))
        E = Rep.sum() - sq.sum()
        # dE/dr_ij = 2 drep - 0.5 dbnd (1/sq_i + 1/sq_j)
        dEdr = 2 * drep - 0.5 * dbnd * (1 / sq[self.i] + 1 / sq[self.j])
        f = (dEdr / r)[:, None] * dv
        G = np.zeros_like(X)
        for c in range(3):
            G[:, c] = np.bincount(self.i, f[:, c], n) - np.bincount(self.j, f[:, c], n)
        return E, G.ravel()


def relax(X, pot):
    r = minimize(pot.eg, X.ravel(), jac=True, method='L-BFGS-B',
                 options={'gtol': 1e-5, 'maxiter': 5000, 'maxcor': 30})
    return r.fun, r.x.reshape(-1, 3)


MACKAY = {13: [(0, 0), (0, 1)], 55: [(0, 0), (0, 1), (0, 2)], 147: [(0, 0), (0, 1), (0, 2), (0, 3)]}
OUTER = {13: [(0, 2), (1, 1)], 55: [(0, 3), (1, 2)], 147: [(0, 4), (1, 3), (2, 2), (2, 3)]}


def build(core_sp, shell_sp, Ncore, outer, mode):
    path = MACKAY[Ncore] + [outer]
    dc, ds = PURE[core_sp][4], PURE[shell_sp][4]
    X, D, I = ic.build_cluster(path, [dc] * len(MACKAY[Ncore]) + [ds])
    # make sure the shell does not overlap: put outer vertex radius >= core radius + contact
    Rc = np.linalg.norm(X[I < I.max()], axis=1).max()
    Xo = X[I == I.max()]
    Ro = np.linalg.norm(Xo, axis=1).max()
    rmin = Rc + 0.5 * (dc + ds) * 0.95
    if Ro < rmin:
        X[I == I.max()] *= rmin / Ro
    sp = [core_sp] * int((I < I.max()).sum()) + [shell_sp] * int((I == I.max()).sum())
    pot = Gupta(sp, mode)
    E, X1 = relax(X, pot)
    N = len(sp)
    Delta = (E + sum(ECOH[s] for s in sp)) / N ** (2 / 3)
    shift = ic.max_angular_shift(X, X1, I == I.max())
    # core alone (relaxed) -> per-shell-atom excess energy relative to bulk shell metal
    Xc = X[I < I.max()]
    Ec, _ = relax(Xc, Gupta([core_sp] * len(Xc), mode))
    Ns = int((I == I.max()).sum())
    eps_shell = (E - Ec) / Ns + ECOH[shell_sp]
    return dict(E=float(E), Delta=float(Delta), N=N, shift=float(shift), eps_shell=float(eps_shell), Ns=Ns)


if __name__ == '__main__':
    # gradient check
    sp = ['Ni'] * 13 + ['Ag'] * 32
    X, D, I = ic.build_cluster([(0, 0), (0, 1), (1, 1)], [2.49, 2.49, 2.89])
    g = Gupta(sp, 'lit'); x = X.ravel() + 0.02 * np.random.default_rng(0).normal(size=X.size)
    E, G = g.eg(x); errs = []
    for k in [0, 7, 50, 100, 130]:
        e = np.zeros_like(x); e[k] = 1e-6
        errs.append(abs((g.eg(x + e)[0] - g.eg(x - e)[0]) / 2e-6 - G[k]) / max(abs(G[k]), 1e-8))
    print('gupta grad rel err', max(errs))
    # pure-metal sanity: Mackay 13/55/147 energies per atom
    out = dict(grad_err=max(errs), pure={}, cases=[])
    for m in PURE:
        for Nc, path in MACKAY.items():
            X, D, I = ic.build_cluster(path, [PURE[m][4]] * len(path))
            E, X1 = relax(X, Gupta([m] * len(X)))
            out['pure']['%s%d' % (m, Nc)] = E / len(X)
    print(out['pure'])
    pairs = [('Ni', 'Ag'), ('Cu', 'Ag'), ('Ni', 'Cu')]
    for mode in ['avg', 'lit']:
        for core, shl in pairs:
            for Nc in [13, 55, 147]:
                for o in OUTER[Nc]:
                    t = time.time()
                    r = build(core, shl, Nc, o, mode)
                    r.update(mode=mode, core=core, shell=shl, Ncore=Nc, outer=o,
                             step=ic.step_type(MACKAY[Nc][-1], o))
                    out['cases'].append(r)
                    print(mode, core, shl, Nc, o, r['N'], '%.4f' % r['Delta'], '%.4f' % r['eps_shell'], '%.1f' % r['shift'],
                          '%.0fs' % (time.time() - t), flush=True)
                    json.dump(out, open('../outputs/gupta_results.json', 'w'), indent=1)
