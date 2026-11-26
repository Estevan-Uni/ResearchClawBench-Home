"""Langevin (BAOAB) growth simulations of multi-component LJ clusters with
atom-by-atom deposition, plus structural identification of the grown layer.

LJ: eps=1 for all pairs, minimum at r0_ij=(d_i+d_j)/2 (sigma_ij=r0_ij/2^(1/6)),
cutoff 2.5 sigma_ij, energy shifted.  Unit masses, gamma=1/tau, dt=0.005.
"""
import numpy as np, json, time, sys, os
from scipy.spatial import cKDTree
from scipy.optimize import minimize
import icoshell as ic

RC_FAC = 2.5 / 2 ** (1 / 6)      # cutoff in units of r0_ij


class NLForce:
    def __init__(self, D, skin=0.4):
        self.D = np.asarray(D, float)
        self.skin = skin
        self.Xref = None

    def set_D(self, D):
        self.D = np.asarray(D, float)
        self.Xref = None

    def build(self, X):
        rmax = RC_FAC * self.D.max() + self.skin
        t = cKDTree(X)
        p = t.query_pairs(rmax, output_type='ndarray')
        self.i, self.j = p[:, 0], p[:, 1]
        r0 = 0.5 * (self.D[self.i] + self.D[self.j])
        self.r0 = r0
        self.rc2 = (RC_FAC * r0) ** 2
        sc = 1 / RC_FAC ** 6
        self.eshift = sc * sc - 2 * sc
        self.Xref = X.copy()

    def __call__(self, X):
        if self.Xref is None or len(self.Xref) != len(X) or \
                np.max(np.sum((X - self.Xref) ** 2, 1)) > (0.5 * self.skin) ** 2:
            self.build(X)
        dv = X[self.i] - X[self.j]
        r2 = np.einsum('ij,ij->i', dv, dv)
        m = r2 < self.rc2
        s2 = r2[m] / self.r0[m] ** 2
        u6 = s2 ** -3
        e = (u6 * u6 - 2 * u6 - self.eshift).sum()
        fr = (12 * u6 * u6 - 12 * u6) / r2[m]          # -dE/dr / r
        f = fr[:, None] * dv[m]
        n = len(X)
        F = np.empty_like(X)
        ii, jj = self.i[m], self.j[m]
        for c in range(3):
            F[:, c] = np.bincount(ii, f[:, c], n) - np.bincount(jj, f[:, c], n)
        return e, F


def quench(X, D):
    f = NLForce(D, skin=1.0)

    def eg(x):
        e, F = f(x.reshape(-1, 3))
        return e, -F.ravel()
    r = minimize(eg, X.ravel(), jac=True, method='L-BFGS-B',
                 options={'gtol': 1e-5, 'maxiter': 4000, 'maxcor': 30})
    return r.fun, r.x.reshape(-1, 3)


def full_energy(X, D):
    """Energy with the same truncated-shifted LJ (for comparing to ideal)."""
    f = NLForce(D, skin=1.0)
    return f(X)[0]


# --------------------------------------------------------------- templates
def relaxed_template(path, diams):
    X, D, I = ic.build_cluster(path, diams)
    E, X1 = quench(X, D)
    return E, X1, D, I


def kabsch(P, Q):
    """rotation R minimising |R P - Q| (rows are points)."""
    H = P.T @ Q
    U, s, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    Dm = np.diag([1, 1, d])
    return Vt.T @ Dm @ U.T


def core_frame(X, centre, nn=12):
    """Permutation-free fit of the 12 nearest neighbours of `centre` onto the
    canonical icosahedron directions; returns rotation R (x_frame = R x) and rms."""
    c = X[centre]
    d = np.linalg.norm(X - c, axis=1)
    d[centre] = 1e9
    idx = np.argsort(d)[:nn]
    U = X[idx] - c
    U /= np.linalg.norm(U, axis=1)[:, None]
    V = ic.ICO_V
    best = (1e9, None)
    # candidate initial alignments: u0->V0 and a neighbour of u0 -> neighbour of V0
    vn = [j for j in range(12) if abs(np.dot(V[0], V[j]) - 1 / np.sqrt(5)) < 1e-3]
    for a in range(nn):
        cosang = U @ U[a]
        nbrs = np.argsort(-cosang)[1:6]
        for b in nbrs[:2]:
            P = np.array([U[a], U[b], np.cross(U[a], U[b])])
            Q = np.array([V[0], V[vn[0]], np.cross(V[0], V[vn[0]])])
            R = kabsch(P, Q)
            for _ in range(10):
                W = U @ R.T
                asg = np.argmax(W @ V.T, axis=1)
                if len(set(asg)) < nn:
                    break
                R = kabsch(U, V[asg])
            W = U @ R.T
            rms = np.sqrt(np.mean(np.min(np.sum((W[:, None] - V[None]) ** 2, -1), 1)))
            if rms < best[0]:
                best = (rms, R)
        if best[0] < 0.05:
            break
    return best[1], best[0]


def unit(X):
    return X / np.linalg.norm(X, axis=1)[:, None]


def min_ang_sep(Y):
    c = Y @ Y.T
    np.fill_diagonal(c, -1)
    return np.degrees(np.arccos(np.clip(c.max(), -1, 1)))


def coverage(layer_dirs, tmpl_dirs, tol_deg, both=False):
    """fraction of template sites matched one-to-one by a layer atom within tol
    (recall); with both=True also returns precision = matched / layer atoms."""
    if len(layer_dirs) == 0:
        return (0.0, 0.0) if both else 0.0
    c = np.clip(tmpl_dirs @ layer_dirs.T, -1, 1)
    ang = np.degrees(np.arccos(c))         # (Nt, Nl)
    pairs = np.argwhere(ang < tol_deg)
    order = np.argsort(ang[pairs[:, 0], pairs[:, 1]])
    ut, ul = set(), set()
    for p in pairs[order]:
        if p[0] not in ut and p[1] not in ul:
            ut.add(p[0]); ul.add(p[1])
    if both:
        return len(ut) / len(tmpl_dirs), len(ut) / len(layer_dirs)
    return len(ut) / len(tmpl_dirs)


def chirality_index(layer_dirs, tA, tB, vert_dirs, vtol=8.0):
    """(nA-nB)/(nA+nB) over layer atoms not near the 12 shared vertices;
    each atom assigned to the template (hand) with the nearer site."""
    if len(layer_dirs) == 0:
        return 0.0, 0
    cv = np.degrees(np.arccos(np.clip((layer_dirs @ vert_dirs.T).max(1), -1, 1)))
    L = layer_dirs[cv > vtol]
    if len(L) == 0:
        return 0.0, 0
    aA = np.arccos(np.clip((L @ tA.T).max(1), -1, 1))
    aB = np.arccos(np.clip((L @ tB.T).max(1), -1, 1))
    nA, nB = np.sum(aA < aB), np.sum(aB < aA)
    return float((nA - nB) / max(nA + nB, 1)), int(nA + nB)


class Identifier:
    """Templates (directions in the canonical frame) of candidate outer shells."""

    def __init__(self, seed_path, seed_diams, d_new, candidates, tol_frac=0.35):
        self.cands = {}
        for c in candidates:
            E, X1, D, I = relaxed_template(seed_path + [c], list(seed_diams) + [d_new])
            Y = unit(X1[I == len(seed_path)])
            self.cands[c] = dict(dirs=Y, tol=tol_frac * min_ang_sep(Y), E=E, N=len(Y))
        self.vert = ic.ICO_V

    def identify(self, X, centre, layer_idx):
        if centre == 'auto':
            centre = auto_centre(X)
        if isinstance(layer_idx, str) and layer_idx == 'radial':
            d = np.linalg.norm(X - X[centre], axis=1)
            o = np.argsort(d)
            layer_idx = o[13:]
        R, rms = core_frame(X, centre)
        com = X[centre]
        V = (X[layer_idx] - com) @ R.T
        V = V[np.linalg.norm(V, axis=1) > 0.3]        # exclude an atom sitting at the centre
        L = unit(V)
        cp = {str(c): coverage(L, v['dirs'], v['tol'], both=True) for c, v in self.cands.items()}
        cov = {k: v[0] for k, v in cp.items()}
        prec = {k: v[1] for k, v in cp.items()}
        f1 = {k: (2 * v[0] * v[1] / (v[0] + v[1]) if v[0] + v[1] > 0 else 0.0) for k, v in cp.items()}
        out = dict(cov=cov, prec=prec, f1=f1, n_layer=int(len(L)), frame_rms=float(rms), centre=int(centre), n_bound=n_bound(X))
        # chirality between mirror pairs present
        for c in self.cands:
            m = (c[1], c[0])
            if c[0] < c[1] and m in self.cands:
                ci, n = chirality_index(L, self.cands[c]['dirs'], self.cands[m]['dirs'], self.vert)
                out['chi_%d%d' % c] = ci
                out['chi_n_%d%d' % c] = n
        return out


# --------------------------------------------------------------- dynamics
def robust_com(X):
    """COM of the bound cluster (outliers beyond 2 units of the bulk removed);
    returns com and max radius of bound atoms."""
    med = np.median(X, 0)
    r = np.linalg.norm(X - med, axis=1)
    b = r < np.percentile(r, 90) + 2.0
    com = X[b].mean(0)
    r = np.linalg.norm(X[b] - com, axis=1)
    return com, r.max()


def auto_centre(X):
    com, _ = robust_com(X)
    return int(np.argmin(np.linalg.norm(X - com, axis=1)))


def n_bound(X, cut=1.6):
    """size of the largest connected component (bond cutoff in absolute units)"""
    import scipy.sparse.csgraph as cg
    t = cKDTree(X)
    A = t.sparse_distance_matrix(t, cut)
    nc, lab = cg.connected_components(A, directed=False)
    return int(np.bincount(lab).max())
def run_growth(seed_X, seed_D, d_dep, n_dep, T=0.22, tau_dep=60.0, dt=0.005, gamma=1.0,
               rng=None, ident=None, ident_every=10, centre=0, anneal_tau=100.0,
               cool_tau=100.0, d_dep_seq=None, log=None, snap_every=None,
               id_centre=None, layer='index'):
    """seed atoms first; deposits atoms (diameter d_dep or sequence d_dep_seq).
    Returns dict with history and final structures."""
    rng = np.random.default_rng(rng)
    idc = centre if id_centre is None else id_centre
    lay = (lambda n: np.arange(nseed, n)) if layer == 'index' else (lambda n: 'radial')
    X = seed_X.copy() - seed_X.mean(0)
    D = np.array(seed_D, float)
    nseed = len(X)
    V = rng.normal(scale=np.sqrt(T), size=X.shape)
    force = NLForce(D)
    e, F = force(X)
    c1 = np.exp(-gamma * dt)
    c2 = np.sqrt(1 - c1 * c1)
    steps_dep = int(round(tau_dep / dt))
    hist = []
    snaps = []

    def bstep(X, V, F, Tcur, nsteps):
        for s in range(nsteps):
            V += 0.5 * dt * F
            X += 0.5 * dt * V
            V *= c1
            V += c2 * np.sqrt(Tcur) * rng.normal(size=V.shape)
            X += 0.5 * dt * V
            e, F = force(X)
            V += 0.5 * dt * F
            if s % 50 == 0:
                # recentre on robust COM, reflecting wall
                com, rb = robust_com(X)
                X -= com
                r = np.linalg.norm(X, axis=1)
                rw = rb + 4.0
                out = r > rw
                if out.any():
                    rh = X[out] / r[out][:, None]
                    vr = (V[out] * rh).sum(1)
                    V[out] -= (2 * np.maximum(vr, 0))[:, None] * rh
        return X, V, F, e

    t0 = time.time()
    for k in range(n_dep):
        dk = d_dep if d_dep_seq is None else d_dep_seq[k]
        com, Rmax = robust_com(X)
        X -= com
        u = rng.normal(size=3); u /= np.linalg.norm(u)
        xnew = u * (Rmax + 1.3 * max(dk, D.max()))
        vnew = -u * np.sqrt(2 * T) + rng.normal(scale=np.sqrt(T), size=3) * 0.5
        X = np.vstack([X, xnew]); V = np.vstack([V, vnew]); D = np.append(D, dk)
        force.set_D(D)
        e, F = force(X)
        X, V, F, e = bstep(X, V, F, T, steps_dep)
        if ident is not None and ((k + 1) % ident_every == 0 or k == n_dep - 1):
            Eq, Xq = quench(X, D)
            idr = ident.identify(Xq, idc, lay(len(X)))
            idr.update(n=k + 1, Eq=float(Eq), t=time.time() - t0)
            hist.append(idr)
            if log:
                print(json.dumps(idr), file=log, flush=True)
        if snap_every and (k + 1) % snap_every == 0:
            snaps.append(dict(n=k + 1, X=X.tolist(), D=D.tolist()))
    # as-grown quench
    Eg, Xg = quench(X, D)
    res = dict(hist=hist, E_grown=float(Eg), X_grown=Xg.tolist(), D=D.tolist(), nseed=nseed)
    if ident is not None:
        res['id_grown'] = ident.identify(Xg, idc, lay(len(X)))
    # final anneal at T then linear cool to 0.05
    if anneal_tau > 0:
        X, V, F, e = bstep(X, V, F, T, int(anneal_tau / dt))
        nc = int(cool_tau / dt)
        nb = 20
        for b in range(nb):
            Tb = T + (0.05 - T) * (b + 1) / nb
            X, V, F, e = bstep(X, V, F, Tb, nc // nb)
        Ea, Xa = quench(X, D)
        res.update(E_anneal=float(Ea), X_anneal=Xa.tolist())
        if ident is not None:
            res['id_anneal'] = ident.identify(Xa, idc, lay(len(X)))
    res['wall'] = time.time() - t0
    res['snaps'] = snaps
    return res
