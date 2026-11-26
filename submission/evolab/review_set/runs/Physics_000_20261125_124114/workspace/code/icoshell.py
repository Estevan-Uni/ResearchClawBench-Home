"""Core library: Caspar-Klug (h,k) icosahedral shells, multi-shell clusters,
LJ / Morse / harmonic-contact energies with analytic gradients, relaxation,
chirality-class convention and the analytic vertex-contact mismatch formula.

Conventions
-----------
* Triangular lattice a1=(1,0), a2=(1/2, sqrt3/2); P = h*a1 + k*a2, T = h^2+hk+k^2.
* Shell (h,k) has N = 10T+2 sites; its 12 five-fold vertices lie at radius
  S*a*sqrt(T), S = sin(2pi/5), a = in-shell nearest-neighbour spacing.
* (h,0) and (0,h) are the same Mackay shell; (h,k) and (k,h) are mirror images.
* Class convention (stated in the report): write the shell as (m,n) with
  m=min(h,k), n=max(h,k).  m=0 -> 'MC'; m=n>0 -> 'BG'; 0<m<n -> 'Ch{m}'.
  The class index c=m counts anti-Mackay (h-raising) steps off the Mackay line.
"""
import numpy as np
from scipy.optimize import minimize

S5 = np.sin(2 * np.pi / 5)
SQ3 = np.sqrt(3.0)
A1 = np.array([1.0, 0.0])
A2 = np.array([0.5, SQ3 / 2])


def T_of(h, k):
    return h * h + h * k + k * k


def N_of(h, k):
    return 1 if (h, k) == (0, 0) else 10 * T_of(h, k) + 2


def shell_class(h, k):
    m, n = min(h, k), max(h, k)
    if n == 0:
        return 'C'          # central atom
    if m == 0:
        return 'MC'
    if m == n:
        return 'BG'
    return 'Ch%d' % m


def class_index(h, k):
    return min(h, k)


def hand(h, k):
    """+1 for h<k ('right'), -1 for h>k ('left'), 0 achiral."""
    if h == 0 or k == 0 or h == k:
        return 0
    return 1 if h < k else -1


# ---------------------------------------------------------------- icosahedron
def icosahedron():
    p = (1 + np.sqrt(5)) / 2
    V = []
    for s1 in (-1, 1):
        for s2 in (-1, 1):
            V += [(0, s1, s2 * p), (s1, s2 * p, 0), (s2 * p, 0, s1)]
    V = np.array(V, float)
    V /= np.linalg.norm(V[0])
    # faces: triples with mutual edge length = min edge
    d = np.linalg.norm(V[:, None] - V[None], axis=-1)
    e = d[d > 1e-9].min()
    F = []
    for i in range(12):
        for j in range(i + 1, 12):
            for k in range(j + 1, 12):
                if abs(d[i, j] - e) < 1e-6 and abs(d[j, k] - e) < 1e-6 and abs(d[i, k] - e) < 1e-6:
                    f = [i, j, k]
                    n = np.cross(V[j] - V[i], V[k] - V[i])
                    if np.dot(n, V[i] + V[j] + V[k]) < 0:
                        f = [i, k, j]      # counter-clockwise seen from outside
                    F.append(f)
    assert len(F) == 20
    return V, np.array(F), e


ICO_V, ICO_F, ICO_E = icosahedron()


def shell(h, k, a=1.0, dedup_tol=1e-6):
    """Ideal flat-faced (h,k) shell with in-shell spacing a (vertex radius S a sqrt T)."""
    if (h, k) == (0, 0):
        return np.zeros((1, 3))
    T = T_of(h, k)
    P = h * A1 + k * A2
    c, s = 0.5, SQ3 / 2
    Q = np.array([c * P[0] - s * P[1], s * P[0] + c * P[1]])  # R60 P
    M = np.column_stack([P, Q])
    Minv = np.linalg.inv(M)
    nmax = h + k + 1
    pts2 = []
    for i in range(-nmax, nmax + 1):
        for j in range(-nmax, nmax + 1):
            x = i * A1 + j * A2
            w1, w2 = Minv @ x
            w0 = 1 - w1 - w2
            if min(w0, w1, w2) > -1e-9:
                pts2.append((w0, w1, w2))
    W = np.array(pts2)
    L = a * np.sqrt(T)
    R = L * S5                       # vertex circumradius
    V = ICO_V * R
    out = []
    for f in ICO_F:
        out.append(W @ V[f])
    X = np.vstack(out)
    # dedup
    keep = []
    for x in X:
        if not keep or np.min(np.linalg.norm(np.array(keep) - x, axis=1)) > dedup_tol * max(1, R):
            keep.append(x)
    X = np.array(keep)
    assert len(X) == 10 * T + 2, (h, k, len(X))
    return X


def vertex_mask(X):
    r = np.linalg.norm(X, axis=1)
    return r > r.max() - 1e-6 * max(1, r.max())


# ------------------------------------------------------------ multi-shell
def build_cluster(path, diams):
    """path: list of (h,k) starting with (0,0); diams: particle diameter per shell.
    Shell spacing a_i = d_i ; returns X, per-atom diameter, shell index."""
    X, D, I = [], [], []
    for s, ((h, k), d) in enumerate(zip(path, diams)):
        Y = shell(h, k, a=d)
        X.append(Y)
        D.append(np.full(len(Y), d))
        I.append(np.full(len(Y), s))
    return np.vstack(X), np.concatenate(D), np.concatenate(I)


def vertex_contact_diams(path, d0=1.0):
    """Diameters such that vertex atoms of consecutive shells touch radially and
    in-shell spacing equals the diameter (the analytic model)."""
    d = [d0]
    for s in range(1, len(path)):
        Tp = T_of(*path[s])
        Rprev = 0.0 if s == 1 else S5 * d[-1] * np.sqrt(T_of(*path[s - 1]))
        # S d' sqrtT' - Rprev = (d+d')/2
        dn = (Rprev + d[-1] / 2) / (S5 * np.sqrt(Tp) - 0.5)
        d.append(dn)
    return np.array(d)


def delta_star(T, Tp):
    """Analytic vertex-contact optimal mismatch for step T -> T' (T=0 central atom)."""
    # valid also for T=0 (central atom: previous radius 0)
    return (1 - S5 * (np.sqrt(Tp) - np.sqrt(T))) / (S5 * np.sqrt(Tp) - 0.5)


def delta_star_limit(T, Tp):
    """Large-T limit (1 - S cos(theta))/(S sqrt T), cos(theta)=dsqrtT."""
    return (1 - S5 * (np.sqrt(Tp) - np.sqrt(T))) / (S5 * np.sqrt(T))


def step_type(a, b):
    """Classify step between shells a=(h,k) -> b=(h',k') using canonical (m<=n) forms
    and the class convention.  Returns 'conservative', 'class-raising',
    'class-lowering', or 'non-NN'."""
    (h, k), (h2, k2) = a, b
    dv = np.array([h2 - h, k2 - k])
    nn = [(1, 0), (0, 1), (-1, 1), (1, -1), (-1, 0), (0, -1)]
    # allow mirror of target
    isnn = tuple(dv) in nn or (k2 - h, h2 - k) in nn
    if not isnn:
        return 'non-NN'
    ca, cb = class_index(h, k), class_index(h2, k2)
    if cb == ca:
        return 'conservative'
    return 'class-raising' if cb > ca else 'class-lowering'


def outward_nn_steps(h, k):
    """Distinct outward nearest-neighbour steps from (h,k) (T'>T), as canonical
    representatives with chirality kept (h,k orientation)."""
    T = T_of(h, k)
    cand = [(h, k + 1), (h + 1, k), (h - 1, k + 1), (h + 1, k - 1)]
    out = []
    for c in cand:
        if min(c) < 0:
            continue
        if T_of(*c) <= T:
            continue
        # identify same shells: (n,0)==(0,n); avoid duplicates
        key = tuple(sorted(c)) if 0 in c else c
        if key not in [tuple(sorted(o)) if 0 in o else o for o in out]:
            out.append(c)
    return out


def step_cos(a, b):
    P = a[0] * A1 + a[1] * A2
    Q = b[0] * A1 + b[1] * A2
    dv = Q - P
    if np.linalg.norm(P) == 0:
        return 1.0
    return float(np.dot(P, dv) / np.linalg.norm(P) / np.linalg.norm(dv))


# ------------------------------------------------------------- potentials
def pair_lists(n):
    i, j = np.triu_indices(n, 1)
    return i, j


class PairPot:
    """Pair potential with per-pair equilibrium distance r0_ij = (d_i+d_j)/2."""

    def __init__(self, D, kind='LJ', rho=6.0, pairs=None, kspring=1.0):
        self.D = np.asarray(D, float)
        self.kind = kind
        self.rho = rho
        n = len(D)
        if pairs is None:
            self.i, self.j = pair_lists(n)
        else:
            self.i, self.j = pairs
        self.r0 = 0.5 * (self.D[self.i] + self.D[self.j])
        self.n = n

    def eg(self, x):
        X = x.reshape(-1, 3)
        dv = X[self.i] - X[self.j]
        r = np.sqrt((dv * dv).sum(1))
        s = r / self.r0
        if self.kind == 'LJ':          # min at r0, depth 1
            u6 = s ** -6
            e = u6 * u6 - 2 * u6
            de = (-12 * u6 * u6 + 12 * u6) / r       # dE/dr
        elif self.kind == 'Morse':
            ex = np.exp(self.rho * (1 - s))
            e = ex * (ex - 2)
            de = (-2 * self.rho / self.r0) * ex * (ex - 1)
        elif self.kind == 'harm':
            e = (s - 1) ** 2
            de = 2 * (s - 1) / self.r0
        E = e.sum()
        f = (de / r)[:, None] * dv
        G = np.zeros_like(X)
        np.add.at(G, self.i, f)
        np.add.at(G, self.j, -f)
        return E, G.ravel()


def relax(X, pot, gtol=1e-6, maxiter=5000):
    res = minimize(pot.eg, X.ravel(), jac=True, method='L-BFGS-B',
                   options={'gtol': gtol, 'maxiter': maxiter, 'maxcor': 30})
    return res.fun, res.x.reshape(-1, 3)


def contact_pairs(X, D, tol=1.15):
    n = len(X)
    i, j = np.triu_indices(n, 1)
    r = np.linalg.norm(X[i] - X[j], axis=1)
    r0 = 0.5 * (D[i] + D[j])
    m = r < tol * r0
    return i[m], j[m]


def max_angular_shift(X0, X1, mask):
    a = X0[mask] / np.linalg.norm(X0[mask], axis=1)[:, None]
    b = X1[mask] / np.linalg.norm(X1[mask], axis=1)[:, None]
    c = np.clip((a * b).sum(1), -1, 1)
    return np.degrees(np.arccos(c)).max()


def cluster_energy(path, diams, kind='LJ', rho=6.0, return_X=False, scale_lj=True):
    """Relaxed energy of the multi-shell cluster.  For 'LJ' the classic 12-6 with
    sigma_ij = r0_ij/2^(1/6) (min at r0, depth 1)."""
    X, D, I = build_cluster(path, diams)
    if kind == 'harm':
        pairs = contact_pairs(X, D)
        pot = PairPot(D, 'harm', pairs=pairs)
    else:
        pot = PairPot(D, kind, rho=rho)
    E, X1 = relax(X, pot)
    if return_X:
        return E, X, X1, D, I
    return E
