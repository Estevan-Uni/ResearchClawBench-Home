"""Statistical layer: survival probability, comparison approximations, Bayes factor with Regge pile-up."""
import numpy as np
import physics as ph

# ---------------- timescale conventions ----------------
T_SALPETER = 4.5e7
AGES = {  # (fiducial, list for log-uniform marginalisation)
    "M33 X-7": dict(fid=T_SALPETER, alt=3e6, marg=np.logspace(5, np.log10(3e6), 9)),
    "IRAS 09149-6206": dict(fid=T_SALPETER, alt=1e10, marg=np.logspace(6, np.log10(4.5e7), 9)),
}
MU_RANGE = {"M33 X-7": (1e-14, 1e-10), "IRAS 09149-6206": (1e-21, 1e-17)}


def p_surv(M, a, mu, tage, w=None, **kw):
    """Posterior probability that the BH has NOT been spun down: 1 - P[(M,a) in unstable region]."""
    u = ph.unstable(M, a, mu, tage, **kw)
    if w is None:
        return 1.0 - u.mean()
    return 1.0 - np.sum(w * u) / np.sum(w)


def p_surv_curve(M, a, mus, tage, **kw):
    if np.ndim(tage) == 0:
        return np.array([p_surv(M, a, mu, tage, **kw) for mu in mus])
    return np.mean([p_surv_curve(M, a, mus, t, **kw) for t in tage], axis=0)


def gaussian_draws(M, a, n=20000, diag=False, seed=1):
    """Gaussian approximation in (ln M, a) with full or diagonal covariance, truncated to 0<a<1."""
    rng = np.random.default_rng(seed)
    X = np.c_[np.log(M), a]
    mu = X.mean(0); C = np.cov(X.T)
    if diag:
        C = np.diag(np.diag(C))
    out = []
    while sum(len(o) for o in out) < n:
        d = rng.multivariate_normal(mu, C, size=n)
        d = d[(d[:, 1] > 0) & (d[:, 1] < 1)]
        out.append(d)
    d = np.concatenate(out)[:n]
    return np.exp(d[:, 0]), d[:, 1]


def shuffled(M, a, seed):
    rng = np.random.default_rng(seed)
    return M, rng.permutation(a)


def point_rule(M, a, mus, tage, which="median", **kw):
    """1 if the point estimate lies in the stable region (P_surv=1) else 0."""
    if which == "median":
        m0, a0 = np.median(M), np.median(a)
    else:
        m0, a0 = np.exp(np.mean(np.log(M))), np.mean(a)
    return np.array([0.0 if ph.unstable(np.array([m0]), np.array([a0]), mu, tage, **kw)[0] else 1.0 for mu in mus])


def box_rule(M, a, mus, tage, nsig=1.0, ngrid=7, **kw):
    """Exclude mu only if the whole +-nsig box in (ln M, a) is unstable; returns fraction of box stable."""
    lm, s1 = np.mean(np.log(M)), np.std(np.log(M)); am, s2 = np.mean(a), np.std(a)
    g1 = np.exp(np.linspace(lm - nsig * s1, lm + nsig * s1, ngrid))
    g2 = np.clip(np.linspace(am - nsig * s2, am + nsig * s2, ngrid), 0, 0.9999)
    GM, GA = np.meshgrid(g1, g2)
    return np.array([1 - ph.unstable(GM.ravel(), GA.ravel(), mu, tage, **kw).mean() for mu in mus])


# ---------------- KDE density in (ln M, a), reflected at a=1 ----------------
class KDE2:
    def __init__(self, M, a, bw_scale=1.0):
        self.x = np.log(M); self.y = a
        n = len(M)
        C = np.cov(np.c_[self.x, self.y].T)
        h = n ** (-1 / 6) * bw_scale  # Scott factor, d=2
        self.C = C * h ** 2
        self.Ci = np.linalg.inv(self.C)
        self.norm = 1 / (2 * np.pi * np.sqrt(np.linalg.det(self.C)) * n)

    def __call__(self, lx, y, chunk=4000):
        lx = np.atleast_1d(lx); y = np.atleast_1d(y)
        out = np.empty(len(lx))
        for s in range(0, len(lx), chunk):
            dx = lx[s:s + chunk, None] - self.x[None]
            tot = 0
            for yy in (self.y, 2 - self.y):  # reflection at a=1
                dy = y[s:s + chunk, None] - yy[None]
                q = self.Ci[0, 0] * dx ** 2 + 2 * self.Ci[0, 1] * dx * dy + self.Ci[1, 1] * dy ** 2
                tot = tot + np.exp(-0.5 * q).sum(1)
            out[s:s + chunk] = tot * self.norm
        return out


class GaussDens:
    """Analytic Gaussian density in (ln M, a) (for injection mocks)."""
    def __init__(self, mean, C):
        self.m = mean; self.Ci = np.linalg.inv(C); self.norm = 1 / (2 * np.pi * np.sqrt(np.linalg.det(C)))

    def __call__(self, lx, y):
        dx = lx - self.m[0]; dy = y - self.m[1]
        q = self.Ci[0, 0] * dx ** 2 + 2 * self.Ci[0, 1] * dx * dy + self.Ci[1, 1] * dy ** 2
        return self.norm * np.exp(-0.5 * q)


def pileup_weights(M, mu, tage, na0=400, lmax=ph.LMAX, **kw):
    """For each M, fraction of a0~U(0,1) that ends on each Regge trajectory l after tage.
    Returns w[l-1, iM], asat[l-1, iM]."""
    a0 = (np.arange(na0) + 0.5) / na0
    MM = np.repeat(M[:, None], na0, 1); AA = np.repeat(a0[None], len(M), 0)
    af = ph.regge_spin(MM, AA, mu, tage, lmax=lmax, **kw)
    alpha = ph.ALPHA_PER_MSUN_EV * M * mu
    W = np.zeros((lmax, len(M))); AS = np.zeros((lmax, len(M)))
    moved = af < AA - 1e-12
    assigned = np.zeros_like(moved)
    for l in range(lmax, 0, -1):  # final spins equal to a_sat(l)
        asl = ph.a_sat(alpha, l)
        on = moved & (np.abs(af - asl[:, None]) < 1e-9) & ~assigned
        assigned |= on
        W[l - 1] = on.mean(1); AS[l - 1] = asl
    return W, AS


def bayes_factor(M, a, mu, tage, dens, lgrid, **kw):
    """B(mu) = p(d|H_mu)/p(d|H_0) with a0~U(0,1) prior: P_surv + sum_l int dlnM w_l(M) p(lnM, a_l(M))."""
    ps = p_surv(M, a, mu, tage, **kw)
    Mg = np.exp(lgrid)
    W, AS = pileup_weights(Mg, mu, tage, **kw)
    pile = 0.0
    dl = lgrid[1] - lgrid[0]
    for l in range(W.shape[0]):
        if W[l].max() <= 0:
            continue
        pile += np.sum(W[l] * dens(lgrid, AS[l])) * dl
    return ps + pile, ps, pile


def intervals(mus, excluded):
    """Contiguous excluded intervals (geometric midpoints at edges)."""
    out = []; i = 0; n = len(mus)
    while i < n:
        if excluded[i]:
            j = i
            while j + 1 < n and excluded[j + 1]:
                j += 1
            lo = mus[i] if i == 0 else np.sqrt(mus[i - 1] * mus[i])
            hi = mus[j] if j == n - 1 else np.sqrt(mus[j] * mus[j + 1])
            out.append((lo, hi)); i = j + 1
        else:
            i += 1
    return out


def fmt_int(iv):
    return " ∪ ".join(f"[{lo:.2e}, {hi:.2e}]" for lo, hi in iv) if iv else "none"
