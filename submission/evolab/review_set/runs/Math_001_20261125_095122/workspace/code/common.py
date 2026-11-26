"""Common utilities for the VOS (Variable and Operator Splitting) study.

Problem class:  min_x  F(x) = f(x) + g(x)
    f(x) = 1/2 ||A x - b||^2 + (mu/2) ||x||^2      (smooth, L-smooth, mu-convex)
    g(x) = lam ||x||_1                              (non-smooth, handled by prox)

lam = 0, mu > 0  -> ridge;  lam > 0, mu = 0 -> Lasso;  both > 0 -> elastic net.

All first-order methods track A*(iterate) by linearity so that every method
costs exactly 2 matrix-vector products (one A, one A^T) per iteration.
Diagnostics (objective, accurate gap, fixed-point residual, recovery error)
are computed with *fresh* products and are NOT counted in the cost.
"""
import numpy as np
import scipy.linalg as sla

DATA = '/workspace/data/complex_optimization_data.npy'
OUT = '/workspace/outputs'
IMG = '/workspace/report/images'


def load_data():
    d = np.load(DATA, allow_pickle=True).item()
    return d['A'], d['b'], d['x_true'], d['meta']


def soft(v, t):
    return np.sign(v) * np.maximum(np.abs(v) - t, 0.0)


class Problem:
    def __init__(self, A, b, lam=0.0, mu=0.0, smax=None):
        self.A, self.b, self.lam, self.mu = A, b, float(lam), float(mu)
        if smax is None:
            smax = np.linalg.norm(A, 2)
        self.Lf = smax ** 2            # Lipschitz const. of the LS part
        self.L = self.Lf + self.mu     # Lipschitz const. of grad f
        self.xs = None                 # reference minimiser
        self.mv = 0                    # counted matvecs

    # --- oracles -----------------------------------------------------
    def Ax(self, x):
        self.mv += 1
        return self.A @ x

    def grad_from_Ax(self, x, Ax):
        self.mv += 1
        return self.A.T @ (Ax - self.b) + self.mu * x

    def f_from_Ax(self, x, Ax):
        r = Ax - self.b
        return 0.5 * r @ r + 0.5 * self.mu * x @ x

    def g(self, x):
        return self.lam * np.abs(x).sum()

    def F_from_Ax(self, x, Ax):
        return self.f_from_Ax(x, Ax) + self.g(x)

    def prox(self, v, t):
        return soft(v, self.lam * t) if self.lam > 0 else v.copy()

    # --- uncounted diagnostics ----------------------------------------
    def F(self, x):
        return self.F_from_Ax(x, self.A @ x)

    def set_ref(self, xs):
        self.xs = xs.copy()
        self.Axs = self.A @ xs
        self.rs = self.Axs - self.b
        self.Fs = self.F_from_Ax(xs, self.Axs)

    def gap(self, x, Ax=None):
        """Accurate F(x)-F(x*) using differences (avoids cancellation)."""
        if Ax is None:
            Ax = self.A @ x
        d = Ax - self.Axs
        val = 0.5 * d @ (2 * self.rs + d)
        val += 0.5 * self.mu * (x - self.xs) @ (x + self.xs)
        val += self.lam * (np.abs(x) - np.abs(self.xs)).sum()
        return val

    def fpr(self, x, Ax=None):
        """Fixed-point (prox-gradient) residual ||x - prox(x - grad f(x)/L)||."""
        if Ax is None:
            Ax = self.A @ x
        gr = self.A.T @ (Ax - self.b) + self.mu * x
        return np.linalg.norm(x - self.prox(x - gr / self.L, 1.0 / self.L))

    def kkt(self, x):
        """Max violation of the Lasso/elastic-net optimality conditions."""
        gr = self.A.T @ (self.A @ x - self.b) + self.mu * x
        S = x != 0
        on = np.abs(gr[S] + self.lam * np.sign(x[S])).max() if S.any() else 0.0
        off = max(0.0, (np.abs(gr[~S]) - self.lam).max()) if (~S).any() else 0.0
        return on, off, -(np.abs(gr[~S]) - self.lam).max() if (~S).any() else np.inf

    def duality_gap(self, x):
        """Primal-dual gap for mu=0 Lasso (rigorous certificate of F*)."""
        r = self.A @ x - self.b
        if self.mu == 0:
            c = np.abs(self.A.T @ r).max()
            nu = r * min(1.0, self.lam / c) if self.lam > 0 else r
            cc = min(1.0, self.lam / c) if self.lam > 0 else 1.0
            # P - D = 1/2(1-c)^2||r||^2 + c x^T A^T r + lam||x||_1  (cancellation-free form)
            return 0.5 * (1 - cc) ** 2 * (r @ r) + cc * x @ (self.A.T @ r) + self.lam * np.abs(x).sum()
        return np.nan


# =====================================================================
#  Diagnostics recorder
# =====================================================================
class Recorder:
    def __init__(self, P, x_true=None, every=1, full=True):
        self.P, self.xt, self.every, self.full = P, x_true, every, full
        self.h = {k: [] for k in ['k', 'F', 'gap', 'fpr', 'relerr', 'smis', 'nnz', 'mv', 'dstar']}
        if P.xs is not None:
            self.S = P.xs != 0

    def __call__(self, k, x):
        if k % self.every:
            return
        P = self.P
        Ax = P.A @ x
        h = self.h
        h['k'].append(k)
        h['F'].append(P.F_from_Ax(x, Ax))
        h['mv'].append(P.mv)
        if P.xs is not None:
            h['gap'].append(P.gap(x, Ax))
            h['smis'].append(int(np.sum((x != 0) != self.S)))
            h['dstar'].append(np.linalg.norm(x - P.xs))
        if self.full:
            h['fpr'].append(P.fpr(x, Ax))
        h['nnz'].append(int(np.count_nonzero(x)))
        if self.xt is not None:
            h['relerr'].append(np.linalg.norm(x - self.xt) / np.linalg.norm(self.xt))

    def out(self):
        return {k: np.array(v) for k, v in self.h.items() if len(v)}


# =====================================================================
#  Algorithms (each costs 2 counted matvecs / iteration)
# =====================================================================
def pg(P, x0, K, rec=None):
    """Proximal gradient = forward-backward (explicit-implicit Euler) of gradient flow."""
    x = x0.copy(); Ax = P.Ax(x)
    if rec: rec(0, x)
    for k in range(1, K + 1):
        gr = P.grad_from_Ax(x, Ax)
        x = P.prox(x - gr / P.L, 1.0 / P.L)
        Ax = P.Ax(x)
        if rec: rec(k, x)
    return x


def fista(P, x0, K, rec=None, restart=None):
    """Beck-Teboulle FISTA; restart in {None,'grad','fun'} (O'Donoghue-Candes)."""
    x = x0.copy(); Ax = P.Ax(x); xp, Axp = x.copy(), Ax.copy()
    y, Ay = x.copy(), Ax.copy(); t = 1.0; Fx = P.F_from_Ax(x, Ax)
    nres = 0
    if rec: rec(0, x)
    for k in range(1, K + 1):
        gr = P.grad_from_Ax(y, Ay)
        xn = P.prox(y - gr / P.L, 1.0 / P.L); Axn = P.Ax(xn)
        tn = 0.5 * (1 + np.sqrt(1 + 4 * t * t))
        do_r = False
        if restart == 'grad':
            do_r = (y - xn) @ (xn - x) > 0
        elif restart == 'fun':
            Fn = P.F_from_Ax(xn, Axn); do_r = Fn > Fx; Fx = Fn
        xp, Axp, x, Ax = x, Ax, xn, Axn
        if do_r:
            t, tn = 1.0, 1.0; nres += 1
            y, Ay = x.copy(), Ax.copy()
        else:
            beta = (t - 1) / tn
            y = x + beta * (x - xp); Ay = Ax + beta * (Ax - Axp)
        t = tn
        if rec: rec(k, x)
    P.nrestart = nres
    return x


def nesterov_const(P, x0, K, mu, rec=None):
    """Constant-momentum Nesterov (V-FISTA): beta=(1-sqrt q)/(1+sqrt q), q=mu/L."""
    q = mu / P.L; beta = (1 - np.sqrt(q)) / (1 + np.sqrt(q))
    x = x0.copy(); Ax = P.Ax(x); y, Ay = x.copy(), Ax.copy()
    xs = [x.copy()]
    for k in range(1, K + 1):
        gr = P.grad_from_Ax(y, Ay)
        xn = P.prox(y - gr / P.L, 1.0 / P.L); Axn = P.Ax(xn)
        y = xn + beta * (xn - x); Ay = Axn + beta * (Axn - Ax)
        x, Ax = xn, Axn
        xs.append(x.copy())
        if rec: rec(k, x)
    return xs


def alpha_II(L, mu, gam):
    """Positive root of L a^2 = (1+a)(gam + a mu)."""
    a2, a1, a0 = L - mu, -(gam + mu), -gam
    return (-a1 + np.sqrt(a1 * a1 - 4 * a2 * a0)) / (2 * a2)


def alpha_I(L, gam):
    """Positive root of L a^2 = gam (1+a)."""
    return (gam + np.sqrt(gam * gam + 4 * L * gam)) / (2 * L)


def vos2(P, x0, K, rec=None, mu=None, gamma0=None, restart=None, monotone=False,
         lyap=False, keep=False):
    """VOS-AG-II: predictor / prox-gradient corrector discretisation of the AG flow
         x' = y - x,  gamma y' = mu (x - y) - dF(x),  gamma' = mu - gamma.
    restart='fun': if F increases, y <- x, gamma <- gamma0 (momentum reset).
    monotone=True: accept the corrector only if it decreases F (MFISTA-style);
                   the y-update is unchanged, so the Lyapunov proof still applies.
    mu: convexity parameter used by the algorithm (default P.mu).
    """
    mu = P.mu if mu is None else mu
    L = P.L
    g0 = (mu if mu > 0 else L) if gamma0 is None else gamma0
    x = x0.copy(); Ax = P.Ax(x); y, Ay = x.copy(), Ax.copy(); gam = g0
    Fx = P.F_from_Ax(x, Ax)
    if rec: rec(0, x)
    E, al, xs_list = [], [], []
    if lyap: E.append(P.gap(x) + 0.5 * gam * np.sum((y - P.xs) ** 2))
    if keep: xs_list.append(x.copy())
    nres = 0
    for k in range(1, K + 1):
        a = alpha_II(L, mu, gam)
        at = a * gam / (gam + a * mu)
        xh = (x + at * y) / (1 + at); Axh = (Ax + at * Ay) / (1 + at)
        gr = P.grad_from_Ax(xh, Axh)
        z = P.prox(xh - gr / L, 1.0 / L); Az = P.Ax(z)
        G = L * (xh - z); AG = L * (Axh - Az)
        gb = gam + a * mu
        y = (gam * y + a * mu * xh - a * G) / gb
        Ay = (gam * Ay + a * mu * Axh - a * AG) / gb
        gam = gb / (1 + a)
        Fz = P.F_from_Ax(z, Az)
        if monotone:
            if Fz <= Fx:
                x, Ax, Fx = z, Az, Fz
        else:
            if restart == 'fun' and Fz > Fx:
                y, Ay, gam = z.copy(), Az.copy(), g0; nres += 1
            x, Ax, Fx = z, Az, Fz
        if lyap:
            E.append(P.gap(x) + 0.5 * gam * np.sum((y - P.xs) ** 2)); al.append(a)
        if keep: xs_list.append(x.copy())
        if rec: rec(k, x)
    P.nrestart = nres
    if lyap: return x, np.array(E), np.array(al)
    if keep: return xs_list
    return x


def vos1(P, x0, K, rec=None, mu=None, gamma0=None, restart=None, lyap=False):
    """VOS-AG-I: semi-implicit splitting; y by a proximal (implicit in g) step,
    x by the convex-combination corrector x+ = (x + a y+)/(1+a);  L a^2 = gam (1+a)."""
    mu = P.mu if mu is None else mu
    L = P.L
    g0 = (mu if mu > 0 else L) if gamma0 is None else gamma0
    x = x0.copy(); Ax = P.Ax(x); y, Ay = x.copy(), Ax.copy(); gam = g0
    Fx = P.F_from_Ax(x, Ax)
    if rec: rec(0, x)
    E, al = [], []
    if lyap: E.append(P.gap(x) + 0.5 * gam * np.sum((y - P.xs) ** 2))
    for k in range(1, K + 1):
        a = alpha_I(L, gam)
        z = (x + a * y) / (1 + a); Az = (Ax + a * Ay) / (1 + a)
        gr = P.grad_from_Ax(z, Az)
        gb = gam + a * mu
        v = (gam * y + a * mu * z - a * gr) / gb
        y = P.prox(v, a / gb); Ay = P.Ax(y)
        x = (x + a * y) / (1 + a); Ax = (Ax + a * Ay) / (1 + a)
        gam = gb / (1 + a)
        Fn = P.F_from_Ax(x, Ax)
        if restart == 'fun' and Fn > Fx:
            y, Ay, gam = x.copy(), Ax.copy(), g0
        Fx = Fn
        if lyap:
            E.append(P.gap(x) + 0.5 * gam * np.sum((y - P.xs) ** 2)); al.append(a)
        if rec: rec(k, x)
    if lyap: return x, np.array(E), np.array(al)
    return x


# =====================================================================
#  ADMM for  min f(x) + g(z)  s.t. x = z   (scaled dual u)
# =====================================================================
class ADMMSolver:
    def __init__(self, P, AAt=None):
        self.P = P
        self.AAt = P.A @ P.A.T if AAt is None else AAt
        self.Atb = P.A.T @ P.b

    def factor(self, beta):
        m = self.AAt.shape[0]
        # (A^T A + (beta+mu) I)^{-1} via Woodbury with chol(A A^T + c I)
        self.c = beta + self.P.mu
        self.cf = sla.cho_factor(self.AAt + self.c * np.eye(m))
        self.beta = beta

    def xsolve(self, rhs):
        P = self.P
        w = P.Ax(rhs)
        P.mv += 1
        return (rhs - P.A.T @ sla.cho_solve(self.cf, w)) / self.c

    def run(self, beta, K, x0=None, rec=None, fast=False, eta=0.999, lyap=False, zs=None, us=None):
        P = self.P; self.factor(beta)
        n = P.A.shape[1]
        z = np.zeros(n) if x0 is None else x0.copy(); u = np.zeros(n)
        if rec: rec(0, z)
        V = []
        if lyap: V.append(beta * np.sum((u - us) ** 2) + beta * np.sum((z - zs) ** 2))
        zh, uh, zp, up, ak, cp = z.copy(), u.copy(), z.copy(), u.copy(), 1.0, np.inf
        for k in range(1, K + 1):
            if not fast:
                x = self.xsolve(self.Atb + beta * (z - u))
                z = P.prox(x + u, 1.0 / beta)
                u = u + x - z
            else:
                x = self.xsolve(self.Atb + beta * (zh - uh))
                z = P.prox(x + uh, 1.0 / beta)
                u = uh + x - z
                c = beta * np.sum((u - uh) ** 2) + beta * np.sum((z - zh) ** 2)
                if c < eta * cp:
                    an = 0.5 * (1 + np.sqrt(1 + 4 * ak * ak))
                    zh = z + (ak - 1) / an * (z - zp); uh = u + (ak - 1) / an * (u - up)
                    ak, cp = an, c
                else:
                    ak = 1.0; zh, uh = zp.copy(), up.copy(); cp = cp / eta
                    z, u = zp.copy(), up.copy()
                zp, up = z.copy(), u.copy()
            if lyap: V.append(beta * np.sum((u - us) ** 2) + beta * np.sum((z - zs) ** 2))
            if rec: rec(k, z)
        if lyap: return z, u, np.array(V)
        return z, u


# =====================================================================
#  Reference solution + certificate
# =====================================================================
def reference(P, K=4000, x0=None, polish=True, verbose=False):
    n = P.A.shape[1]
    x0 = np.zeros(n) if x0 is None else x0
    if P.mu > 0:
        x = vos2(P, x0, K, gamma0=P.mu)
    else:
        x = fista(P, x0, K, restart='grad')
    if polish and P.lam > 0:
        for it in range(20):
            S = x != 0
            s = np.sign(x[S]); AS = P.A[:, S]
            H = AS.T @ AS + P.mu * np.eye(S.sum())
            xS = np.linalg.solve(H, AS.T @ P.b - P.lam * s)
            xn = np.zeros(n); xn[S] = xS
            ok_sign = np.all(np.sign(xS) == s)
            gr = P.A.T @ (P.A @ xn - P.b) + P.mu * xn
            ok_off = np.all(np.abs(gr[~S]) <= P.lam * (1 + 1e-12))
            if ok_sign and ok_off:
                x = xn; break
            # otherwise take more prox-grad steps and retry
            x = pg(P, x, 200)
    elif polish and P.lam == 0:
        A = P.A
        # x* = A^T (A A^T + mu I)^{-1} b   (unique minimiser of ridge)
        x = A.T @ np.linalg.solve(A @ A.T + P.mu * np.eye(A.shape[0]), P.b)
    return x


def first_hit(k, v, tol):
    idx = np.nonzero(v <= tol)[0]
    return int(k[idx[0]]) if len(idx) else None
