"""Minimal numpy Gaussian-process library: MinMax (Tanimoto-for-counts) and RBF kernels,
type-II ML hyperparameter fitting with bounds, closed-form LOOCV."""
import numpy as np
from scipy.optimize import minimize


def minmax_kernel(A, B, chunk=256):
    """Generalised Tanimoto (MinMax) kernel on non-negative vectors: sum(min)/sum(max)."""
    A = np.asarray(A, np.float32); B = np.asarray(B, np.float32)
    cols = np.where((A.sum(0) > 0) | (B.sum(0) > 0))[0]
    A = A[:, cols]; B = B[:, cols]
    sA = A.sum(1); sB = B.sum(1)
    K = np.zeros((len(A), len(B)), np.float64)
    for i in range(0, len(A), chunk):
        a = A[i:i + chunk]
        mn = np.minimum(a[:, None, :], B[None, :, :]).sum(-1)
        mx = sA[i:i + chunk, None] + sB[None, :] - mn  # sum max = sum a + sum b - sum min
        K[i:i + chunk] = mn / np.maximum(mx, 1e-12)
    return K


def sqdist(A, B):
    A = np.atleast_2d(A); B = np.atleast_2d(B)
    return np.maximum((A ** 2).sum(1)[:, None] + (B ** 2).sum(1)[None] - 2 * A @ B.T, 0)


class CompositeGP:
    """y = m + f(x) + eps;  k = sum_j s_j * K_j;  K_j either precomputed (minmax) or RBF on feature block.
    noise_i = sn^2 + (alpha * std_i)^2 (heteroscedastic optional)."""

    def __init__(self, blocks, hetero=False, mean='const'):
        # blocks: list of ('pre', name) or ('rbf', name)
        self.blocks = blocks; self.hetero = hetero; self.mean = mean

    def _kernels(self, theta, Xtr, Xte=None):
        Ks = []; p = 0
        for kind, name in self.blocks:
            s = np.exp(theta[p]); p += 1
            if kind == 'pre':
                K = Xtr[name] if Xte is None else Xte[name]
                Ks.append(s * K)
            else:
                ell = np.exp(theta[p]); p += 1
                A = Xtr[name]; B = A if Xte is None else Xte[name + '_te']
                Ks.append(s * np.exp(-0.5 * sqdist(B / ell, A / ell) if Xte is not None else -0.5 * sqdist(A / ell, A / ell)))
        return sum(Ks), p

    def _noise(self, theta, p, std):
        sn2 = np.exp(theta[p]); p += 1
        if self.hetero:
            a = np.exp(theta[p]); p += 1
            return sn2 + (a * std) ** 2, p
        return sn2 + 0 * std, p

    def nlml(self, theta, X, y, std):
        K, p = self._kernels(theta, X)
        nv, p = self._noise(theta, p, std)
        Ky = K + np.diag(nv)
        n = len(y)
        try:
            L = np.linalg.cholesky(Ky + 1e-8 * np.eye(n))
        except np.linalg.LinAlgError:
            return 1e10
        # const mean via GLS (profile out)
        one = np.ones(n)
        Li1 = np.linalg.solve(L, one); Liy = np.linalg.solve(L, y)
        m = (Li1 @ Liy) / (Li1 @ Li1) if self.mean == 'const' else 0.0
        r = Liy - m * Li1
        return 0.5 * r @ r + np.log(np.diag(L)).sum() + 0.5 * n * np.log(2 * np.pi)

    def fit(self, X, y, std, theta0, bounds, restarts=3, seed=0):
        rng = np.random.default_rng(seed)
        best = None
        for r in range(restarts):
            t0 = np.array(theta0) if r == 0 else np.array([rng.uniform(lo, hi) for lo, hi in bounds])
            res = minimize(self.nlml, t0, args=(X, y, std), method='L-BFGS-B', bounds=bounds)
            if best is None or res.fun < best.fun:
                best = res
        self.theta = best.x; self.X = X; self.y = y; self.std = std
        self._precompute()
        return self

    def _precompute(self):
        K, p = self._kernels(self.theta, self.X)
        nv, _ = self._noise(self.theta, p, self.std)
        n = len(self.y)
        Ky = K + np.diag(nv) + 1e-8 * np.eye(n)
        self.Kinv = np.linalg.inv(Ky)
        one = np.ones(n)
        self.m = (one @ self.Kinv @ self.y) / (one @ self.Kinv @ one) if self.mean == 'const' else 0.0
        self.alpha = self.Kinv @ (self.y - self.m)
        self.noise_tr = nv

    def loo(self):
        """Closed-form LOO predictive mean/var of latent+noise (Rasmussen & Williams eq. 5.12),
        with the mean held fixed at the full-data GLS value."""
        d = np.diag(self.Kinv)
        mu = self.y - self.alpha / d
        var = 1.0 / d
        return mu, var

    def predict(self, Xte, std_te=None, kdiag=None):
        """Xte must hold cross-kernel matrices for 'pre' blocks (n_te x n_tr) and name+'_te' features for rbf blocks."""
        Ks, p = self._kernels(self.theta, self.X, Xte)
        mu = self.m + Ks @ self.alpha
        # prior diag: sum of s_j (minmax self-similarity = 1, rbf = 1)
        pp = 0; prior = 0.0
        for kind, _ in self.blocks:
            prior += np.exp(self.theta[pp]); pp += 1 if kind == 'pre' else 2
        var_f = prior - np.einsum('ij,jk,ik->i', Ks, self.Kinv, Ks)
        var_f = np.maximum(var_f, 1e-9)
        nv = None
        if std_te is not None:
            nv, _ = self._noise(self.theta, pp, std_te)
        return mu, var_f, nv
