"""Learned prior over auxiliary constructions: L2 logistic regression on
problem-agnostic hand features, trained only on synthetic data (no IMO labels).
"""
import glob
import json
import os

import numpy as np
from scipy.optimize import minimize

from geo.aux import FAMILIES, FEAT_NAMES


class LogReg:
    def __init__(self, l2=1.0, pos_weight=None):
        self.l2, self.pos_weight = l2, pos_weight

    def fit(self, X, y):
        self.mu = X.mean(0)
        self.sd = X.std(0) + 1e-6
        Xs = (X - self.mu) / self.sd
        Xs = np.hstack([Xs, np.ones((len(Xs), 1))])
        pw = self.pos_weight or max(1.0, (len(y) - y.sum()) / max(1, y.sum()))
        w_s = np.where(y > 0, pw, 1.0)
        n = len(y)

        def f(w):
            z = Xs @ w
            p = 1 / (1 + np.exp(-z))
            ll = w_s * (y * np.logaddexp(0, -z) + (1 - y) * np.logaddexp(0, z))
            g = Xs.T @ (w_s * (p - y))
            reg = self.l2 * np.sum(w[:-1] ** 2) / 2
            greg = np.concatenate([self.l2 * w[:-1], [0]])
            return ll.sum() / n + reg / n, g / n + greg / n
        r = minimize(f, np.zeros(Xs.shape[1]), jac=True, method='L-BFGS-B', options=dict(maxiter=500))
        self.w = r.x
        return self

    def score(self, X):
        Xs = (X - self.mu) / self.sd
        return np.hstack([Xs, np.ones((len(Xs), 1))]) @ self.w


def heuristic_from_features(F):
    """Re-create the hand-designed heuristic score from feature columns."""
    base = np.array([5, 4, 4.5, 4, 4.5, 3.5, 3, 3, 2.5, 2.5, 1.0, 1.0, 2, 1.5])
    fam = F[:, :len(FAMILIES)]
    nargs = F[:, FEAT_NAMES.index('n_distinct_args')] * 6
    ngoal = F[:, FEAT_NAMES.index('frac_goal_args')] * nargs
    return fam @ base + 0.8 * ngoal - 0.1 * nargs


def first_rank(scores, y, rng=None):
    """1-based rank of the first positive when sorting by decreasing score (ties broken randomly)."""
    if y.sum() == 0:
        return None
    rng = rng or np.random.RandomState(0)
    tie = rng.rand(len(scores)) * 1e-9
    order = np.argsort(-(scores + tie))
    pos = np.flatnonzero(y[order])
    return int(pos[0]) + 1


def expected_first_rank_random(n, s):
    """E[rank of first success] for a random permutation of n items with s successes."""
    return (n + 1) / (s + 1)


def p_solved_random(n, s, k):
    """P(first success within the first k) for random order."""
    if s == 0:
        return 0.0
    k = min(k, n)
    # 1 - C(n-s, k)/C(n, k)
    p_fail = 1.0
    for i in range(k):
        p_fail *= max(0.0, (n - s - i) / (n - i))
        if p_fail == 0:
            break
    return 1 - p_fail


def mrr_groups(scores, y, g):
    rr = []
    for gid in np.unique(g):
        m = g == gid
        if y[m].sum() == 0:
            continue
        r = first_rank(scores[m], y[m])
        rr.append(1.0 / r)
    return float(np.mean(rr)) if rr else float('nan'), len(rr)


def mrr_random(y, g):
    rr = []
    for gid in np.unique(g):
        m = g == gid
        n, s = int(m.sum()), int(y[m].sum())
        if s == 0:
            continue
        # E[1/rank] for random order: sum_k P(first=k)/k
        e, pf = 0.0, 1.0
        for k in range(1, n - s + 2):
            pk = pf * s / (n - k + 1)
            e += pk / k
            pf *= (n - k + 1 - s) / (n - k + 1)
            if pf <= 0:
                break
        rr.append(e)
    return float(np.mean(rr))


def load_synth(dirs):
    Fs, ys, gs, src = [], [], [], []
    off = 0
    for d, tag in dirs:
        for f in sorted(glob.glob(os.path.join(d, '*.npz'))):
            z = np.load(f)
            if len(z['y']) == 0:
                continue
            Fs.append(z['F']); ys.append(z['y'].astype(float))
            gs.append(z['g'] + off); src += [tag] * len(z['y'])
        off += 10 ** 7
    if not Fs:
        return None
    return np.vstack(Fs), np.concatenate(ys), np.concatenate(gs), np.array(src)
