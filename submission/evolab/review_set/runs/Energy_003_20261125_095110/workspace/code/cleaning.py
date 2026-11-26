"""Fault injection, two-stage cleaning (detect -> reconstruct), and evaluation utilities.

This is an independent, transparent implementation of a two-stage clean (anomaly detection
followed by reconstruction). It is NOT a replication of the HEEW paper's own algorithm,
whose details are not available in this workspace.
"""
import numpy as np, pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from scipy.optimize import linear_sum_assignment

import os
# For environment-independent reproducibility we use our own numpy K-means (k-means++ init, Lloyd,
# n_init restarts) by default; set HEEW_USE_SKLEARN=1 to use scikit-learn instead.
HAVE_SK = False
if os.environ.get("HEEW_USE_SKLEARN") == "1":
    from sklearn.cluster import KMeans as _SKKMeans
    HAVE_SK = True

# ------------------------------------------------------------------ corruption
def corrupt(y, seed=0, miss_rate=0.05, out_rate=0.02, iso_frac=0.4, block_len=(3, 24)):
    """Return corrupted series and boolean masks (missing, outlier) + outlier type array."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y, float); n = len(y); yc = y.copy()
    miss = np.zeros(n, bool); out = np.zeros(n, bool); otype = np.array([""] * n, dtype=object)
    sigma = np.std(y)
    # --- missing: blocks then isolated points
    n_miss = int(round(miss_rate * n)); n_block_target = int(round((1 - iso_frac) * n_miss))
    while miss.sum() < n_block_target:
        L = int(rng.integers(block_len[0], block_len[1] + 1)); s = int(rng.integers(1, n - L - 1))
        if miss[max(0, s - 2):s + L + 2].any(): continue
        miss[s:s + L] = True
    while miss.sum() < n_miss:
        s = int(rng.integers(1, n - 1))
        if miss[s - 1:s + 2].any(): continue
        miss[s] = True
    # --- outliers: count in hours; types spike 60 %, zero 15 %, scale block 15 %, stuck block 10 %
    n_out = int(round(out_rate * n))
    quota = {"spike": int(0.60 * n_out), "zero": int(0.15 * n_out), "scale": int(0.15 * n_out)}
    quota["stuck"] = n_out - sum(quota.values())
    free = lambda a, b: not (miss[max(0, a - 1):b + 1].any() or out[max(0, a - 1):b + 1].any())
    for typ, q in quota.items():
        cnt = 0
        while cnt < q:
            if typ in ("spike", "zero"):
                s = int(rng.integers(1, n - 1)); L = 1
            else:
                L = int(min(rng.integers(3, 7), q - cnt)) if q - cnt >= 3 else q - cnt
                s = int(rng.integers(1, n - L - 1))
            if not free(s, s + L): continue
            if typ == "spike":
                yc[s] = y[s] + rng.choice([-1, 1]) * rng.uniform(3, 6) * sigma
            elif typ == "zero":
                yc[s] = 0.0
            elif typ == "scale":
                yc[s:s + L] = y[s:s + L] * rng.choice([0.4, 0.5, 1.8, 2.2])
            elif typ == "stuck":
                yc[s:s + L] = y[s - 1]
            out[s:s + L] = True; otype[s:s + L] = typ; cnt += L
    yc = np.where(yc < 0, 0.0, yc)  # meters cannot read negative; clip negative spikes to 0
    yc[miss] = np.nan
    return yc, miss, out, otype

# ------------------------------------------------------------------ stage 1: detection
def seasonal_reference(y, mode="daily", half=7):
    """Robust reference: median of the same clock hour over +/-`half` days ('daily'),
    or of the same hour-of-week over +/-`half` weeks ('weekly'). NaNs ignored. Centre excluded."""
    Y = np.asarray(y, float).reshape(-1, 24)                       # days x 24
    if mode == "weekly":
        step = 7
    else:
        step = 1
    nd = Y.shape[0]; ref = np.full_like(Y, np.nan)
    for d in range(nd):
        idx = [d + k * step for k in range(-half, half + 1) if k != 0 and 0 <= d + k * step < nd]
        with np.errstate(all="ignore"):
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ref[d] = np.nanmedian(Y[idx], axis=0)
    return ref.ravel()

def flat_runs(y, min_len=4, tol=1e-9):
    y = np.asarray(y, float); n = len(y); flag = np.zeros(n, bool); i = 0
    while i < n:
        j = i
        while j + 1 < n and not np.isnan(y[i]) and abs(y[j + 1] - y[i]) <= tol: j += 1
        if j - i + 1 >= min_len: flag[i + 1:j + 1] = True  # first value of a stuck run is genuine
        i = j + 1
    return flag

def detect(yc, k=4.0, mode="daily", half=7, n_iter=2, rules=True, allow_zero=False):
    """Stage 1: robust seasonal-residual (Hampel-type) test + physical rules. Returns flag mask."""
    y = np.asarray(yc, float).copy(); obs = ~np.isnan(y); flag = np.zeros(len(y), bool)
    if rules:
        flag |= obs & (y < 0)
        if not allow_zero: flag |= obs & (y == 0)
        flag |= flat_runs(y, 4)
    for _ in range(n_iter):
        work = y.copy(); work[flag] = np.nan
        ref = seasonal_reference(work, mode, half)
        e = work - ref
        med = np.nanmedian(e); mad = 1.4826 * np.nanmedian(np.abs(e - med))
        flag |= obs & (np.abs(e - med) > k * mad)
    return flag

# ------------------------------------------------------------------ stage 2: reconstruction
def runs(mask):
    """List of (start, end_exclusive) for contiguous True runs."""
    m = np.r_[False, mask, False].astype(int); d = np.diff(m)
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))

def linear_fill(y):
    return pd.Series(y).interpolate(limit_direction="both").values

def _profile(y, mode="weekly"):
    """Reference profile for long gaps: mean of t-168 and t+168 (if observed), else same-hour
    median over +/-7 days, else hour-of-week mean."""
    n = len(y); p = np.full(n, np.nan)
    back = np.r_[np.full(168, np.nan), y[:-168]]; fwd = np.r_[y[168:], np.full(168, np.nan)]
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        p = np.nanmean(np.c_[back, fwd], axis=1)
        back2 = np.r_[np.full(336, np.nan), y[:-336]]; fwd2 = np.r_[y[336:], np.full(336, np.nan)]
        p2 = np.nanmean(np.c_[back2, fwd2], axis=1)
    p = np.where(np.isnan(p), p2, p)
    m = seasonal_reference(y, "daily", 7); p = np.where(np.isnan(p), m, p)
    how = pd.Series(y).groupby(np.arange(n) % 168).transform("mean").values
    return np.where(np.isnan(p), how, p)

def anchored_fill(y, prof, short=2):
    """Fill NaN runs: linear for runs <= `short`; otherwise profile + offset linearly blended between
    the discrepancies (observed - profile) at the two gap edges."""
    y = np.asarray(y, float).copy(); out = y.copy(); lin = linear_fill(y); n = len(y)
    for a, b in runs(np.isnan(y)):
        if b - a <= short or np.isnan(prof[a:b]).any():
            out[a:b] = lin[a:b]; continue
        dl = y[a - 1] - prof[a - 1] if a > 0 and not np.isnan(prof[a - 1]) else np.nan
        dr = y[b] - prof[b] if b < n and not np.isnan(prof[b]) else np.nan
        if np.isnan(dl) and np.isnan(dr): dl = dr = 0.0
        dl = dr if np.isnan(dl) else dl; dr = dl if np.isnan(dr) else dr
        w = (np.arange(a, b) - (a - 1)) / (b - (a - 1))
        out[a:b] = prof[a:b] + (1 - w) * dl + w * dr
    return out

def profile_fill(y, short=2):
    return anchored_fill(y, _profile(np.asarray(y, float)), short)

def cluster_fill(y, k=3, seed=0, short=2):
    """Cluster-aware reconstruction: K-means on complete days, assign incomplete days to the nearest
    centroid on their observed hours, fill with the (edge-anchored) centroid profile."""
    y = np.asarray(y, float); Y = y.reshape(-1, 24); full = ~np.isnan(Y).any(1)
    cent, _ = kmeans(Y[full], k, seed)
    prof = np.full_like(Y, np.nan)
    for d in range(Y.shape[0]):
        o = ~np.isnan(Y[d])
        if o.sum() == 0:
            # whole day missing -> use neighbour-day cluster (previous complete day, same weekday a week earlier)
            ref = Y[d - 7] if d >= 7 and not np.isnan(Y[d - 7]).any() else np.nanmean(Y, 0)
            c = np.argmin(((cent - ref) ** 2).sum(1))
        else:
            c = np.argmin(((cent[:, o] - Y[d, o]) ** 2).mean(1))
        prof[d] = cent[c]
    return anchored_fill(y, prof.ravel(), short)

def ridge_fill(y, X, alpha=1.0, short=0):
    """Cross-sectional reconstruction: ridge regression of y on sibling series X (n x p) + hour and
    weekday dummies, trained on observed hours."""
    y = np.asarray(y, float); n = len(y); h = np.arange(n) % 24; dw = (np.arange(n) // 24 + 2) % 7  # 2014-01-01 = Wed
    F = np.c_[X, np.eye(24)[h][:, 1:], np.eye(7)[dw][:, 1:]]
    mu, sd = F.mean(0), F.std(0) + 1e-12; Fz = (F - mu) / sd
    o = ~np.isnan(y); ym = y[o].mean()
    A = Fz[o].T @ Fz[o] + alpha * np.eye(F.shape[1]); beta = np.linalg.solve(A, Fz[o].T @ (y[o] - ym))
    pred = Fz @ beta + ym; out = y.copy(); out[~o] = pred[~o]
    return out

def hierarchy_fill(y, parent, siblings_sum):
    out = np.asarray(y, float).copy(); m = np.isnan(out); out[m] = (parent - siblings_sum)[m]; return out

# ------------------------------------------------------------------ clustering
def kmeans(X, k=3, seed=0, n_init=20, max_iter=300):
    """K-means (sklearn if available, otherwise numpy k-means++/Lloyd with identical settings)."""
    X = np.asarray(X, float)
    if HAVE_SK:
        km = _SKKMeans(n_clusters=k, n_init=n_init, random_state=seed).fit(X)
        return km.cluster_centers_, km.labels_
    rng = np.random.default_rng(seed); best = (np.inf, None, None)
    for _ in range(n_init):
        C = [X[rng.integers(len(X))]]
        for _j in range(1, k):
            d2 = np.min(((X[:, None] - np.array(C)[None]) ** 2).sum(-1), 1)
            C.append(X[rng.choice(len(X), p=d2 / d2.sum())])
        C = np.array(C)
        for _it in range(max_iter):
            lab = np.argmin(((X[:, None] - C[None]) ** 2).sum(-1), 1)
            Cn = np.array([X[lab == j].mean(0) if (lab == j).any() else C[j] for j in range(k)])
            if np.allclose(Cn, C): break
            C = Cn
        inertia = ((X - C[lab]) ** 2).sum()
        if inertia < best[0]: best = (inertia, C, lab)
    return best[1], best[2]

def ari(a, b):
    a = np.asarray(a); b = np.asarray(b)
    ua, ia = np.unique(a, return_inverse=True); ub, ib = np.unique(b, return_inverse=True)
    M = np.zeros((len(ua), len(ub))); np.add.at(M, (ia, ib), 1)
    c2 = lambda x: x * (x - 1) / 2
    s = c2(M).sum(); sa = c2(M.sum(1)).sum(); sb = c2(M.sum(0)).sum(); e = sa * sb / c2(len(a))
    return (s - e) / (0.5 * (sa + sb) - e)

def match_centroids(C_true, C_other):
    cost = ((C_true[:, None] - C_other[None]) ** 2).mean(-1)
    r, c = linear_sum_assignment(cost); perm = np.empty(len(r), int); perm[r] = c
    return perm  # C_other[perm[i]] matches C_true[i]

def cluster_compare(Yt, Yo, k=3, seed=0):
    Ct, lt = kmeans(Yt, k, seed); Co, lo = kmeans(Yo, k, seed); perm = match_centroids(Ct, Co)
    Cm = Co[perm]; inv = np.argsort(perm); lom = inv[lo]
    return dict(C_true=Ct, C_other=Cm, l_true=lt, l_other=lom,
                cent_RMSE=float(np.sqrt(((Ct - Cm) ** 2).mean())), cent_MAE=float(np.abs(Ct - Cm).mean()),
                ARI=float(ari(lt, lom)), agree=float((lt == lom).mean()))

# ------------------------------------------------------------------ metrics
def err(y, yhat, mask=None):
    y = np.asarray(y); yhat = np.asarray(yhat)
    if mask is not None: y, yhat = y[mask], yhat[mask]
    e = yhat - y
    return dict(MAE=float(np.abs(e).mean()), RMSE=float(np.sqrt((e ** 2).mean())),
                MAPE=float((np.abs(e) / np.abs(y)).mean() * 100))

def prf(flag, truth):
    tp = int((flag & truth).sum()); fp = int((flag & ~truth).sum()); fn = int((~flag & truth).sum())
    p = tp / (tp + fp) if tp + fp else 0.0; r = tp / (tp + fn) if tp + fn else 0.0
    return dict(TP=tp, FP=fp, FN=fn, precision=p, recall=r, F1=2 * p * r / (p + r) if p + r else 0.0)
