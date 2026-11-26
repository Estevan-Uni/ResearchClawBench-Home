"""Minimal nanopore signal simulator and aligners (move table, basecaller-guided banded DTW, full DTW).
All quantities in z-scaled pore-model units (the provided models have mean 0 / SD 1)."""
import numpy as np
from common import *

STRIDE = 5


class Model:
    def __init__(self, name):
        df, K, X = load_model(name)
        self.K = K
        self.mu = df.current_mean.values.astype(float)
        self.sd = df.current_std.values.astype(float)
        self.dw = df.dwell_time.values.astype(float)
        self.X = X


def random_seq(rng, n):
    return rng.integers(0, 4, n)


def kmer_codes(seq, K):
    n = len(seq) - K + 1
    c = np.zeros(n, dtype=np.int64)
    for p in range(K):
        c = c * 4 + seq[p:p + n]
    return c


def simulate_read(rng, model, codes, noise_infl=2.0, scale=None, shift=None, mod_shift=None, dwell_scale=1.0):
    """Gaussian samples per k-mer, geometric dwell, read-level scale/shift.
    mod_shift: optional per-k-mer additive shift (model units) applied to the true signal."""
    L = len(codes)
    mu = model.mu[codes].copy()
    if mod_shift is not None:
        mu = mu + mod_shift
    sd = model.sd[codes] * noise_infl
    d = rng.geometric(1.0 / np.maximum(model.dw[codes] * dwell_scale, 1.0))
    bounds = np.r_[0, np.cumsum(d)]                     # bounds[i] = first sample of k-mer i
    T = bounds[-1]
    lab = np.repeat(np.arange(L), d)
    z = mu[lab] + rng.normal(0, 1, T) * sd[lab]
    scale = rng.normal(10, 1) if scale is None else scale
    shift = rng.normal(90, 5) if shift is None else shift
    raw = z * scale + shift
    return dict(raw=raw, lab=lab, bounds=bounds, T=T, L=L, scale=scale, shift=shift)


def move_table(rng, read, jitter_strides=1.0, err_rate=0.05):
    """Basecaller-like boundaries: true boundaries + Gaussian jitter (+ occasional larger errors),
    quantised to a 5-sample stride, made monotone."""
    b = read["bounds"][1:-1].astype(float)
    j = rng.normal(0, jitter_strides * STRIDE, len(b))
    big = rng.random(len(b)) < err_rate
    j[big] += rng.choice([-1, 1], big.sum()) * rng.integers(2, 5, big.sum()) * STRIDE
    q = np.round((b + j) / STRIDE) * STRIDE
    q = np.clip(q, 0, read["T"])
    q = np.maximum.accumulate(q).astype(int)
    bounds = np.r_[0, q, read["T"]]
    return bounds


def labels_from_bounds(bounds, T):
    lab = np.zeros(T, dtype=int)
    # number of interior boundaries <= t
    inner = bounds[1:-1]
    lab = np.searchsorted(inner, np.arange(T), side="right")
    return lab


def norm_from_assignment(raw, lab, mu_ref):
    """Least-squares scale/shift: raw ~ scale*mu + shift using a sample->kmer assignment."""
    m = mu_ref[lab]
    A = np.c_[m, np.ones_like(m)]
    (s, b), *_ = np.linalg.lstsq(A, raw, rcond=None)
    return s, b


def dtw(z, mu, sd, band_center=None, W=None):
    """Sample-level segmental DTW: every sample is assigned to one k-mer, path is monotone,
    each k-mer gets >=1 sample. Optional band |i - band_center[t]| <= W. Returns labels, cells."""
    T, L = len(z), len(mu)
    inv = 1.0 / (2 * sd ** 2); lsd = np.log(sd)
    D = np.full(L, np.inf)
    back = np.zeros((T, L), dtype=bool)  # True = came from i-1 (advance)
    idx = np.arange(L)
    cells = 0
    if W is not None:
        lo_all = np.clip(band_center - W, 0, L - 1); hi_all = np.clip(band_center + W, 0, L - 1)
    for t in range(T):
        c = (z[t] - mu) ** 2 * inv + lsd
        if W is not None:
            lo, hi = lo_all[t], hi_all[t]
            if t == 0: lo = 0
            if t == T - 1: hi = L - 1
            mask = (idx < lo) | (idx > hi)
            c = np.where(mask, np.inf, c)
            cells += hi - lo + 1
        else:
            cells += L
        if t == 0:
            Dn = np.full(L, np.inf); Dn[0] = c[0]
        else:
            adv = np.r_[np.inf, D[:-1]]
            take = adv < D
            back[t] = take
            Dn = c + np.where(take, adv, D)
        D = Dn
    if not np.isfinite(D[L - 1]):
        return None, cells
    lab = np.zeros(T, dtype=int); i = L - 1
    for t in range(T - 1, -1, -1):
        lab[t] = i
        if t > 0 and back[t, i]:
            i -= 1
    return lab, cells


def bcdtw(raw, mu, sd, mv_bounds, W, n_pass=2):
    """Basecaller-guided banded DTW (Uncalled4-style): band around move-table k-mer index,
    normalisation initialised from the move table and re-estimated after each pass."""
    T = len(raw)
    mv_lab = labels_from_bounds(mv_bounds, T)
    s, b = norm_from_assignment(raw, mv_lab, mu)
    cells = 0; lab = mv_lab
    for _ in range(n_pass):
        z = (raw - b) / s
        w = W
        while True:
            new, c = dtw(z, mu, sd, band_center=mv_lab, W=w)
            cells += c
            if new is not None: break
            w *= 2   # infeasible band (rare): widen
        lab = new
        s, b = norm_from_assignment(raw, lab, mu)
    return lab, (s, b), cells


def full_dtw(raw, mu, sd, init_lab, n_pass=2):
    s, b = norm_from_assignment(raw, init_lab, mu)
    cells = 0
    for _ in range(n_pass):
        z = (raw - b) / s
        lab, c = dtw(z, mu, sd)
        cells += c
        s, b = norm_from_assignment(raw, lab, mu)
    return lab, (s, b), cells


def event_means(z, lab, L):
    s = np.bincount(lab, weights=z, minlength=L); n = np.bincount(lab, minlength=L)
    m = np.full(L, np.nan); m[n > 0] = s[n > 0] / n[n > 0]
    return m, n


def bounds_from_labels(lab, L):
    # first sample of each kmer (NaN if kmer has 0 samples)
    first = np.full(L, -1)
    ch = np.r_[0, np.flatnonzero(np.diff(lab)) + 1]
    first[lab[ch]] = ch
    return first
