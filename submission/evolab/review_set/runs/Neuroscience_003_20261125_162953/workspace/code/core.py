"""Core library: DELVE-style feature selection, baselines, PHATE/DPT re-implementations, metrics.
numpy/scipy only; sklearn used opportunistically (try-import) with fallbacks.
Selection functions NEVER receive labels (age/phase/batch/true DE)."""
import numpy as np, h5py
from scipy import sparse
from scipy.spatial import cKDTree
from scipy.sparse.linalg import eigsh
from scipy.cluster.vq import kmeans2
from scipy.stats import spearmanr, rankdata
from scipy.spatial.distance import pdist, squareform, cdist

try:
    from sklearn.linear_model import lars_path
    HAVE_SK = True
except Exception:
    HAVE_SK = False

# ----------------------------------------------------------------------------- data
def load_rpe(path='/workspace/data/adata_RPE.h5ad'):
    f = h5py.File(path, 'r')
    X = f['X'][:].astype(np.float64)
    cat = lambda k: f['obs/__categories/' + k][:].astype(str)[f['obs/' + k][:]]
    obs = dict(phase=cat('phase'), state=cat('state'), batch=cat('batch'),
               age=f['obs/annotated_age'][:].astype(float))
    var = f['var/_index'][:].astype(str)
    f.close()
    assert X.shape == (2759, 241)
    assert set(obs['phase']) == {'G0', 'G1', 'S', 'G2'}
    assert 'nan' in set(obs['state'])
    assert obs['age'].min() >= 0 and abs(obs['age'].max() - 25.0667) < 0.01
    return X, obs, var

# ----------------------------------------------------------------------------- basics
def zscore(X):
    s = X.std(0); s[s == 0] = 1
    return (X - X.mean(0)) / s

def pca(X, n=20, whiten=False):
    Xc = X - X.mean(0)
    n = min(n, min(Xc.shape) - 1)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    Z = U[:, :n] * (1.0 if whiten else S[:n])
    return Z, Vt[:n], S[:n] ** 2 / (len(X) - 1)

def knn(Z, k):
    d, i = cKDTree(Z).query(Z, k=k + 1)
    return d[:, 1:], i[:, 1:]

def knn_affinity(Z, k=10, heat=True):
    """Symmetric kNN graph with heat kernel (adaptive bandwidth = dist to k-th nn)."""
    d, idx = knn(Z, k)
    n = len(Z)
    sig = d[:, -1] + 1e-12
    if heat:
        w = np.exp(-d ** 2 / (sig[:, None] * sig[idx]))
    else:
        w = np.ones_like(d)
    W = sparse.csr_matrix((w.ravel(), (np.repeat(np.arange(n), k), idx.ravel())), shape=(n, n))
    W = W.maximum(W.T)
    return W

def graph_from_features(X, k=10, npc=20, whiten=False):
    Xz = zscore(X)
    Z = pca(Xz, npc, whiten)[0] if Xz.shape[1] > npc else Xz
    return knn_affinity(Z, k), Z

# ----------------------------------------------------------------------------- graph scores
def laplacian_score(X, W):
    """He et al. 2005. Lower = smoother on the graph."""
    d = np.asarray(W.sum(1)).ravel()
    Dsum = d.sum()
    Xt = X - (d @ X) / Dsum
    num = (Xt * (d[:, None] * Xt)).sum(0) - (Xt * (W @ Xt)).sum(0)  # f'Lf
    den = (Xt * (d[:, None] * Xt)).sum(0)
    return num / np.maximum(den, 1e-12)

def morans_i(X, W):
    Xc = X - X.mean(0)
    n = len(X); S0 = W.sum()
    return (n / S0) * (Xc * (W @ Xc)).sum(0) / np.maximum((Xc ** 2).sum(0), 1e-12)

def neighbourhood_variance(X, idx):
    """mean squared difference to kNN neighbours / (2 * global variance); low = smooth."""
    v = X.var(0); v[v == 0] = 1
    diff = ((X[:, None, :] - X[idx]) ** 2).mean((0, 1))
    return diff / (2 * v)

# ----------------------------------------------------------------------------- kernel herding
def kernel_herding(Z, m, D=1000, seed=0):
    rng = np.random.default_rng(seed)
    sub = rng.choice(len(Z), min(len(Z), 1000), replace=False)
    sigma = np.median(pdist(Z[sub]))
    Wf = rng.normal(0, 1.0 / sigma, (Z.shape[1], D)); b = rng.uniform(0, 2 * np.pi, D)
    Phi = np.sqrt(2.0 / D) * np.cos(Z @ Wf + b)
    mu = Phi.mean(0); w = mu.copy(); chosen = []
    mask = np.zeros(len(Z), bool)
    for _ in range(m):
        s = Phi @ w; s[mask] = -np.inf
        j = int(np.argmax(s)); chosen.append(j); mask[j] = True
        w = w + mu - Phi[j]
    return np.array(chosen)

# ----------------------------------------------------------------------------- DELVE-style
DEFAULT_CFG = dict(npc=20, k=10, n_anchors=500, n_rff=1000, n_perm=500, p_thresh=0.05,
                   n_seeds=5, consistency=0.6, seed_graph_k=10, noise_filter=False,
                   dedup=False, whiten=False, noise_q=99, dedup_r=0.9)

def dedup_medoids(Xz, r=0.9):
    C = np.abs(np.corrcoef(Xz.T)); C = np.nan_to_num(C)
    from scipy.cluster.hierarchy import linkage, fcluster
    Dm = 1 - C; np.fill_diagonal(Dm, 0); Dm = np.clip((Dm + Dm.T) / 2, 0, None)
    L = linkage(squareform(Dm, checks=False), 'average')
    lab = fcluster(L, 1 - r, 'distance')
    med = []
    for g in np.unique(lab):
        ii = np.where(lab == g)[0]
        med.append(ii[np.argmax(C[np.ix_(ii, ii)].sum(1))])
    return np.sort(np.array(med)), lab

def _dyn_scores(Xz, nbr_sets):
    M = Xz[nbr_sets].mean(1)  # anchors x features
    v = Xz.var(0); v[v == 0] = 1
    return M, M.var(0) / v

def delve(X, cfg=None, verbose=False):
    """Two-step unsupervised DELVE-style selection. Returns dict with full ranking (best first)."""
    c = dict(DEFAULT_CFG); c.update(cfg or {})
    rng = np.random.default_rng(0)
    Xz = zscore(X); n, p = Xz.shape
    feat = np.arange(p)
    if c['dedup']:
        feat, _ = dedup_medoids(Xz, c['dedup_r'])
    info = {}
    # --- step 1: graph, sketch, neighbourhood means, dynamic score
    def step1(fs, seed):
        Z = pca(Xz[:, fs], c['npc'], c['whiten'])[0] if len(fs) > c['npc'] else Xz[:, fs]
        _, idx = knn(Z, c['k'])
        anc = kernel_herding(Z, min(c['n_anchors'], n), c['n_rff'], seed)
        nb = np.hstack([anc[:, None], idx[anc]])
        M, dyn = _dyn_scores(Xz, nb)
        return M, dyn, nb
    fs = feat.copy()
    if c['noise_filter']:
        for it in range(3):
            M, dyn, nb = step1(fs, 0)
            null = []
            for r in range(5):
                Xp = np.column_stack([rng.permutation(Xz[:, j]) for j in range(p)])
                null.append(_dyn_scores(Xp, nb)[1])
            thr = np.percentile(np.concatenate(null), c['noise_q'])
            keep = fs[dyn[fs] > thr]
            if len(keep) == len(fs) or len(keep) < 5:
                fs = keep if len(keep) >= 5 else fs; break
            fs = keep
        info['noise_kept'] = len(fs)
    sel_counts = np.zeros(p)
    for s in range(c['n_seeds']):
        M, dyn, nb = step1(fs, s)
        P = M[:, fs]; P = (P - P.mean(0)) / (P.std(0) + 1e-12)
        nmod = max(2, int(round(np.sqrt(len(fs) / 2))))
        np.random.seed(s)
        _, lab = kmeans2(P.T, nmod, minit='++', seed=s)
        Cm = np.corrcoef(P.T)
        r2 = np.random.default_rng(100 + s)
        for m in np.unique(lab):
            mem = np.where(lab == m)[0]
            if len(mem) < 2: continue
            sd = dyn[fs][mem].mean()
            coh = Cm[np.ix_(mem, mem)][np.triu_indices(len(mem), 1)].mean()
            nd = np.empty(c['n_perm']); nc = np.empty(c['n_perm'])
            for t in range(c['n_perm']):
                rs = r2.choice(len(fs), len(mem), replace=False)
                nd[t] = dyn[fs][rs].mean()
                nc[t] = Cm[np.ix_(rs, rs)][np.triu_indices(len(rs), 1)].mean()
            pd_ = (1 + (nd >= sd).sum()) / (1 + c['n_perm'])
            pc_ = (1 + (nc >= coh).sum()) / (1 + c['n_perm'])
            if pd_ < c['p_thresh'] and pc_ < c['p_thresh']:
                sel_counts[fs[mem]] += 1
    seeds = np.where(sel_counts >= c['consistency'] * c['n_seeds'])[0]
    if len(seeds) < 3:  # fallback: top dynamic features
        seeds = fs[np.argsort(-dyn[fs])[:max(3, len(fs) // 10)]]
        info['seed_fallback'] = True
    # --- step 2: seed graph, Laplacian score for ALL features
    Zs = zscore(X[:, seeds])
    if Zs.shape[1] > c['npc']:
        Zs = pca(Zs, c['npc'])[0]
    Ws = knn_affinity(Zs, c['seed_graph_k'])
    ls = laplacian_score(Xz, Ws)
    ls = np.where(Xz.std(0) == 0, np.inf, ls)
    info.update(seeds=seeds, dyn=dyn, lap=ls, sel_counts=sel_counts)
    return dict(ranking=np.argsort(ls), score=-ls, info=info)

# ----------------------------------------------------------------------------- baselines (all label-free except oracle)
def mcfs(X, W, n_eig=5, n_nonzero=30):
    """Cai et al. 2010: spectral embedding + L1 (LARS-lasso) regression to n_nonzero, score=max|coef|."""
    Xz = zscore(X)
    d = np.asarray(W.sum(1)).ravel(); Dm = sparse.diags(1 / np.sqrt(d))
    A = Dm @ W @ Dm
    vals, vecs = eigsh(A, k=n_eig + 1, which='LA')
    o = np.argsort(-vals); vecs = (Dm @ vecs[:, o])[:, 1:]
    vecs = (vecs - vecs.mean(0)) / vecs.std(0)
    p = X.shape[1]; coef = np.zeros((n_eig, p))
    for e in range(n_eig):
        y = vecs[:, e]
        if HAVE_SK:
            _, _, cp = lars_path(Xz, y, method='lasso', max_iter=min(n_nonzero, p))
            coef[e] = cp[:, -1]
        else:  # OMP fallback
            act = []; r = y.copy()
            for _ in range(min(n_nonzero, p)):
                j = int(np.argmax(np.abs(Xz.T @ r))); act.append(j)
                b = np.linalg.lstsq(Xz[:, act], y, rcond=None)[0]; r = y - Xz[:, act] @ b
            coef[e, act] = b
    sc = np.abs(coef).max(0)
    tie = np.abs(Xz.T @ vecs).max(1) / len(X) * 1e-6  # tie-break zero-coef features
    return sc + tie

def diffusion_components(Z, k=15, n=10):
    W = knn_affinity(Z, k)
    d = np.asarray(W.sum(1)).ravel(); Dm = sparse.diags(1 / np.sqrt(d))
    vals, vecs = eigsh(Dm @ W @ Dm, k=n + 1, which='LA')
    o = np.argsort(-vals)
    return vals[o], (Dm @ vecs[:, o])

def baseline_scores(X, seed=0, n_select=30, oracle=None):
    """Return dict name -> score (higher = better) over ALL features."""
    rng = np.random.default_rng(seed)
    Xz = zscore(X); p = X.shape[1]
    W, Z = graph_from_features(X, 10, 20)
    _, idx = knn(Z, 10)
    out = {}
    out['random'] = rng.random(p)
    out['variance'] = X.var(0)
    m = X.mean(0)
    out['dispersion'] = np.where(m > 1e-8, X.var(0) / np.maximum(m, 1e-8), 0)
    _, Vt, ev = pca(Xz, 10)
    out['pca_loading'] = (np.abs(Vt) * np.sqrt(ev)[:, None]).max(0)
    out['laplacian'] = -laplacian_score(Xz, W)
    out['mcfs'] = mcfs(X, W, 5, n_select)
    out['hotspot_moran'] = morans_i(Xz, W)
    out['nbr_variance'] = -neighbourhood_variance(Xz, idx)
    cent, lab = kmeans2(Z, 10, minit='++', seed=seed)
    G = np.eye(lab.max() + 1)[lab]; ng = G.sum(0)
    means = (G.T @ Xz) / np.maximum(ng, 1)[:, None]
    ssb = (ng[:, None] * means ** 2).sum(0); sst = (Xz ** 2).sum(0)
    out['cluster_F'] = ssb / np.maximum(sst - ssb, 1e-12)
    _, dc = diffusion_components(Z, 15, 5)
    dc = dc[:, 1:]
    dcz = (dc - dc.mean(0)) / dc.std(0)
    out['diffusion_corr'] = np.abs(Xz.T @ dcz / len(X)).max(1)
    if oracle is not None:
        out['oracle_rho'] = np.abs([spearmanr(X[:, j], oracle)[0] if X[:, j].std() > 0 else 0 for j in range(p)])
    out = {k: np.nan_to_num(v, nan=-np.inf) for k, v in out.items()}
    return out

# ----------------------------------------------------------------------------- DPT
def dpt(Z, root, k=15, n_comp=15):
    vals, vecs = diffusion_components(Z, k, n_comp)
    lam = vals[1:]; psi = vecs[:, 1:]
    lam = np.clip(lam, -0.999, 0.999)
    Mpsi = psi * (lam / (1 - lam))
    return np.sqrt(((Mpsi - Mpsi[root]) ** 2).sum(1))

def panel_embedding(X, feats, npc=10):
    Xz = zscore(X[:, feats])
    return pca(Xz, npc)[0] if Xz.shape[1] > npc else Xz

# ----------------------------------------------------------------------------- metrics
def knn_cv_accuracy(Z, y, k=5, folds=5, seed=0, balanced=False):
    rng = np.random.default_rng(seed); n = len(y)
    f = rng.permutation(n) % folds; pred = np.empty(n, dtype=y.dtype)
    for i in range(folds):
        tr, te = f != i, f == i
        _, nn = cKDTree(Z[tr]).query(Z[te], k=k)
        ytr = y[tr][nn]
        u = np.unique(y)
        counts = np.stack([(ytr == c).sum(1) for c in u], 1)
        pred[te] = u[np.argmax(counts, 1)]
    if balanced:
        return np.mean([(pred[y == c] == c).mean() for c in np.unique(y)])
    return (pred == y).mean()

def nmi(a, b):
    _, ai = np.unique(a, return_inverse=True); _, bi = np.unique(b, return_inverse=True)
    C = np.zeros((ai.max() + 1, bi.max() + 1)); np.add.at(C, (ai, bi), 1)
    P = C / C.sum(); pa = P.sum(1); pb = P.sum(0)
    nz = P > 0
    I = (P[nz] * np.log(P[nz] / (pa[:, None] * pb[None, :])[nz])).sum()
    ha = -(pa[pa > 0] * np.log(pa[pa > 0])).sum(); hb = -(pb[pb > 0] * np.log(pb[pb > 0])).sum()
    return 2 * I / (ha + hb) if ha + hb > 0 else 0.0

def kmeans_nmi(Z, y, n_clusters, seeds=5):
    r = []
    for s in range(seeds):
        _, lab = kmeans2(Z, n_clusters, minit='++', seed=s)
        r.append(nmi(lab, y))
    return float(np.mean(r))

def mixing_entropy(Z, b, k=30):
    _, nn = knn(Z, k)
    u = np.unique(b); H = []
    for i in range(len(Z)):
        pr = np.array([(b[nn[i]] == c).mean() for c in u]); pr = pr[pr > 0]
        H.append(-(pr * np.log2(pr)).sum())
    return float(np.mean(H))

# ----------------------------------------------------------------------------- PHATE re-implementation
def phate_embed(X, knn_=5, alpha=40, t=None, n_iter=100, seed=0, npca=None):
    """PHATE (Moon et al. 2019) re-implementation: alpha-decay kernel -> P -> t by VNE knee ->
    potential -log(P^t+eps) -> classical MDS init -> SMACOF metric MDS."""
    Z = X
    if npca and X.shape[1] > npca:
        Z = pca(X, npca)[0]
    D = squareform(pdist(Z))
    eps_k = np.sort(D, 1)[:, knn_]
    K = 0.5 * np.exp(-(D / eps_k[:, None]) ** alpha) + 0.5 * np.exp(-(D / eps_k[None, :]) ** alpha)
    K[K < 1e-4] = 0
    P = K / K.sum(1, keepdims=True)
    if t is None:
        ev = np.abs(np.linalg.eigvals(P)).real if len(P) < 1500 else None
        d = np.sqrt(K.sum(1)); A = K / d[:, None] / d[None, :]
        ev = np.abs(np.linalg.eigvalsh(A))
        ts = np.arange(1, 60); H = []
        for tt in ts:
            e = ev ** tt; e = e / e.sum(); e = e[e > 0]; H.append(-(e * np.log(e)).sum())
        H = np.array(H)
        # knee: max distance to the chord
        x0, y0, x1, y1 = ts[0], H[0], ts[-1], H[-1]
        dist = np.abs((y1 - y0) * ts - (x1 - x0) * H + x1 * y0 - y1 * x0)
        t = int(ts[np.argmax(dist)])
    Pt = np.linalg.matrix_power(P, t)
    U = -np.log(Pt + 1e-7)
    Du = squareform(pdist(U))
    # classical MDS
    n = len(Du); J = np.eye(n) - 1.0 / n
    B = -0.5 * J @ (Du ** 2) @ J
    w, v = eigsh(B, k=2, which='LA')
    Y = v * np.sqrt(np.maximum(w, 0))
    # SMACOF iterations
    for _ in range(n_iter):
        Dy = squareform(pdist(Y)); Dy[Dy == 0] = 1e-12
        Bm = -Du / Dy; np.fill_diagonal(Bm, 0); np.fill_diagonal(Bm, -Bm.sum(1))
        Y = Bm @ Y / n
    return Y, t
