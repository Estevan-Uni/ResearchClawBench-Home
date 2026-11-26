import numpy as np, torch, hashlib
from scipy.stats import rankdata

ELEMENTS = ['Fe','Co','Ni','Mn','Cr','V','Ti','Nd','Pr','Sm','Gd','Ho','Er','Yb',
            'O','F','Cl','Br','I','S','Se','Te','B','C','N','P','Si','H']
N_MAG = 14

def load_cache():
    return torch.load('/workspace/outputs/graphs_cache.pt', weights_only=False)

def auc(y, s):
    y = np.asarray(y); s = np.asarray(s, float)
    n1 = y.sum(); n0 = len(y) - n1
    if n1 == 0 or n0 == 0: return np.nan
    r = rankdata(s)
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)

def ap(y, s):
    y = np.asarray(y); s = np.asarray(s, float)
    o = np.argsort(-s, kind='stable'); yy = y[o]
    tp = np.cumsum(yy); prec = tp / np.arange(1, len(yy) + 1)
    return (prec * yy).sum() / max(yy.sum(), 1)

def hits_at(y, s, k):
    o = np.argsort(-np.asarray(s, float), kind='stable')
    return int(np.asarray(y)[o[:k]].sum())

def ef_at(y, s, frac=0.05):
    k = max(1, int(round(frac * len(y))))
    return (hits_at(y, s, k) / k) / (np.mean(y))

def wilson(k, n, z=1.96):
    if n == 0: return (np.nan, np.nan, np.nan)
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0, c - h), min(1, c + h)

def bootstrap_ci(y, s, fn, B=1000, seed=0):
    rng = np.random.default_rng(seed); y = np.asarray(y); s = np.asarray(s)
    vals = []
    for _ in range(B):
        i = rng.integers(0, len(y), len(y))
        if y[i].sum() == 0: continue
        vals.append(fn(y[i], s[i]))
    return np.percentile(vals, [2.5, 97.5])

def perm_p(y, s, fn=ap, B=1000, seed=0):
    rng = np.random.default_rng(seed); y = np.asarray(y)
    obs = fn(y, s); null = np.array([fn(rng.permutation(y), s) for _ in range(B)])
    return (1 + (null >= obs).sum()) / (B + 1), null

def graph_hash(g):
    x = g['x'].numpy(); ei = g['ei'].numpy(); ea = g['ea'].numpy()
    # permutation-sensitive exact hash + permutation-invariant multiset hash
    exact = hashlib.md5(x.tobytes() + ei.tobytes() + np.round(ea, 5).tobytes()).hexdigest()
    comp = tuple(np.bincount(x, minlength=28))
    bonds = sorted((min(ELEMENTS[x[a]], ELEMENTS[x[b]]), max(ELEMENTS[x[a]], ELEMENTS[x[b]]), round(float(v), 4), int(t))
                   for (a, b), (v, t) in zip(ei.T, ea) if a < b)
    inv = hashlib.md5(repr((comp, bonds)).encode()).hexdigest()
    return exact, inv

def descriptors(g):
    """~30 hand-crafted graph descriptors (used for baselines and diagnostics)."""
    x = g['x'].numpy(); ei = g['ei'].numpy(); ea = g['ea'].numpy()
    n = len(x); mag = x < N_MAG; m = ei[0] < ei[1]
    src, dst = ei[0][m], ei[1][m]; v = ea[m, 0]; t = ea[m, 1].astype(int)
    deg = np.bincount(ei[0], minlength=n)
    mm = mag[src] & mag[dst]; ml = mag[src] ^ mag[dst]; ll = ~mag[src] & ~mag[dst]
    d = {
        'n_atoms': n, 'n_mag': mag.sum(), 'frac_mag': mag.mean(), 'n_3d': (x < 7).sum(), 'n_4f': ((x >= 7) & (x < 14)).sum(),
        'n_species': len(np.unique(x)), 'n_mag_species': len(np.unique(x[mag])), 'n_bonds': m.sum(),
        'bond_density': m.sum() / n, 'frac_isolated': (deg == 0).mean(), 'max_deg': deg.max(),
        'mean_bond': v.mean() if len(v) else 0, 'std_bond': v.std() if len(v) else 0,
        'min_bond': v.min() if len(v) else 0, 'max_bond': v.max() if len(v) else 0,
        'n_type0': (t == 0).sum(), 'n_type1': (t == 1).sum(), 'n_type2': (t == 2).sum(),
        'n_MM': mm.sum(), 'n_ML': ml.sum(), 'n_LL': ll.sum(),
        'mean_bond_MM': v[mm].mean() if mm.any() else 0, 'mean_bond_ML': v[ml].mean() if ml.any() else 0,
        'mean_bond_LL': v[ll].mean() if ll.any() else 0,
        'mag_deg_mean': deg[mag].mean() if mag.any() else 0, 'lig_deg_mean': deg[~mag].mean() if (~mag).any() else 0,
        'n_components': n_components(n, src, dst), 'n_O': (x == 14).sum(), 'n_halide': ((x >= 15) & (x <= 18)).sum(),
        'n_chalc': ((x >= 19) & (x <= 21)).sum(),
    }
    return d

def n_components(n, src, dst):
    p = list(range(n))
    def f(a):
        while p[a] != a: p[a] = p[p[a]]; a = p[a]
        return a
    for a, b in zip(src, dst): p[f(a)] = f(b)
    return len({f(i) for i in range(n)})
