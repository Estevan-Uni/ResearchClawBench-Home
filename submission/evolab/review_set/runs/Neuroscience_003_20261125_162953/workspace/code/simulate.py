"""Own numpy count simulator for linear / bifurcating / tree trajectories with ground-truth DE genes."""
import numpy as np

TOPO = {
    # segment: (parent, t_start, t_end); paths = leaf segments
    'linear': {'segs': {'S0': (None, 0, 1 / 3), 'S1': ('S0', 1 / 3, 2 / 3), 'S2': ('S1', 2 / 3, 1)},
               'leaves': ['S2']},
    'bifurcating': {'segs': {'R': (None, 0, 0.4), 'A': ('R', 0.4, 1), 'B': ('R', 0.4, 1)},
                    'leaves': ['A', 'B']},
    'tree': {'segs': {'R': (None, 0, 0.33), 'A': ('R', 0.33, 1), 'B': ('R', 0.33, 0.66),
                      'B1': ('B', 0.66, 1), 'B2': ('B', 0.66, 1)},
             'leaves': ['A', 'B1', 'B2']},
}

def _path(segs, leaf):
    p = [leaf]
    while segs[p[-1]][0] is not None:
        p.append(segs[p[-1]][0])
    return p[::-1]

def simulate(topology='bifurcating', n_cells=1500, n_genes=500, n_de=100, effect_scale=1.0,
             lib_scale=1.0, dropout_mid=None, dropout_slope=1.0, n_confound=40, theta=5.0, seed=0):
    rng = np.random.default_rng(seed)
    T = TOPO[topology]; segs = T['segs']; names = list(segs)
    leaf = rng.choice(T['leaves'], n_cells)
    t = rng.uniform(0, 1, n_cells)
    seg = np.empty(n_cells, dtype=object)
    for i in range(n_cells):
        for s in _path(segs, leaf[i]):
            if segs[s][1] <= t[i] <= segs[s][2]:
                seg[i] = s
    on_path = {s: np.array([s in _path(segs, l) for l in leaf]) for s in names}
    base = rng.normal(0.0, 1.0, n_genes)
    logmu = np.tile(base, (n_cells, 1))
    de = rng.choice(n_genes, n_de, replace=False)
    for g in de:
        s = names[rng.integers(len(names))]
        a, b = segs[s][1], segs[s][2]
        eff = rng.uniform(1.0, 2.5) * rng.choice([-1, 1]) * effect_scale
        if rng.random() < 0.5:  # sigmoid switch inside segment, persisting downstream
            mid = rng.uniform(a + 0.2 * (b - a), b - 0.2 * (b - a))
            prof = 1 / (1 + np.exp(-(t - mid) / (0.08 * (b - a) + 0.01)))
            # only on lineages passing through segment
            prof = prof * on_path[s]
        else:  # transient bump within segment
            mid = (a + b) / 2; w = (b - a) / 3
            prof = np.exp(-0.5 * ((t - mid) / w) ** 2) * on_path[s]
        logmu[:, g] += eff * prof
    # trajectory-independent confounder (e.g. cell size / stress programme) on some non-DE genes
    non_de = np.setdiff1d(np.arange(n_genes), de)
    conf = rng.choice(non_de, n_confound, replace=False)
    z = rng.normal(0, 1, n_cells)
    logmu[:, conf] += 0.8 * z[:, None] * rng.choice([-1, 1], n_confound)
    lib = rng.lognormal(np.log(lib_scale), 0.4, n_cells)
    mu = lib[:, None] * np.exp(logmu)
    counts = rng.negative_binomial(theta, theta / (theta + mu)).astype(float)
    if dropout_mid is not None:
        pdrop = 1 / (1 + np.exp(dropout_slope * (np.log(mu) - dropout_mid)))
        counts[rng.random(counts.shape) < pdrop] = 0
    tot = counts.sum(1); tot[tot == 0] = 1
    Xn = np.log1p(counts / tot[:, None] * np.median(tot))
    is_de = np.zeros(n_genes, bool); is_de[de] = True
    return dict(X=Xn, t=t, seg=seg.astype(str), leaf=leaf, is_de=is_de, confound=conf,
                zero_frac=(counts == 0).mean())
