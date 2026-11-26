"""Positive control: plant a structure-defined label (symmetric M-X-M' bridge) at ~5% prevalence on the same graphs."""
import sys; sys.path.insert(0, '/workspace/code')
import numpy as np
from common import N_MAG
from model import featurize

def has_motif(item, tol=0.1):
    x, s, d, v, t = item; mag = (x < N_MAG).numpy()
    nb = {}
    for a, b, vv in zip(s.tolist(), d.tolist(), v.tolist()):
        if not mag[b] and mag[a] and a != b: nb.setdefault(b, []).append((a, vv))
    for X, lst in nb.items():
        for i in range(len(lst)):
            for j in range(i + 1, len(lst)):
                if lst[i][0] != lst[j][0] and abs(lst[i][1] - lst[j][1]) < tol: return True
    return False

def control_subset(gs, n_pos, seed):
    """Keep all motif-free graphs as negatives and n_pos randomly chosen motif graphs as positives."""
    items = featurize(gs); mot = np.array([has_motif(it) for it in items])
    rng = np.random.default_rng(seed); pos = rng.choice(np.where(mot)[0], n_pos, replace=False)
    keep = np.sort(np.concatenate([np.where(~mot)[0], pos])); y = mot[keep].astype(int)
    return keep, y, mot
