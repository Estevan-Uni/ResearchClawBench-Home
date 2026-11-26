"""Shared utilities: data loading, chronological splits, preprocessing, metrics."""
import os, json, time
import numpy as np
from sklearn.preprocessing import QuantileTransformer
from sklearn.metrics import (f1_score, roc_auc_score, accuracy_score, balanced_accuracy_score,
                             normalized_mutual_info_score, confusion_matrix, recall_score, precision_score)

ROOT = '/workspace'
OUT = os.path.join(ROOT, 'outputs')
IMG = os.path.join(ROOT, 'report', 'images')
CLASSES = ['Analysis', 'Backdoor', 'Benign', 'DoS', 'Exploits', 'Fuzzers', 'Generic',
           'Reconnaissance', 'Shellcode', 'Worms']
BENIGN = 2
NOVEL = [0, 1, 8, 9]            # Analysis, Backdoor, Shellcode, Worms (few-shot / novel)
BASE = [2, 3, 4, 5, 6, 7]       # Benign + frequent attacks (meta-train)
UNKNOWN_TYPES = [1, 3, 5, 7]    # Backdoor, DoS, Fuzzers, Reconnaissance (leave-one-out)


def load():
    d = np.load(os.path.join(OUT, 'data.npz'))
    D = {k: d[k] for k in d.files}
    # re-index nodes to 0..n-1
    nodes, inv = np.unique(np.concatenate([D['src'], D['dst']]), return_inverse=True)
    n = len(D['src'])
    D['src_i'] = inv[:n].astype(np.int64)
    D['dst_i'] = inv[n:].astype(np.int64)
    D['n_nodes'] = len(nodes)
    D['layer_src'], D['layer_dst'] = causal_layers(D['src_i'], D['dst_i'], len(nodes))
    return D


def causal_layers(src, dst, n_nodes, thr=50):
    """Layer marks are all zero in the file; derive them causally: a node is an
    'intermediate' (layer-1) device once its past degree exceeds thr."""
    deg = np.zeros(n_nodes, np.int64)
    ls = np.zeros(len(src), np.int64); ld = np.zeros(len(src), np.int64)
    for i in range(len(src)):
        s, d = src[i], dst[i]
        ls[i] = deg[s] > thr; ld[i] = deg[d] > thr
        deg[s] += 1; deg[d] += 1
    return ls, ld


def chrono_split(n, fr=(0.70, 0.15, 0.15)):
    a = int(n * fr[0]); b = int(n * (fr[0] + fr[1]))
    return np.arange(0, a), np.arange(a, b), np.arange(b, n)


def quantile_fit(X, idx_train, seed=0):
    qt = QuantileTransformer(n_quantiles=1000, output_distribution='uniform', subsample=200000,
                             random_state=seed)
    qt.fit(X[idx_train])
    return qt.transform(X).astype(np.float32)


def known_metrics(y_true, y_pred, p_attack=None, classes=range(10)):
    yb = (y_true != BENIGN).astype(int); pb = (y_pred != BENIGN).astype(int)
    out = {
        'bin_f1': f1_score(yb, pb), 'bin_acc': accuracy_score(yb, pb),
        'macro_f1': f1_score(y_true, y_pred, labels=list(classes), average='macro', zero_division=0),
        'weighted_f1': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        'acc': accuracy_score(y_true, y_pred),
        'per_class_f1': f1_score(y_true, y_pred, labels=list(classes), average=None, zero_division=0).tolist(),
        'per_class_recall': recall_score(y_true, y_pred, labels=list(classes), average=None, zero_division=0).tolist(),
    }
    att = y_true != BENIGN
    if att.any():
        out['attack_bal_acc'] = balanced_accuracy_score(y_true[att], y_pred[att])
    if p_attack is not None:
        out['bin_auc'] = roc_auc_score(yb, p_attack)
    out['cm'] = confusion_matrix(y_true, y_pred, labels=list(classes)).tolist()
    return out


def episode_metrics(y_true, y_pred):
    return f1_score(y_true, y_pred, average='macro', zero_division=0), \
        normalized_mutual_info_score(y_true, y_pred)


def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    def conv(o):
        if isinstance(o, (np.floating,)): return float(o)
        if isinstance(o, (np.integer,)): return int(o)
        if isinstance(o, np.ndarray): return o.tolist()
        raise TypeError(type(o))
    with open(path, 'w') as f:
        json.dump(obj, f, indent=1, default=conv)


def mean_ci(x):
    x = np.asarray(x, float)
    return float(x.mean()), float(1.96 * x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else 0.0


# ---------------- plotting style (reference categorical palette, fixed order) ----------------
PAL = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948']
GRAY = '#8a8985'; LIGHTGRAY = '#c9c8c3'; INK = '#0b0b0b'; INK2 = '#52514e'
# class -> colour: Benign gray, attacks in fixed order; 9th attack (Worms) dark ink as extra slot
CLASS_COL = {0: PAL[0], 1: PAL[1], 2: LIGHTGRAY, 3: PAL[2], 4: PAL[3], 5: PAL[4], 6: PAL[5],
             7: PAL[6], 8: PAL[7], 9: '#3d3c39'}


def style():
    import matplotlib as mpl
    mpl.rcParams.update({
        'figure.dpi': 130, 'savefig.dpi': 160, 'savefig.bbox': 'tight',
        'font.size': 9, 'axes.titlesize': 10, 'axes.labelsize': 9,
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2, 'ytick.color': INK2,
        'axes.grid': True, 'grid.color': '#e6e5e1', 'grid.linewidth': 0.6,
        'axes.prop_cycle': mpl.cycler(color=PAL), 'legend.frameon': False,
        'lines.linewidth': 2, 'figure.facecolor': '#fcfcfb', 'axes.facecolor': '#fcfcfb',
    })
