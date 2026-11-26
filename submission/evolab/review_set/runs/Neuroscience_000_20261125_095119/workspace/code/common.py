"""Shared helpers for the SimBA-style audit."""
import numpy as np, pandas as pd, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, 'data'); OUT = os.path.join(ROOT, 'outputs'); IMG = os.path.join(ROOT, 'report', 'images')
os.makedirs(OUT, exist_ok=True); os.makedirs(IMG, exist_ok=True)
BODYPARTS = ['Nose', 'Ear_left', 'Ear_right', 'Center', 'Lat_left', 'Lat_right', 'Tail_base', 'Tail_end']
POSE_COLS = [f'{bp}_{a}_{c}' for a in (1, 2) for bp in BODYPARTS for c in ('x', 'y', 'p')]
BEHAVIOURS = ['Attack', 'Sniffing']
FPS = 30  # assumed
SEED = 42

def load_together():
    X = pd.read_csv(os.path.join(DATA, 'Together_1_features_extracted.csv'), index_col=0)
    T = pd.read_csv(os.path.join(DATA, 'Together_1_targets_inserted.csv'), index_col=0)
    return X, T

def load_reference():
    return pd.read_csv(os.path.join(DATA, 'Together_1_machine_results_reference.csv'), index_col=0)

def bouts(y):
    """Return list of (start, end_inclusive) of runs of 1s."""
    y = np.asarray(y).astype(int); d = np.diff(np.r_[0, y, 0])
    s = np.where(d == 1)[0]; e = np.where(d == -1)[0] - 1
    return list(zip(s, e))

def close_gaps(y, max_gap=1):
    """Fill runs of 0 of length <= max_gap that are flanked by 1s (annotation comb artefact)."""
    y = np.asarray(y).astype(int).copy(); b = bouts(y)
    for (s0, e0), (s1, e1) in zip(b[:-1], b[1:]):
        if s1 - e0 - 1 <= max_gap: y[e0 + 1:s1] = 1
    return y
