"""Data loading utilities for the MAPF .npy benchmark files.
Each file holds 5 consecutive np.save objects:
 [obstacle grid (-1 obstacle, 0 free), start-occupancy grid, dict, starts (N,2), goals (N,2)]
"""
import os, re, glob
import numpy as np

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')

FAMILIES = ['random_small', 'random_medium', 'random_large', 'empty', 'maze', 'room', 'warehouse']

def load_instance(path):
    objs = []
    with open(path, 'rb') as fh:
        for _ in range(5):
            objs.append(np.load(fh, allow_pickle=True))
    grid = (objs[0] == -1).astype(np.uint8)   # 1 = obstacle
    starts = objs[3].astype(np.int64); goals = objs[4].astype(np.int64)
    return grid, starts, goals

def list_configs():
    out = []
    for fam in FAMILIES:
        for d in sorted(glob.glob(os.path.join(DATA, fam, '*'))):
            name = os.path.basename(d)
            n = int(re.findall(r'\d+', name)[0])
            out.append((fam, name, n, d))
    out.sort(key=lambda x: (FAMILIES.index(x[0]), x[2]))
    return out

def instance_files(cfg_dir):
    fs = glob.glob(os.path.join(cfg_dir, '*.npy'))
    fs.sort(key=lambda f: int(re.findall(r'(\d+)\.npy', f)[0]))
    return fs
