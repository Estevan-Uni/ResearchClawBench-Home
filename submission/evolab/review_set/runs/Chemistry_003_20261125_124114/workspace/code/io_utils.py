"""Minimal extended-XYZ reader (no ASE dependency)."""
import re
import numpy as np


def read_extxyz(fn):
    lines = open(fn).read().split('\n')
    frames = []
    i = 0
    while i < len(lines) and lines[i].strip():
        n = int(lines[i])
        header = lines[i + 1]
        info = {}
        for m in re.finditer(r'(\w+)=("[^"]*"|\S+)', header):
            k, v = m.group(1), m.group(2).strip('"')
            info[k] = v
        rows = [l.split() for l in lines[i + 2:i + 2 + n]]
        sym = [r[0] for r in rows]
        arr = np.array([[float(x) for x in r[1:]] for r in rows])
        fr = dict(symbols=sym, pos=arr[:, :3])
        if arr.shape[1] >= 6:
            fr['forces'] = arr[:, 3:6]
        if 'energy' in info:
            fr['energy'] = float(info['energy'])
        if 'total_charge' in info:
            fr['total_charge'] = float(info['total_charge'])
        if 'true_charges' in info:
            fr['true_charges'] = np.array([float(x) for x in info['true_charges'].split()])
        frames.append(fr)
        i += 2 + n
    return frames
