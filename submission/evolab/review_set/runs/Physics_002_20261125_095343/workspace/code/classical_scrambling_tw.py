"""(a) Scrambling diagnostics at N=16 (collision prob 2^N sum p^2, half-cut entanglement entropy) vs depth;
(c) contraction-width heuristic (line-graph elimination: best of time-ordered and greedy min-degree/min-fill)."""
import os, sys, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from circuits import random_circuit, statevector_series
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
N = 16
rows = []
page = sum(1 / k for k in range(2 ** 8 + 1, 2 ** 16 + 1)) - (2 ** 8 - 1) / (2 * 2 ** 8)  # Page entropy (nats)
for g in ['all2all', '2d', '1d']:
    for s in range(5):
        L = random_circuit(N, 24, g, 300 + s)
        rcut = np.random.default_rng(s).permutation(N)[:8] if g == 'all2all' else np.arange(8)
        for d, psi in enumerate(statevector_series(N, L), 1):
            p = np.abs(psi.ravel()) ** 2
            other = [q for q in range(N) if q not in rcut]
            M = np.transpose(psi, list(rcut) + other).reshape(2 ** 8, -1)
            sv = np.linalg.svd(M, compute_uv=False) ** 2; sv = sv[sv > 1e-16]
            rows.append(dict(geometry=g, seed=s, d=d, collision=2 ** N * (p ** 2).sum(), S_half=-(sv * np.log(sv)).sum() / np.log(2)))
S = pd.DataFrame(rows); S.to_csv(os.path.join(ROOT, 'outputs/scrambling_N16.csv'), index=False)
print('Page entropy (bits)', page / np.log(2))
print(S.groupby(['geometry', 'd'])[['collision', 'S_half']].mean().unstack(0).round(3).to_string())

def line_graph(N, L):
    seg = {}; nid = 0; cur = {}
    adj = {}
    def new():
        nonlocal nid; nid += 1; adj[nid - 1] = set(); return nid - 1
    order = []
    for q in range(N): cur[q] = new(); order.append(cur[q])
    for us, prs in L:
        for a, b in prs:
            ia, ib = cur[a], cur[b]; oa, ob = new(), new(); order += [oa, ob]
            cl = [ia, ib, oa, ob]
            for x in cl:
                for y in cl:
                    if x != y: adj[x].add(y)
            cur[a], cur[b] = oa, ob
    return adj, order

def elim_width(adj, order=None):
    adj = {k: set(v) for k, v in adj.items()}; w = 0
    remaining = set(adj)
    while remaining:
        if order is not None: v = order.pop(0)
        else:
            md = min(len(adj[x]) for x in remaining)
            cands = [x for x in remaining if len(adj[x]) == md][:30]
            def fill(x):
                nb = list(adj[x]); return sum(1 for i in range(len(nb)) for j in range(i + 1, len(nb)) if nb[j] not in adj[nb[i]])
            v = min(cands, key=fill)
        nb = adj[v]; w = max(w, len(nb))
        for x in nb:
            adj[x] |= nb; adj[x].discard(x); adj[x].discard(v)
        remaining.discard(v); del adj[v]
    return w

rows = []
for g in ['all2all', '2d', '1d']:
    for d in [2, 4, 6, 8, 10, 12, 16, 20, 24]:
        for s in range(2):
            L = random_circuit(56, d, g, 1234 + s)
            adj, order = line_graph(56, L)
            wt = elim_width(adj, list(order)); wg = elim_width(adj)
            rows.append(dict(geometry=g, N=56, d=d, seed=s, width_time=wt, width_greedy=wg, width=min(wt, wg)))
            print(g, d, s, wt, wg, flush=True)
W = pd.DataFrame(rows); W.to_csv(os.path.join(ROOT, 'outputs/contraction_width_N56.csv'), index=False)
print('done')
