"""Fixed episodic few-shot test set shared by every method.
Novel classes (never labelled during training): Analysis, Backdoor, Shellcode, Worms.
Settings: '4way' (novel only) and '5way' (Benign + 4 novel; Benign pool from the test period).
Pools: 'all' (every flow) and 'dedup' (one flow per unique feature vector per class).
K in {1,5,10}, Q=15 queries/class, 500 episodes each. Stores global event indices."""
import sys, numpy as np
sys.path.insert(0, '/workspace/code')
from common import *

D = load(); y = D['attack']; X = D['msg']
tr, va, te = chrono_split(len(y))
rng = np.random.default_rng(12345)
pools = {}
for pool in ['all', 'dedup']:
    P = {}
    for c in NOVEL + [BENIGN]:
        idx = np.where(y == c)[0]
        if c == BENIGN:
            idx = idx[idx >= te[0]]
        if pool == 'dedup':
            _, first = np.unique(X[idx], axis=0, return_index=True)
            idx = np.sort(idx[first])
        P[c] = idx
    pools[pool] = P
out = {}
Q, NEP = 15, 500
for pool in ['all', 'dedup']:
    for setting, cls in [('4way', NOVEL), ('5way', [BENIGN] + NOVEL)]:
        for K in [1, 5, 10]:
            S = np.zeros((NEP, len(cls), K), np.int64); Qi = np.zeros((NEP, len(cls), Q), np.int64)
            for e in range(NEP):
                for j, c in enumerate(cls):
                    pick = rng.choice(pools[pool][c], K + Q, replace=False)
                    S[e, j] = pick[:K]; Qi[e, j] = pick[K:]
            out[f'{pool}_{setting}_K{K}_S'] = S; out[f'{pool}_{setting}_K{K}_Q'] = Qi
            out[f'{pool}_{setting}_classes'] = np.array(cls)
np.savez(f'{OUT}/episodes.npz', **out)
print({p: {CLASSES[c]: len(v) for c, v in pools[p].items()} for p in pools})
