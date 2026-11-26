"""Track A: the same numpy CGCNN trained on the *provided* property-prediction block.

Alignment assumption (the file gives 100 Z / 117 descriptors / 97 targets with no mapping):
sample i is the K5 template (dataset edge list) with 5 boron atoms (Z=5) whose scalar node
descriptors are node_features[i:i+5] (sliding window; i = 0..96 uses indices 0..100).
Node input = [one-hot(Z=5) (constant), descriptor]. No coordinates -> no edge features.
Target read as formation energy per atom (eV/atom, assumed units).
"""
import os
import numpy as np
from common import load_dataset, save_json, style, IMG, OUT, C
import gnn_np as G

SEED = 0; print('seed', SEED)
d = load_dataset(); y = d['targets']; nf = d['node_features']; n = len(y)
EDGES = d['edge_list'].reshape(-1, 2)
MASK = np.zeros((5, 5)); MASK[EDGES[:, 0], EDGES[:, 1]] = 1; MASK += MASK.T
X = np.stack([np.stack([np.ones(5), nf[i:i + 5]], 1) for i in range(n)])     # (97,5,2)
E0 = np.zeros((n, 5, 5, 0))
print('node-input std across graphs:', X[:, :, 1].std(0).mean())

def mae(a, b): return float(np.mean(np.abs(a - b)))

def fit_gnn(tr, va, te, yy, seed, epochs=300):
    ym, ys = yy[tr].mean(), yy[tr].std()
    p = G.init_params(2, 0, H=16, L=2, Hfc=16, seed=seed)
    data = dict(Xtr=X[tr], Etr=E0[tr], ytr=(yy[tr] - ym) / ys, Xva=X[va], Eva=E0[va], yva=(yy[va] - ym) / ys, mask=MASK)
    best, hist, bep = G.train(p, data, epochs=epochs, bs=16, lr=3e-3, wd=1e-4, seed=seed, verbose=0)
    pred = G.forward(best, X[te], E0[te], MASK) * ys + ym
    emb = G.embed(best, X, E0, MASK)
    return pred, hist, bep, ys, float(emb.std(0).mean())

rng = np.random.default_rng(SEED)
rows = []; curves = None
for s in range(10):
    perm = rng.permutation(n); tr, va, te = perm[:77], perm[77:87], perm[87:]
    pred, hist, bep, ys, estd = fit_gnn(tr, va, te, y, seed=s)
    ysc = rng.permutation(y[np.r_[tr, va]]); yscr = y.copy(); yscr[np.r_[tr, va]] = ysc
    pred_sc, _, _, _, _ = fit_gnn(tr, va, te, yscr, seed=s)
    idx = np.arange(n); lut = {r: y[tr][idx[tr] % 8 == r].mean() for r in range(8)}
    pred_mod8 = np.array([lut[i % 8] for i in te])
    A = np.c_[nf[np.arange(n)[:, None] + np.arange(5)], np.ones(n)]
    w = np.linalg.lstsq(A[tr], y[tr], rcond=None)[0]
    rows.append(dict(split=s, GNN=mae(y[te], pred), mean=mae(y[te], np.full(len(te), y[tr].mean())),
                     linear=mae(y[te], A[te] @ w), GNN_yscrambled=mae(y[te], pred_sc), index_mod8=mae(y[te], pred_mod8),
                     best_epoch=bep, emb_std=estd))
    if s == 0:
        curves = (hist, ys, y[te], pred)
    print(rows[-1], flush=True)

# blocked split: train on first 70%, val next 10%, test last 20% (extrapolation in index)
tr, va, te = np.arange(68), np.arange(68, 78), np.arange(78, 97)
pred_b, hist_b, _, ys_b, _ = fit_gnn(tr, va, te, y, seed=0)
idx = np.arange(n); lut = {r: y[tr][idx[tr] % 8 == r].mean() for r in range(8)}
blocked = dict(GNN=mae(y[te], pred_b), mean=mae(y[te], np.full(len(te), y[tr].mean())),
               index_mod8=mae(y[te], np.array([lut[i % 8] for i in te])))
keys = ['GNN', 'mean', 'linear', 'GNN_yscrambled', 'index_mod8']
summ = {k: dict(mean=float(np.mean([r[k] for r in rows])), std=float(np.std([r[k] for r in rows]))) for k in keys}
win = int(sum(r['GNN'] < r['mean'] for r in rows))
save_json(dict(random_splits=rows, summary_random=summ, gnn_beats_mean_in=f'{win}/10', blocked=blocked,
               alignment='sample i <- node_features[i:i+5], Z=5, K5 edges'), 'gnn_trackA_results.json')
print(summ, blocked, win)

plt = style()
fig, ax = plt.subplots(1, 3, figsize=(14, 4))
hist, ys, yt, pt = curves
ax[0].semilogy(np.array(hist['train']) * ys ** 2, c=C[0], label='train'); ax[0].semilogy(np.array(hist['val']) * ys ** 2, c=C[1], label='val')
ax[0].axhline(np.var(y), c='k', ls=':', label='target variance'); ax[0].set(xlabel='epoch', ylabel='MSE (eV²/atom²)', title='(a) Provided data: loss (split 0)'); ax[0].legend()
for k, cc in zip(keys, [C[0], C[5], C[2], C[3], C[4]]):
    v = [r[k] for r in rows]; ax[1].scatter(np.full(10, keys.index(k)) + rng.uniform(-.1, .1, 10), v, c=cc, s=15)
    ax[1].errorbar(keys.index(k) + 0.25, np.mean(v), np.std(v), fmt='D', c=cc, capsize=3)
ax[1].set_xticks(range(5)); ax[1].set_xticklabels(['GNN', 'Mean', 'Linear', 'GNN\n(y-scrambled)', 'index mod 8\n(leak control)'], fontsize=8)
ax[1].set(ylabel='test MAE (eV/atom)', title='(b) 10 random 77/10/10 splits')
ax[2].plot(y, 'k.-', lw=0.6, ms=4, label='target'); ax[2].plot(te, pred_b, 'o', c=C[0], ms=4, label=f"GNN, blocked test (MAE {blocked['GNN']:.2f})")
ax[2].plot(te, [lut[i % 8] for i in te], 's', c=C[4], ms=4, mfc='none', label=f"index mod 8 (MAE {blocked['index_mod8']:.2f})")
ax[2].axvspan(77.5, 97, color=C[1], alpha=0.08); ax[2].set(xlabel='sample index', ylabel='target (eV/atom)', title='(c) Blocked split (last 20% = test)'); ax[2].legend(fontsize=7)
fig.tight_layout(); fig.savefig(os.path.join(IMG, 'fig4_gnn_trackA_provided.png'))
