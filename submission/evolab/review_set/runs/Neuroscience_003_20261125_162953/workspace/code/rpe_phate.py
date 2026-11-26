import sys; sys.path.insert(0, '/workspace/code')
import numpy as np, pandas as pd, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, seaborn as sns
from core import *
try:
    import phate; HAVE_PHATE = True
except Exception:
    HAVE_PHATE = False
X, obs, var = load_rpe()
R = dict(np.load('/workspace/outputs/rpe_rankings.npz'))
panels = [('All 241 features', np.arange(241)), ('DELVE-faithful top-30', R['DELVE-faithful'][:30]),
          ('DELVE +dedup top-30', R['+dedup'][:30]), ('Variance top-30', R['variance'][:30]),
          ('Laplacian (full graph) top-30', R['laplacian'][:30])]
emb = {}
for name, f in panels:
    Xz = zscore(X[:, f])
    if HAVE_PHATE:
        Y = phate.PHATE(knn=5, n_jobs=1, verbose=0, random_state=0).fit_transform(Xz); t = 'auto'
    else:
        Y, t = phate_embed(Xz, knn_=5, alpha=40, npca=20 if Xz.shape[1] > 20 else None, n_iter=60)
    emb[name] = Y; print(name, 't=', t, flush=True)
np.savez('/workspace/outputs/rpe_phate.npz', **{k: v for k, v in emb.items()})
order = ['G0', 'G1', 'S', 'G2']; pal = dict(zip(order, sns.color_palette('Set2', 4)))
fig, ax = plt.subplots(len(panels), 3, figsize=(13, 4.0 * len(panels)))
for r, (name, _) in enumerate(panels):
    Y = emb[name]
    for p in order:
        m = obs['phase'] == p; ax[r, 0].scatter(Y[m, 0], Y[m, 1], s=2, color=pal[p], label=p)
    ax[r, 0].legend(markerscale=5, fontsize=8)
    s = ax[r, 1].scatter(Y[:, 0], Y[:, 1], s=2, c=obs['age'], cmap='viridis'); plt.colorbar(s, ax=ax[r, 1], label='age (h)')
    for b, col in zip(['1', '2'], ['C0', 'C1']):
        m = obs['batch'] == b; ax[r, 2].scatter(Y[m, 0], Y[m, 1], s=2, color=col, alpha=.5, label='batch ' + b)
    ax[r, 2].legend(markerscale=5, fontsize=8)
    ax[r, 0].set_ylabel(name, fontsize=11)
    for c, ttl in enumerate(['phase', 'annotated age', 'batch']):
        ax[r, c].set_xticks([]); ax[r, c].set_yticks([])
        if r == 0: ax[r, c].set_title(ttl)
fig.suptitle('PHATE embeddings (' + ('phate package' if HAVE_PHATE else 're-implemented PHATE, α=40, knn=5') + ')', y=1.0)
plt.tight_layout(); plt.savefig('/workspace/report/images/fig3_phate_panels.png', dpi=110, bbox_inches='tight')
print('done')
