"""Data audit of the M-AI-Synth file: array lengths, structure, periodicity."""
import numpy as np
from common import load_dataset, save_json, style, IMG, C
import os

d = load_dataset()
for k, v in d.items():
    print(k, (len(v) if hasattr(v, '__len__') else v))

y = d['targets']; nf = d['node_features']; el = d['edge_list'].reshape(-1, 2)
a, b = d['lattice_a'], d['lattice_b']
# node features: 13x9 grid?
grid = nf.reshape(13, 9)
r, c = np.meshgrid(np.arange(13), np.arange(9), indexing='ij')
grid_err = np.abs(grid - (-2 + 0.2 * r + 0.5 * c)).max()
# periodicity of target
def first_periodic(x, p):
    for s in range(len(x) - p):
        if np.allclose(x[s:-p], x[s + p:]):
            return s
    return None
per = {p: first_periodic(y, p) for p in range(1, 20)}
per_y = min((p for p in per if per[p] is not None and per[p] < len(y) - 2 * p), default=None)
per_a = min(p for p in range(1, 20) if np.allclose(a[:-p], a[p:]))
shift = [s for s in range(7) if np.allclose(np.roll(a, -s)[:95], b[:95])]
pairs = np.unique(np.round(np.c_[a, b], 4), axis=0)
edges_K5 = sorted(map(tuple, el)) == [(i, j) for i in range(5) for j in range(i + 1, 5)]
audit = dict(
    lengths={k: (len(v) if hasattr(v, '__len__') else 1) for k, v in d.items()},
    atomic_numbers_unique=np.unique(d['atomic_numbers']).tolist(),
    node_features_grid13x9_maxerr=grid_err, node_feature_unique=len(np.unique(nf)),
    edge_list_is_K5=edges_K5, n_edges=len(el),
    target_unique=len(np.unique(y)), target_mean=y.mean(), target_std=y.std(),
    target_period=per_y, target_period_start=per[per_y] if per_y else None,
    lattice_a_period=per_a, lattice_unique_a=len(np.unique(a)),
    lattice_b_is_a_shifted_by=shift, unique_ab_pairs=pairs.tolist(),
    optimisation=dict(T_range=d['T_range'].tolist(), t_range=d['t_range'].tolist(),
                      T_opt=d['T_opt'], t_opt=d['t_opt'], noise=d['noise'], n_iter=d['n_iter']))
save_json(audit, 'data_audit.json')
print({k: v for k, v in audit.items() if k != 'unique_ab_pairs'})
print('pairs', pairs)

plt = style()
fig, ax = plt.subplots(2, 3, figsize=(13, 7))
ax[0, 0].plot(y, 'o-', ms=3, c=C[0]); ax[0, 0].set(title='(a) Property targets (n=97)', xlabel='index', ylabel='target (eV/atom, assumed)')
ax[0, 0].axvspan(per[per_y], len(y), color=C[1], alpha=0.08, label=f'period-{per_y} regime')
ax[0, 0].legend(fontsize=8)
im = ax[0, 1].imshow(grid, cmap='viridis', aspect='auto'); fig.colorbar(im, ax=ax[0, 1])
ax[0, 1].set(title='(b) node_features reshaped 13x9\n(x = -2 + 0.2 r + 0.5 c, exact)', xlabel='column', ylabel='row'); ax[0, 1].grid(False)
th = np.linspace(0, 2 * np.pi, 6)[:-1]; P = np.c_[np.cos(th), np.sin(th)]
for i, j in el:
    ax[0, 2].plot(*P[[i, j]].T, c='grey', lw=1)
ax[0, 2].scatter(*P.T, s=500, c=C[0], zorder=3)
for i in range(5):
    ax[0, 2].text(*P[i], f'{i}\nZ=5', ha='center', va='center', color='w', fontsize=7)
ax[0, 2].set(title='(c) Edge list = complete graph K5', aspect='equal'); ax[0, 2].axis('off')
ax[1, 0].plot(a, 'o-', ms=2, c=C[0], label='a'); ax[1, 0].plot(b, 's-', ms=2, c=C[1], label='b', alpha=0.7)
ax[1, 0].set(title='(d) Lattice arrays (n=101 each)', xlabel='index', ylabel='lattice parameter (Å, assumed)'); ax[1, 0].legend()
ax[1, 1].scatter(a, b, s=80, c=C[2], alpha=0.3)
ax[1, 1].set(title=f'(e) (a,b) pairs: only {len(pairs)} unique', xlabel='a (Å)', ylabel='b (Å)')
ax[1, 2].hist(y, bins=20, color=C[0], alpha=0.8); ax[1, 2].set(title=f'(f) Target distribution ({len(np.unique(y))} unique values)', xlabel='target')
fig.tight_layout(); fig.savefig(os.path.join(IMG, 'fig1_data_overview.png'))
