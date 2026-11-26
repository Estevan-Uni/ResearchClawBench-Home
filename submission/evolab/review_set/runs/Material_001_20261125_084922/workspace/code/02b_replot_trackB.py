"""Re-draw Fig. 2 from saved Track-B outputs (no retraining): cleaner learning-curve axis."""
import os, json
import numpy as np
from common import style, IMG, OUT, C
r = json.load(open(os.path.join(OUT, 'gnn_trackB_results.json'))); z = np.load(os.path.join(OUT, 'gnn_trackB_curves.npz'))
cl = np.load(os.path.join(OUT, 'trackB_clusters.npz')); rng = np.random.default_rng(0); perm = rng.permutation(5000)
v2 = cl['Ef'][perm[:4000]].std() ** 2; NOISE = 0.005
plt = style(); fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))
for (a, b, lab, cc) in [('tr', 'va', 'full', C[0]), ('tr_nod', 'va_nod', 'no distances', C[1]), ('tr_sh', 'va_sh', 'shuffled distances', C[5])]:
    ax[0].semilogy(np.arange(1, len(z[a]) + 1), z[a] * v2, c=cc, label=f'{lab}: train')
    ax[0].semilogy(np.arange(1, len(z[b]) + 1), z[b] * v2, c=cc, ls='--', label=f'{lab}: val')
ax[0].axhline(NOISE ** 2, c='k', ls=':', lw=1, label='label-noise floor (5 meV/atom)²')
ax[0].set(xlabel='epoch', ylabel='MSE (eV²/atom²)', title='(a) Training / validation loss'); ax[0].legend(fontsize=7)
y, p, pl = z['y'], z['pred'], z['pred_lin']; lim = [y.min() - 0.05, y.max() + 0.05]
ax[1].scatter(y, pl, s=8, c=C[2], alpha=0.35, label=f"Ridge (comp+dist+pairs)  MAE={1000*np.mean(np.abs(pl-y)):.1f} meV/atom")
ax[1].scatter(y, p, s=8, c=C[0], alpha=0.7, label=f"GNN  MAE={1000*np.mean(np.abs(p-y)):.1f} meV/atom, R²={r['GNN (full)']['R2']:.4f}")
ax[1].plot(lim, lim, 'k--', lw=1); ax[1].set(xlim=lim, ylim=lim, xlabel='surrogate $E_f$ (eV/atom)', ylabel='predicted $E_f$ (eV/atom)', title='(b) Test-set parity (n = 500)'); ax[1].legend(fontsize=7.5)
lc = {int(k): v for k, v in r['learning_curve_MAE'].items()}; ns = sorted(lc)
ax[2].plot(ns[:-1], [lc[n] * 1000 for n in ns[:-1]], 'o-', c=C[0], label='GNN (100 epochs)')
ax[2].plot(ns[-1], lc[ns[-1]] * 1000, 'D', c=C[0], mfc='w', ms=8, label='GNN, n=4000 (150 epochs)')
for k, cc, lab in [('Mean', C[5], 'Mean'), ('Ridge (composition)', C[1], 'Ridge (composition)'), ('Ridge (comp + sorted distances + pair counts)', C[2], 'Ridge (comp+dist+pairs), n=4000')]:
    ax[2].axhline(r[k]['MAE'] * 1000, c=cc, ls='--', label=lab)
ax[2].axhline(NOISE * np.sqrt(2 / np.pi) * 1000, c='k', ls=':', label='label-noise floor')
ax[2].set_xscale('log'); ax[2].set_yscale('log'); ax[2].set_xticks(ns); ax[2].set_xticklabels([str(n) for n in ns]); ax[2].minorticks_off()
ax[2].set(xlabel='training-set size', ylabel='test MAE (meV/atom)', title='(c) Learning curve vs baselines'); ax[2].legend(fontsize=7, loc='lower left')
fig.tight_layout(); fig.savefig(os.path.join(IMG, 'fig2_gnn_trackB.png')); print('ok')
