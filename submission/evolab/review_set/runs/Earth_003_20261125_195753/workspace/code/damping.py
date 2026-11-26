"""Post-hoc damping control: is the cascade's RMSE gain equivalent to shrinking the single model's anomalies
toward climatology?  f_d = c + alpha_k,v (f - c), alpha fitted per lead/variable on half of the init groups and
evaluated on the other half (2-fold CV).  MSE(alpha) = alpha^2 A - 2 alpha B + O with A = |f'|^2, B = <f',o'>,
O = |o'|^2, all reconstructable exactly from metrics.npz (act, acc, truth_act).  Global scaling leaves ACC unchanged."""
import json, numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from plots import M, LEADS, VI, V8, IM, groups, SYS
O = M['truth_act'] ** 2
def parts(s):
    A = M[f'{s}_act'] ** 2; B = M[f'{s}_acc'] * M[f'{s}_act'] * M['truth_act']; return A, B
def damped_rmse(s):
    A, B = parts(s); fold = groups % 2; out = np.zeros(A.shape[1:]); alph = np.zeros((2,) + A.shape[1:])
    mse = np.zeros_like(A)
    for f in (0, 1):
        tr, te = fold != f, fold == f
        al = np.nansum(B[tr], 0) / np.nansum(A[tr], 0); alph[f] = al
        mse[te] = al[None] ** 2 * A[te] - 2 * al[None] * B[te] + O[te]
    return np.sqrt(np.nanmean(mse, 0)), alph.mean(0)
res = {}
rs = np.sqrt(np.nanmean(M['short_only_mse'], 0)); rc = np.sqrt(np.nanmean(M['cascade_mse'], 0))
rds, als = damped_rmse('short_only'); rdc, alc = damped_rmse('cascade')
re = np.sqrt(np.nanmean(M['ens_mean_mse'], 0)); lm = ~np.isnan(re[:, 0])
fig, axs = plt.subplots(2, 4, figsize=(18, 7.5))
for ax, v in zip(axs.flat, V8):
    j = VI[v]
    ax.plot(LEADS, rc[:, j] / rs[:, j], color=SYS['cascade'][1], lw=2, label='cascade')
    ax.plot(LEADS, rds[:, j] / rs[:, j], color='C0', ls='--', label='single model + post-hoc damping (CV)')
    ax.plot(LEADS, rdc[:, j] / rs[:, j], color='C3', ls=':', label='cascade + post-hoc damping (CV)')
    ax.plot(LEADS[lm], re[lm, j] / rs[lm, j], 'k.-', label='ECMWF ENS mean')
    ax.axhline(1, color='grey', lw=.8); [ax.axvline(x, color='k', lw=.5, alpha=.4) for x in (5, 10)]
    ax.set_title(v); ax.set_xlabel('lead (days)'); ax.set_ylabel('RMSE / RMSE(single model)'); ax.grid(alpha=.3); ax.set_ylim(0.2, 1.6)
    res[v] = {f'd{d}': dict(cascade=round(float(rc[d*4-1, j] / rs[d*4-1, j]), 3), damped_single=round(float(rds[d*4-1, j] / rs[d*4-1, j]), 3),
                           damped_cascade=round(float(rdc[d*4-1, j] / rs[d*4-1, j]), 3), alpha_single=round(float(als[d*4-1, j]), 3)) for d in (3, 5, 7, 10, 15)}
axs.flat[0].legend(fontsize=8)
plt.suptitle('Mechanism test: RMSE relative to the single model. Optimal shrinkage toward climatology (fitted out-of-sample) vs the learned cascade')
plt.tight_layout(); plt.savefig(f'{IM}/fig15_damping_control.png', dpi=105)
json.dump(res, open('/workspace/outputs/damping_results.json', 'w'), indent=1)
for v in V8: print(v, res[v]['d7'], res[v]['d10'], res[v]['d15'])
