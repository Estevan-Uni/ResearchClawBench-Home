"""Validity figure: blocked CV per fold, leakage curve vs block length, circular-shift null."""
import json, numpy as np, pandas as pd
from common import *; from style import *
B = pd.read_csv(f'{OUT}/blocked_cv_metrics.csv'); L = pd.read_csv(f'{OUT}/leakage_curve.csv'); N = pd.read_csv(f'{OUT}/circular_shift_null.csv')
fig, axs = plt.subplots(1, 3, figsize=(14, 4))
for b in BEHAVIOURS:
    d = L.groupby(['behaviour', 'block_len']).AP.agg(['mean', 'std']).loc[b]
    axs[0].errorbar(d.index, d['mean'], yerr=d['std'], marker='o', ms=7, color=BEH_COL[b], capsize=3, label=b)
    prev = T_prev = {'Attack': 0.338, 'Sniffing': 0.133}[b]; axs[0].axhline(prev, color=BEH_COL[b], ls=':', lw=1)
axs[0].set_xscale('log'); axs[0].set_xticks([1, 6, 30, 120, 348], ['1', '6', '30', '120', '348']); axs[0].set_xlabel('contiguous test-block length L (frames)')
axs[0].set_ylabel('held-out AP'); axs[0].set_title('A. Leakage curve (engineered, 100 trees, 3 seeds)\ndotted = prevalence'); axs[0].legend()
e = B[(B.featureset == 'engineered') & (B.fold != 'pooled')].copy(); e['fold'] = e.fold.astype(int)
for k, b in enumerate(BEHAVIOURS):
    d = e[e.behaviour == b]; x = d.fold + (k - 0.5) * 0.36
    axs[1].bar(x, d.AP.fillna(0), 0.34, color=BEH_COL[b], label=f'{b} AP')
    axs[1].scatter(x, d.prevalence, marker='_', s=200, color=INK, zorder=3, label='fold prevalence' if k == 0 else None)
    for xi, (ap, npos) in zip(x, zip(d.AP, d.n_pos_test)):
        if npos == 0: axs[1].text(xi, 0.02, 'no\npos', ha='center', fontsize=7, color=INK2)
axs[1].set_xticks(range(5), [f'fold {i}\n{int(s)}–{int(t)}' for i, s, t in zip(e[e.behaviour == "Attack"].fold, e[e.behaviour == "Attack"].start, e[e.behaviour == "Attack"].end)], fontsize=7)
pool = B[(B.featureset == 'engineered') & (B.fold == 'pooled')].set_index('behaviour')
axs[1].set_title(f'B. Blocked 5-fold CV, 15-frame purge\npooled AP: Attack {pool.loc["Attack","AP"]:.2f}, Sniffing {pool.loc["Sniffing","AP"]:.2f}'); axs[1].legend(fontsize=7); axs[1].set_ylim(0, 1)
for k, b in enumerate(BEHAVIOURS):
    d = N[N.behaviour == b]; nul = d[~d.is_real]; real = d[d.is_real].iloc[0]
    for j, col in enumerate(['AP_random_split', 'AP_blocked']):
        xx = j * 2.5 + k; axs[2].scatter(np.full(len(nul), xx) + np.random.default_rng(0).uniform(-0.15, 0.15, len(nul)), nul[col], s=14, color=BEH_COL[b], alpha=0.5)
        axs[2].scatter([xx], [real[col]], marker='D', s=60, color=INK, zorder=3)
        p = (1 + (nul[col] >= real[col]).sum()) / (1 + len(nul)); axs[2].text(xx, 0.95, f'p={p:.2f}', ha='center', fontsize=8)
axs[2].set_xticks([0, 1, 2.5, 3.5], ['Attack\nrandom split', 'Sniffing\nrandom split', 'Attack\nblocked CV', 'Sniffing\nblocked CV'], fontsize=8)
axs[2].set_ylabel('AP'); axs[2].set_ylim(0, 1.02); axs[2].set_title('C. Circular-shift label null (15 shifts; ◆ = real labels)')
plt.tight_layout(); plt.savefig(f'{IMG}/fig14_validity_leakage.png', dpi=150); plt.close()
out = {}
for b in BEHAVIOURS:
    d = N[N.behaviour == b]; nul = d[~d.is_real]; real = d[d.is_real].iloc[0]
    out[b] = {c: dict(real=float(real[c]), null_mean=float(nul[c].mean()), null_sd=float(nul[c].std()), p=float((1 + (nul[c] >= real[c]).sum()) / (1 + len(nul)))) for c in ['AP_random_split', 'AP_blocked']}
json.dump(out, open(f'{OUT}/circular_shift_null_summary.json', 'w'), indent=1); print(out)
