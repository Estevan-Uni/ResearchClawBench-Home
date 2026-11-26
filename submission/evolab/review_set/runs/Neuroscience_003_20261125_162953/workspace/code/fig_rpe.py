import sys; sys.path.insert(0, '/workspace/code')
import numpy as np, pandas as pd, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, seaborn as sns
df = pd.read_csv('/workspace/outputs/rpe_metrics.csv')
allf = df[df.method == 'all_features'].iloc[0]
d = df[df.method != 'all_features']
delve_m = ['DELVE-faithful', '+noise-filter', '+dedup', '+whiten']
base_m = [m for m in d.method.unique() if m not in delve_m]
mets = [('acc', 'phase kNN accuracy'), ('nmi', 'k-means NMI (phase)'), ('dpt_rho_all', 'DPT ρ (all cells)'), ('dpt_rho_cyc', 'DPT ρ (cycling cells)'), ('batch_bal_acc', 'batch kNN bal. acc (0.5=mixed)')]
fig, ax = plt.subplots(1, 5, figsize=(24, 4.8))
cols = dict(zip(d.method.unique(), sns.color_palette('tab20', d.method.nunique())))
for a, (m, lab) in zip(ax, mets):
    for meth in d.method.unique():
        s = d[d.method == meth].sort_values('k')
        a.plot(s.k, s[m], marker='o', lw=2.5 if meth in delve_m else 1, ls='-' if meth in delve_m else '--', color=cols[meth], label=meth, alpha=1 if meth in delve_m or meth.startswith('oracle') else .7)
    a.axhline(allf[m], color='k', ls=':', label='all 241 features'); a.set(xlabel='panel size k', title=lab); a.axvline(30, color='0.8', lw=.8)
ax[-1].legend(fontsize=7, bbox_to_anchor=(1.02, 1), loc='upper left')
plt.tight_layout(); plt.savefig('/workspace/report/images/fig4_rpe_kcurves.png', dpi=110, bbox_inches='tight')
# k=30 table bars
t = d[d.k == 30].set_index('method')
fig, ax = plt.subplots(1, 3, figsize=(16, 5.5))
for a, m in zip(ax, ['acc', 'nmi', 'dpt_rho_all']):
    s = t[m].sort_values()
    a.barh(s.index, s.values, color=['C3' if i in delve_m else ('0.5' if i.startswith('oracle') else 'C0') for i in s.index])
    a.axvline(allf[m], color='k', ls=':'); a.set_title(dict(mets)[m] + ' @ k=30')
plt.tight_layout(); plt.savefig('/workspace/report/images/fig5_rpe_k30_bars.png', dpi=110)
# bootstrap
b = pd.read_csv('/workspace/outputs/rpe_bootstrap_k30.csv')
rows = []
for c in b.columns:
    m, s = c.split('|'); rows.append(dict(method=m, cells=s, mean=b[c].mean(), lo=b[c].quantile(.025), hi=b[c].quantile(.975)))
ci = pd.DataFrame(rows); ci.to_csv('/workspace/outputs/rpe_bootstrap_ci_k30.csv', index=False)
# paired differences: DELVE variants vs best label-free baseline (by all-cell DPT rho at k=30)
lf = [m for m in base_m if not m.startswith('oracle') and m != 'random']
best = t.loc[lf, 'dpt_rho_all'].idxmax()
pr = []
for m in delve_m:
    for s in ['all', 'cyc']:
        dd = b[f'{m}|{s}'] - b[f'{best}|{s}']
        pr.append(dict(method=m, vs=best, cells=s, diff=dd.mean(), lo=dd.quantile(.025), hi=dd.quantile(.975), p_le0=(dd <= 0).mean()))
pd.DataFrame(pr).to_csv('/workspace/outputs/rpe_paired_diff_k30.csv', index=False)
print(ci.round(3).to_string()); print(pd.DataFrame(pr).round(3).to_string())
