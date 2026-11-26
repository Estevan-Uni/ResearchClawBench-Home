import pandas as pd, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt, seaborn as sns
d = pd.read_csv('/workspace/outputs/rpe_noise.csv')
m0 = pd.read_csv('/workspace/outputs/rpe_metrics.csv'); m0 = m0[(m0.k == 30) | (m0.method == 'all_features')]
meths = ['DELVE-faithful', '+dedup', '+whiten', 'variance', 'dispersion', 'laplacian', 'mcfs', 'hotspot_moran', 'random', 'all_features']
cols = dict(zip(meths, sns.color_palette('tab10', len(meths))))
fig, ax = plt.subplots(2, 2, figsize=(13, 9))
for j, (kind, lvls, xl) in enumerate([('gauss', [0, .5, 1, 2], 'Gaussian noise sd (× feature sd)'), ('dropout', [0, .2, .5], 'fraction of entries set to 0')]):
    for i, met in enumerate(['dpt_rho_all', 'acc']):
        a = ax[i, j]
        for m in meths:
            base = m0[m0.method == m][met].values[0]
            s = d[(d.kind == kind) & (d.method == m)].sort_values('level')
            a.plot([0] + list(s.level), [base] + list(s[met]), marker='o', color=cols[m], lw=3 if m in meths[:3] else 1.3, ls='--' if m == 'DELVE-faithful' else '-', label=m)
        a.set(xlabel=xl, ylabel=met, title=f'RPE corruption: {met} at k=30')
ax[0, 1].legend(fontsize=8, bbox_to_anchor=(1.02, 1), loc='upper left')
plt.tight_layout(); plt.savefig('/workspace/report/images/fig9_rpe_noise.png', dpi=110, bbox_inches='tight')
