import numpy as np, pandas as pd, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, seaborn as sns
df = pd.read_csv('/workspace/outputs/sim_results.csv')
clean = df[df.factor == 'clean']
tab = clean.groupby(['topology', 'method'])[['precision', 'acc', 'nmi', 'dpt_rho', 'conf_frac']].mean().round(3)
tab.to_csv('/workspace/outputs/sim_clean_table.csv'); print(tab.to_string())
meths = ['DELVE-faithful', '+noise-filter', 'laplacian', 'mcfs', 'hotspot_moran', 'nbr_variance', 'cluster_F', 'diffusion_corr', 'pca_loading', 'variance', 'dispersion', 'random', 'oracle_rho']
cols = dict(zip(meths, sns.color_palette('tab20', len(meths))))
# clean figure
fig, ax = plt.subplots(1, 4, figsize=(22, 5))
for a, m in zip(ax, ['precision', 'acc', 'nmi', 'dpt_rho']):
    p = clean[clean.method.isin(meths + ['all_features'])].groupby(['method', 'topology'])[m].mean().unstack()
    p = p.loc[[x for x in meths + ['all_features'] if x in p.index]]
    p.plot.barh(ax=a, width=.8); a.set_title(f'{m} (clean, k=100)'); a.invert_yaxis()
    if m != 'dpt_rho': a.get_legend().remove()
plt.tight_layout(); plt.savefig('/workspace/report/images/fig6_sim_clean.png', dpi=110)
# noise sweeps: clean is level 0 of each factor
rows = []
for f in ['snr', 'library', 'dropout']:
    s = pd.concat([clean.assign(factor=f), df[df.factor == f]]); rows.append(s)
sw = pd.concat(rows)
sw.groupby(['factor', 'level', 'method'])[['precision', 'acc', 'nmi', 'dpt_rho']].mean().round(3).to_csv('/workspace/outputs/sim_sweep_table.csv')
lv = {'snr': ['1.0', '0.6', '0.4', '0.25'], 'library': ['1.0', '0.3', '0.1', '0.03'], 'dropout': ['none', 'm=0', 'm=1', 'm=2']}
fig, ax = plt.subplots(4, 3, figsize=(17, 17))
for j, f in enumerate(['snr', 'library', 'dropout']):
    for i, m in enumerate(['precision', 'acc', 'nmi', 'dpt_rho']):
        a = ax[i, j]
        g = sw[sw.factor == f].groupby(['method', 'level'])[m].mean().unstack(0)
        for meth in meths:
            if meth in g:
                a.plot(g.index, g[meth], marker='o', color=cols[meth], lw=3 if 'DELVE' in meth or 'noise' in meth else 1.2, ls='-' if meth not in ('random', 'oracle_rho') else ':', label=meth)
        a.set_xticks(range(4)); a.set_xticklabels(lv[f]); a.set_title(f'{m} vs {f} (mean of topologies × reps)')
        if f == 'library': a.set_xlabel('library-size scale')
        if f == 'snr': a.set_xlabel('DE effect scale')
        if f == 'dropout': a.set_xlabel('dropout midpoint (higher = more zeros)')
ax[0, 2].legend(fontsize=8, bbox_to_anchor=(1.02, 1), loc='upper left')
plt.tight_layout(); plt.savefig('/workspace/report/images/fig7_sim_noise_sweeps.png', dpi=100, bbox_inches='tight')
# per-topology precision sweeps
fig, ax = plt.subplots(3, 3, figsize=(16, 13))
for i, topo in enumerate(['linear', 'bifurcating', 'tree']):
    for j, f in enumerate(['snr', 'library', 'dropout']):
        a = ax[i, j]; g = sw[(sw.factor == f) & (sw.topology == topo)].groupby(['method', 'level'])['dpt_rho'].mean().unstack(0)
        for meth in meths:
            if meth in g: a.plot(g.index, g[meth], marker='o', color=cols[meth], lw=3 if 'DELVE' in meth or 'noise' in meth else 1.2, label=meth)
        a.set_xticks(range(4)); a.set_xticklabels(lv[f]); a.set_title(f'{topo}: DPT ρ vs {f}')
ax[0, 2].legend(fontsize=8, bbox_to_anchor=(1.02, 1), loc='upper left')
plt.tight_layout(); plt.savefig('/workspace/report/images/fig8_sim_topology_dpt.png', dpi=100, bbox_inches='tight')
print('reps:', df.groupby(['topology', 'factor', 'level']).rep.nunique().min(), df.rep.max() + 1)
