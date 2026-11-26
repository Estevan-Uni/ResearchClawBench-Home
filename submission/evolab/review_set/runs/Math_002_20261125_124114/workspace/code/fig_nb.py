import os, json, numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
O = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'outputs'); IMG = os.path.join(O, '..', 'report', 'images')
d = pd.DataFrame([json.loads(l) for l in open(os.path.join(O, 'nb_study.jsonl'))])
ops = ['PP', 'MARL-learned-W8', 'MARL-learned-W16', 'MARL-learned-Wfull', 'MARL-bc-Wfull', 'MARL-heur-Wfull']
cols = ['#1f77b4', '#d62728', '#ff7f0e', '#e377c2', '#8c564b', '#7f7f7f']
fams = list(dict.fromkeys(d.family))
fig, axes = plt.subplots(1, 3, figsize=(15, 3.4))
for j, (metric, lab) in enumerate([('after', 'local CP after repair (lower better)'), ('p_improve', 'P(repair strictly reduces CP)'), ('ms', 'time per repair (ms, wall)')]):
    ax = axes[j]
    for si, stage in enumerate(['early', 'late']):
        if metric == 'p_improve' and stage == 'late': continue
        sub = d[d.stage == stage]
        x = np.arange(len(fams)) + (0 if stage == 'early' else 0.0)
        for k, op in enumerate(ops):
            v = [sub[(sub.family == f) & (sub.op == op)][metric].mean() for f in fams]
            if stage == 'early':
                ax.bar(x + (k - 2.5) * 0.13, v, 0.13, color=cols[k], label=op if j == 0 else None)
    if metric == 'after':
        b = [d[(d.stage == 'early') & (d.family == f)]['before'].mean() for f in fams]
        ax.scatter(np.arange(len(fams)), b, marker='_', s=300, color='k', label='before repair', zorder=5)
    if metric == 'ms': ax.set_yscale('log')
    ax.set_xticks(np.arange(len(fams))); ax.set_xticklabels(fams, rotation=30, fontsize=7); ax.set_title(lab + ' — early stage', fontsize=9)
axes[0].legend(fontsize=6, ncol=2)
plt.tight_layout(); plt.savefig(os.path.join(IMG, 'nb_study_early.png'), dpi=150); plt.close()
# early vs late summary
g = d.groupby(['stage', 'op'])[['before', 'after', 'p_improve', 'p_accept', 'ms']].mean().reset_index()
fig, ax = plt.subplots(figsize=(7, 3))
for si, stage in enumerate(['early', 'late']):
    s = g[g.stage == stage].set_index('op').reindex(ops)
    ax.bar(np.arange(len(ops)) + (si - 0.5) * 0.38, s['after'] / s['before'], 0.38, label=stage, color=['#4c72b0', '#dd8452'][si])
ax.axhline(1, color='k', lw=0.8); ax.set_xticks(range(len(ops))); ax.set_xticklabels(ops, rotation=20, fontsize=7)
ax.set_ylabel('local CP after / before'); ax.legend(fontsize=8); ax.set_title('Repair quality from identical states & neighbourhoods (mean over 7 configs)', fontsize=9)
plt.tight_layout(); plt.savefig(os.path.join(IMG, 'nb_study_stage.png'), dpi=150); plt.close()
g.to_csv(os.path.join(O, 'table_nb_study.csv'), index=False)
print(g.round(3).to_string())
