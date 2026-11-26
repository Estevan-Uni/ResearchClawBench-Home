import os, sys, json, numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(__file__))
from data_io import *
IMG = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'report', 'images')
cf = list_configs()
fig, axes = plt.subplots(1, 7, figsize=(17, 2.9))
for ax, fam in zip(axes, FAMILIES):
    c = [x for x in cf if x[0] == fam][0]
    g, s, t = load_instance(instance_files(c[3])[0])
    ax.imshow(g, cmap='Greys', vmin=0, vmax=1.4)
    ax.scatter(s[:, 1], s[:, 0], s=4 if g.shape[0] > 20 else 25, c='#1f77b4', label='start')
    ax.scatter(t[:, 1], t[:, 0], s=4 if g.shape[0] > 20 else 25, c='#d62728', marker='x', lw=0.6, label='goal')
    ax.set_title(f'{fam}\n{g.shape[0]}x{g.shape[1]}, {c[2]} agents', fontsize=9); ax.set_xticks([]); ax.set_yticks([])
axes[0].legend(fontsize=6, loc='lower left')
plt.tight_layout(); plt.savefig(os.path.join(IMG, 'maps_overview.png'), dpi=140); plt.close()
D = json.load(open(os.path.join(os.path.dirname(__file__), '..', 'outputs', 'data_summary.json')))['configs']
fig, ax = plt.subplots(figsize=(9, 3))
cols = dict(zip(FAMILIES, plt.cm.tab10(np.arange(7))))
for i, r in enumerate(D):
    ax.bar(i, r['density_mean'], color=cols[r['family']], yerr=[[max(0, r['density_mean'] - r['density_min'])], [max(0, r['density_max'] - r['density_mean'])]], capsize=2)
ax.set_xticks(range(len(D))); ax.set_xticklabels([f"{r['family'][:6]}-{r['agents']}" for r in D], rotation=75, fontsize=7)
ax.set_ylabel('agents / free cells'); ax.set_title('Agent density of the 28 benchmark configurations (bar: mean, whisker: min-max)', fontsize=9)
plt.tight_layout(); plt.savefig(os.path.join(IMG, 'density.png'), dpi=140); plt.close()
print('ok')
