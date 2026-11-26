"""Figure + summary table for the Step-3b identifiability ablation."""
import os, json, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
import sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from style import *
R = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); O = os.path.join(R, 'outputs')
A = json.load(open(os.path.join(O, 'step3b_ablation.json'))); P = ['R_n', 'R_p', 'k_n', 'h']
cfgs = list(dict.fromkeys(a['config'] for a in A)); summ = {}
for c in cfgs:
    E = np.array([[abs(a['rel_err_pct'][p]) for p in P] for a in A if a['config'] == c])
    summ[c] = dict(mean_abs_err_pct=dict(zip(P, E.mean(0).tolist())), max_abs_err_pct=dict(zip(P, E.max(0).tolist())))
json.dump(summ, open(os.path.join(O, 'step3b_summary.json'), 'w'), indent=1)
lab = {'VT_noise2mV': 'V+T, σ=2 mV/0.1 K', 'VT_noise5mV': 'V+T, σ=5 mV/0.3 K', 'VT_noise10mV': 'V+T, σ=10 mV/0.5 K', 'Vonly_noise2mV': 'V only, σ=2 mV'}
fig, ax = plt.subplots(figsize=(9, 4.2)); w = 0.2
for k, c in enumerate(cfgs):
    m = [summ[c]['mean_abs_err_pct'][p] for p in P]; mx = [summ[c]['max_abs_err_pct'][p] for p in P]
    ax.bar(np.arange(4) + (k - 1.5) * w, m, w * 0.9, color=C[k], label=lab[c])
    ax.scatter(np.arange(4) + (k - 1.5) * w, mx, marker='_', s=120, color='#0b0b0b', zorder=3)
ax.set_xticks(range(4)); ax.set_xticklabels(P); ax.set_yscale('log')
ax.set(ylabel='|relative error| [%] (bar: mean of 5 cases, tick: max)', title='Identifiability ablation: noise level and temperature information')
ax.legend(frameon=False, fontsize=8); clean(ax)
fig.tight_layout(); fig.savefig(os.path.join(R, 'report', 'images', 'fig10_ablation.png'), dpi=150)
print(json.dumps(summ, indent=1))
