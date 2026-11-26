import json
import numpy as np
from plotstyle import plt, C, style
style()
b = json.load(open('../outputs/results/bench_scaling.json'))
fig, ax = plt.subplots(figsize=(4.6, 3.6))
for les, col, mk, lab in [(0, C['orange'], 's', 'SR'), (1, C['blue'], 'o', 'LES (direct k-sum)')]:
    r = [x for x in b if x['les'] == les]
    ax.plot([x['N'] for x in r], [x['t'] for x in r], marker=mk, color=col, ms=8, label=lab)
N = np.array([128, 3456])
ax.plot(N, 0.0157 * N / 128, color=C['muted'], lw=1, ls='--'); ax.text(1500, 0.14, '∝ N', color=C['ink2'], fontsize=8)
ax.plot(N, 0.038 * (N / 128) ** 2, color=C['muted'], lw=1, ls=':'); ax.text(1100, 4.5, '∝ N²', color=C['ink2'], fontsize=8)
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xticks([128, 1024, 3456]); ax.set_xticklabels(['128', '1024', '3456']); ax.minorticks_off()
ax.set_xlabel('number of atoms N'); ax.set_ylabel('wall time, E + F (s, 1 CPU core)')
ax.set_title('Evaluation cost vs system size'); ax.legend()
plt.tight_layout(); plt.savefig('../report/images/fig5_scaling.png')
