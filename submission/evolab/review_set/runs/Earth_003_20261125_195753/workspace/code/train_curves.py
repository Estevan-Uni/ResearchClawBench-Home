"""Training-loss curves of all stages (fig04)."""
import json, numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
CK = '/workspace/outputs/ckpt'
fig, axs = plt.subplots(1, 2, figsize=(14, 4.2))
off = 0
for st, c in [('pre', 'k'), ('short', 'C0'), ('medium', 'C3'), ('long', 'C2'), ('shortplus', 'C4')]:
    try: L = json.load(open(f'{CK}/{st}_log.json'))
    except Exception: continue
    it = np.array([r['it'] for r in L]); lo = np.array([r['loss'] for r in L])
    x0 = {'pre': 0, 'short': 2800, 'medium': 3450, 'long': 3850, 'shortplus': 3450}[st]
    axs[0].plot(x0 + it, lo, color=c, label=st)
    v = [(r['it'], r['val_rmse']) for r in L if 'val_rmse' in r]
    if v: axs[1].plot(x0 + np.array(v)[:, 0], np.array(v)[:, 1], 'o-', color=c, label=st)
for a in axs: a.set_xlabel('cumulative iteration (stage offset)'); a.grid(alpha=.3); a.legend()
axs[0].set_ylabel('training loss (lat-weighted L1, normalised, mean over K steps)'); axs[1].set_ylabel('validation RMSE at last rollout step (normalised)')
axs[0].set_title('Training loss; jumps = curriculum K increase / new input distribution'); axs[1].set_title('Held-out validation (Medium/Long: on generated states)')
plt.tight_layout(); plt.savefig('/workspace/report/images/fig04_training.png', dpi=100)
