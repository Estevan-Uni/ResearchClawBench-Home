import json, os, numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
O = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'outputs')
L = json.load(open(os.path.join(O, 'train_log.json')))
bc = [x for x in L if x['phase'] == 'bc'][0]; pp = [x for x in L if x['phase'] == 'ppo']
fig, ax = plt.subplots(figsize=(6.5, 3.2))
t = np.array([x['t'] for x in pp]); e = np.array([x['eval'] for x in pp]); tr = np.array([x['train_local'] for x in pp])
k = 10; sm = np.convolve(tr, np.ones(k) / k, mode='valid')
ax.plot(t[k - 1:], sm, color='#ff7f0e', alpha=0.6, label='training local CP (sampled policy, moving avg)')
ax.plot(t, e, color='#d62728', label='held-out local CP (greedy policy)')
ax.scatter([bc['t']], [bc['eval']], color='k', zorder=5, label=f"after BC (acc {bc['acc']:.2f})")
ax.axhline(bc['pp'], color='#1f77b4', ls='--', label='PP repair on same held-out neighbourhoods')
ax.set_xlabel('training wall time (s)'); ax.set_ylabel('CP after repair (local)'); ax.legend(fontsize=7); ax.set_ylim(0, 30)
ax.set_title('MARL repair policy training (procedurally generated random maps only)', fontsize=9)
plt.tight_layout(); plt.savefig(os.path.join(O, '..', 'report', 'images', 'training.png'), dpi=150)
print(bc, min(e), len(pp))
