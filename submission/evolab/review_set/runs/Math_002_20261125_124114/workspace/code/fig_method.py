import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
fig, ax = plt.subplots(figsize=(11, 3.6)); ax.axis('off'); ax.set_xlim(0, 11); ax.set_ylim(-0.4, 3.6)
def box(x, y, w, h, t, c):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.05', fc=c, ec='k', lw=0.8))
    ax.text(x + w / 2, y + h / 2, t, ha='center', va='center', fontsize=7.5)
def arr(x1, y1, x2, y2, t=''):
    ax.annotate('', (x2, y2), (x1, y1), arrowprops=dict(arrowstyle='->', lw=0.9))
    if t: ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.08, t, fontsize=6.5, ha='center')
box(0.1, 1.4, 1.5, 0.9, 'Initial plan\nsoft PP (exact\nspace-time DP)', '#e8f0fb')
box(2.0, 1.4, 1.6, 0.9, 'ALNS neighbourhood\n(collision-based /\nrandom, N=8)', '#e8f0fb')
box(4.1, 2.35, 2.5, 1.05, 'MARL repair (early)\nshared policy, 26 per-action feats:\nbackground cost-to-go DP, lookahead,\nteam claims + intention messages\nW steps jointly, then soft DP completion', '#fde8e8')
box(4.1, 0.25, 2.5, 0.95, 'PP repair (late)\nrandom order, each agent:\nexact min-(collisions, time) DP\nvs. background + earlier agents', '#e8f5e9')
box(7.1, 1.4, 1.6, 0.9, 'Accept iff CP does not\nincrease (tie: SoC\nnot increased)', '#f3f3f3')
box(9.2, 1.4, 1.6, 0.9, 'Switching controller\nrate rule: MARL vs PP\nprobe CP-reduction/s', '#fff3e0')
arr(1.6, 1.85, 2.0, 1.85); arr(3.6, 1.95, 4.1, 2.8, 'before switch'); arr(3.6, 1.75, 4.1, 0.75, 'after switch')
arr(6.6, 2.8, 7.1, 1.95); arr(6.6, 0.75, 7.1, 1.75); arr(8.7, 1.85, 9.2, 1.85)
ax.plot([10.0, 10.0, 2.8], [1.35, -0.15, -0.15], color='k', lw=0.9); ax.annotate('', (2.8, 1.35), (2.8, -0.16), arrowprops=dict(arrowstyle='->', lw=0.9))
ax.text(8.4, -0.33, 'repeat until CP = 0 or CPU-time budget exhausted', fontsize=7, ha='center')
plt.tight_layout(); plt.savefig('../report/images/method.png', dpi=150)
