"""Workflow schematic and an illustration of MD Tg extraction (bilinear density-temperature fit).
NOTE: no MD trajectories are provided with the task; panel (b) uses a synthetic illustrative curve whose
Tg is set to the median vitrimer MD Tg, purely to document the protocol."""
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from scipy.optimize import minimize
plt.rcParams.update({'font.size': 9})
fig = plt.figure(figsize=(15, 4.6))
ax = fig.add_axes([0.0, 0.0, 0.70, 1.0]); ax.axis('off'); ax.set_xlim(0, 10); ax.set_ylim(0, 4.6)
boxes = [
    (0.1, 2.6, 'MD simulations\n(provided data)', '295 polymers: Tg_MD, Tg_exp\n8,424 acid+epoxide vitrimers:\nTg_MD +/- std', '#dbe9f6'),
    (2.1, 2.6, 'GP calibration', 'dTg = Tg_exp - Tg_MD\nTanimoto(FP) + RBF(MD)\n+ RBF(descriptors), LOOCV\n-> calibrated Tg +/- sigma', '#fde2d7'),
    (4.1, 2.6, 'Dual-encoder graph VAE', 'acid GNN encoder -> z_a\nepoxide GNN encoder -> z_e\n2 SELFIES GRU decoders\n+ Tg head on [z_a, z_e]', '#dff0e6'),
    (6.1, 2.6, 'Latent-space BO', 'latent GP surrogate\nEI (max/min), P(|Tg-T*|<5K)\ntrust region, 8 rounds', '#ece4f2'),
    (8.1, 2.6, 'Candidates', 'decode -> FG filter\n-> re-encode -> virtual MD\n+ GP calibration', '#fff3c4'),
    (2.1, 0.3, '10^6 combinatorial space', 'all 7.7k acids x 7.7k epoxides\nencoded once; latent-GP screen', '#dff0e6'),
    (5.1, 0.3, 'In-silico validation', 'epoxy-resin hold-out (real Tg_exp)\nlatent traversal, RT design +\nchemist edits, recyclability', '#fde2d7'),
]
for x, y, t, s, c in boxes:
    ax.add_patch(FancyBboxPatch((x, y), 1.75, 1.75, boxstyle='round,pad=0.05', fc=c, ec='#444444'))
    ax.text(x + 0.875, y + 1.5, t, ha='center', va='center', fontweight='bold', fontsize=9)
    ax.text(x + 0.875, y + 0.7, s, ha='center', va='center', fontsize=7.4)
for x0 in [1.9, 3.9, 5.9, 7.9]:
    ax.annotate('', xy=(x0 + 0.2, 3.47), xytext=(x0 - 0.02, 3.47), arrowprops=dict(arrowstyle='->', lw=1.5))
ax.annotate('', xy=(3.0, 2.1), xytext=(4.9, 2.55), arrowprops=dict(arrowstyle='->', lw=1.2, color='#555555'))
ax.annotate('', xy=(6.0, 2.1), xytext=(9.0, 2.55), arrowprops=dict(arrowstyle='->', lw=1.2, color='#555555'))
ax.annotate('', xy=(6.5, 2.55), xytext=(3.95, 2.0), arrowprops=dict(arrowstyle='->', lw=1.0, color='#999999', ls='--'))
ax.text(5.0, 4.5, '(a) AI-guided inverse-design workflow', ha='center', fontsize=11, fontweight='bold')
# (b) synthetic bilinear illustration
ax2 = fig.add_axes([0.75, 0.15, 0.23, 0.72])
rng = np.random.default_rng(0)
T = np.arange(200, 651, 25.0); Tg0 = float(np.median(pd.read_csv('data/tg_vitrimer_MD.csv').tg))
rho = np.where(T < Tg0, 1.25 - 2.2e-4 * (T - Tg0), 1.25 - 5.8e-4 * (T - Tg0)) + rng.normal(0, 0.003, len(T))


def sse(p):
    tg, a, b1, b2 = p
    pr = np.where(T < tg, a + b1 * (T - tg), a + b2 * (T - tg)); return ((pr - rho) ** 2).sum()


best = min((minimize(sse, [tg, 1.25, -2e-4, -6e-4], method='Nelder-Mead', options=dict(maxiter=4000)) for tg in range(300, 560, 20)), key=lambda r: r.fun)
tg, a, b1, b2 = best.x
ax2.plot(T, rho, 'o', c='#2a6f97', ms=4, label='MD density (illustrative)')
tt = np.linspace(200, 650, 200)
ax2.plot(tt[tt < tg + 80], a + b1 * (tt[tt < tg + 80] - tg), 'k-', lw=1)
ax2.plot(tt[tt > tg - 80], a + b2 * (tt[tt > tg - 80] - tg), 'k-', lw=1)
ax2.axvline(tg, c='#e07a5f', ls='--'); ax2.text(tg + 8, rho.max(), 'Tg = %.0f K' % tg, color='#e07a5f')
ax2.set_xlabel('Temperature (K)'); ax2.set_ylabel('density (g/cm3)')
ax2.set_title('(b) Tg extraction: bilinear fit\n(synthetic illustration; trajectories not provided)', fontsize=9)
ax2.legend(fontsize=7, loc='lower left')
plt.savefig('report/images/fig0_workflow.png', dpi=140, bbox_inches='tight'); plt.close()
print('Tg fit', tg)
