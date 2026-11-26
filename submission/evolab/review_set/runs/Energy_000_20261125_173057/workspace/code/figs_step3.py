"""Figures for Step 3 (synthetic MMGA benchmark), GA convergence, identifiability, cost comparison."""
import sys, os, json, pickle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from style import *
from ecat_model import CORE, unit_to_phys, phys_to_unit, features
from step3_identify import objective

R = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); O = os.path.join(R, 'outputs'); I = os.path.join(R, 'report', 'images')
S = json.load(open(os.path.join(O, 'step3_synthetic.json'))); CV = pickle.load(open(os.path.join(O, 'step3_curves.pkl'), 'rb'))
B = json.load(open(os.path.join(O, 'step3_physga_baseline.json')))
mm = pickle.load(open(os.path.join(O, 'ann_core.pkl'), 'rb'))['mm']
LAB = {'R_n': 'R_n', 'R_p': 'R_p', 'k_n': 'k_n', 'h': 'h'}

# GA convergence
fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
for c, s in enumerate(S):
    for k, (hb, hm) in enumerate(zip(s['hist_best'], s['hist_median'])):
        ax[0].semilogy(hb, color=C[c], lw=1.6 if k == 0 else 0.6, alpha=1 if k == 0 else 0.5, label=f'case {c}' if k == 0 else None)
        if k == 0: ax[0].semilogy(hm, color=C[c], lw=1.0, ls=':')
ax[0].set(xlabel='generation', ylabel='objective J (ANN)', title='(a) MMGA convergence (solid: best, dotted: median; 5 seeds)')
ax[0].legend(frameon=False, ncol=2)
for b in B:
    pg = b['physGA']; ax[1].semilogy(pg['hist_neval'], np.minimum.accumulate(pg['hist_best']), color=C[b['case']], lw=2, label=f'physics-GA, case {b["case"]}')
    ax[1].axhline(b['MMGA']['J_phys'], color=C[b['case']], ls='--', lw=1.4, label=f'MMGA (physics-verified), case {b["case"]}')
ax[1].set(xlabel='number of physics simulations', ylabel='best objective J (physics)', title='(b) physics-only GA vs MMGA')
ax[1].legend(frameon=False)
for a in ax: clean(a)
fig.tight_layout(); fig.savefig(os.path.join(I, 'fig06_ga_convergence.png'), dpi=150); plt.close(fig)

# true vs identified (relative error bars + seed spread)
fig, ax = plt.subplots(1, 2, figsize=(13, 4.2))
w = 0.16
for c, s in enumerate(S):
    e = [s['rel_err_pct'][n] for n in CORE]
    ax[0].bar(np.arange(4) + (c - 2) * w, e, w * 0.9, color=C[c], label=f'case {c}')
    for j, n in enumerate(CORE):
        ax[0].scatter(np.full(5, j + (c - 2) * w), s['seed_rel_err_pct'][n], s=8, color='#0b0b0b', zorder=3, lw=0)
ax[0].axhline(0, color='#52514e', lw=0.8)
ax[0].set_xticks(range(4)); ax[0].set_xticklabels(['R_n', 'R_p', 'k_n', 'h'])
ax[0].set(ylabel='relative error [%]', title='(a) identification error (bars: verified best; dots: per-seed GA best)')
ax[0].legend(frameon=False, ncol=5, fontsize=8, loc='upper center', bbox_to_anchor=(0.5, -0.1))
JA = np.concatenate([s['J_ann_topk'] for s in S]); JP = np.concatenate([s['J_phys_topk'] for s in S])
ax[1].scatter(JA, JP, s=14, color=C[0], lw=0)
lo, hi = min(JA.min(), JP.min()) * 0.9, max(JA.max(), JP.max()) * 1.1
ax[1].plot([lo, hi], [lo, hi], color='#52514e', lw=1); ax[1].set_xscale('log'); ax[1].set_yscale('log')
ax[1].set(xlabel='J predicted by ANN', ylabel='J re-simulated by PyBaMM', title='(b) top-10 candidates: ANN vs physics objective')
for a in ax: clean(a)
fig.tight_layout(); fig.savefig(os.path.join(I, 'fig07_identification_errors.png'), dpi=150); plt.close(fig)

# overlays
fig, ax = plt.subplots(2, 5, figsize=(20, 7))
for c in range(len(S)):
    cv = CV[c]; tn, Vn, Tn = cv['noisy']; tt, Vt, Tt = cv['truth']; ti, Vi, Ti = cv['ident']
    ax[0, c].plot(tn[::10] / 60, Vn[::10], '.', color='#8a8984', ms=2, label='synthetic data (noisy)')
    ax[0, c].plot(ti / 60, Vi, color=C[c], lw=1.8, label='MMGA identified')
    ax[1, c].plot(tn[::10] / 60, Tn[::10] - 273.15, '.', color='#8a8984', ms=2)
    ax[1, c].plot(ti / 60, Ti - 273.15, color=C[c], lw=1.8)
    m = S[c]['curve']
    ax[0, c].set_title(f'case {c}: RMSE_V {m["rmse_V_mV"]:.2f} mV'); ax[1, c].set_title(f'RMSE_T {m["rmse_T_K"]:.3f} K')
    ax[1, c].set_xlabel('time [min]')
    if c == 0: ax[0, c].set_ylabel('voltage [V]'); ax[1, c].set_ylabel('temperature [°C]'); ax[0, c].legend(frameon=False, fontsize=8)
for a in ax.flat: clean(a)
fig.tight_layout(); fig.savefig(os.path.join(I, 'fig08_synthetic_overlays.png'), dpi=150); plt.close(fig)

# identifiability: 1-D objective profiles on the ANN around the truth (case 0), V-only vs T-only contributions
s = S[0]; u0 = np.array(s['u_true'])
tg = mm.predict(u0[None]); tgt = (tg[0][0], tg[1][0], tg[2][0])
fig, ax = plt.subplots(1, 4, figsize=(18, 3.8))
g = np.linspace(0, 1, 201)
for j, n in enumerate(CORE):
    U = np.tile(u0, (len(g), 1)); U[:, j] = g
    V, T, te = mm.predict(U)
    JV = np.mean((V - tgt[0]) ** 2, -1) / 0.01 ** 2; JT = np.mean((T - tgt[1]) ** 2, -1) / 0.5 ** 2; JQ = ((te - tgt[2]) / tgt[2]) ** 2 / 0.005 ** 2
    x = unit_to_phys(U, CORE)[:, j] * (1e6 if j < 2 else 1)
    ax[j].semilogy(x, JV + 1e-6, color=C[0], lw=2, label='voltage term'); ax[j].semilogy(x, JT + 1e-6, color=C[1], lw=2, label='temperature term')
    ax[j].semilogy(x, JQ + 1e-6, color=C[2], lw=1.5, label='capacity term')
    ax[j].axvline(x[np.argmin(np.abs(g - u0[j]))], color='#52514e', lw=0.8, ls='--'); ax[j].set_xscale('log'); ax[j].xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter()); ax[j].xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%g')); ax[j].set_xticks({0:[2.5,4,6,10],1:[1.5,2,3,4,6],2:[0.2,0.5,1,2,5],3:[5,10,20,40]}[j])
    ax[j].set(xlabel=['R_n [µm]', 'R_p [µm]', 'k_n multiplier', 'h [W m⁻² K⁻¹]'][j], ylim=(1e-4, 1e3))
    clean(ax[j])
ax[0].set_ylabel('objective contribution (others at truth)'); ax[0].legend(frameon=False, fontsize=8)
fig.suptitle('One-at-a-time objective profiles (ANN, nominal case): sensitivity of each signal to each parameter')
fig.tight_layout(); fig.savefig(os.path.join(I, 'fig09_identifiability_profiles.png'), dpi=150); plt.close(fig)
print('ok')
