"""Figures: data overview, LHS simulations (Step 1), ANN loss curve / parity / error (Step 2)."""
import sys, os, json, pickle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from style import *
from ecat_model import CORE, TAU, unit_to_phys

R = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); O = os.path.join(R, 'outputs'); I = os.path.join(R, 'report', 'images')
E = pickle.load(open(os.path.join(O, 'exp_data.pkl'), 'rb'))

# ---------- data overview ----------
fig, ax = plt.subplots(2, 3, figsize=(15, 8))
for k, cell in enumerate(['B0005', 'B0006', 'B0007', 'B0018']):
    caps = [c['cap'] for c in E['NASA'][cell]]
    ax[0, 0].plot(caps, color=C[k], lw=1.6, label=cell)
cs = E['CS2_36']; ax[0, 0].set(xlabel='discharge cycle', ylabel='capacity [Ah]', title='(a) NASA PCoE capacity fade (2 A CC)')
ax[0, 0].legend(frameon=False)
for k, cyc in enumerate([0, 50, 100, 167]):
    c = E['NASA']['B0005'][cyc]
    ax[0, 1].plot(c['t'] / 60, c['V'], color=SEQ[k], lw=1.6, label=f'cycle {cyc}')
    ax[0, 2].plot(c['t'] / 60, c['T'], color=SEQ[k], lw=1.6, label=f'cycle {cyc}')
ax[0, 1].set(xlabel='time [min]', ylabel='voltage [V]', title='(b) NASA B0005 discharge voltage'); ax[0, 1].legend(frameon=False)
ax[0, 2].set(xlabel='time [min]', ylabel='surface temperature [°C]', title='(c) NASA B0005 temperature'); ax[0, 2].legend(frameon=False)
ax[1, 0].plot([c['cap'] for c in cs], color=C[0], lw=1.6)
ax[1, 0].set(xlabel='discharge cycle', ylabel='capacity [Ah]', title='(d) CALCE CS2_36 capacity (1.1 A CC)')
for k, cyc in enumerate([0, 50, 100, 150]):
    c = cs[cyc]; ax[1, 1].plot(c['t'] / 60, c['V'], color=SEQ[k], lw=1.6, label=f'cycle {cyc}')
ax[1, 1].set(xlabel='time [min]', ylabel='voltage [V]', title='(e) CS2_36 discharge voltage (no T logged)'); ax[1, 1].legend(frameon=False)
ox = E['Oxford']['dc']; a2 = ax[1, 2]
a2.plot((ox['t'] - ox['t'][0]) / 60, ox['V'], color=C[0], lw=1.0, label='voltage [V]')
a2.set(xlabel='time [min]', ylabel='voltage [V]', title='(f) Oxford Artemis urban drive cycle (40 °C)')
ins = a2.inset_axes([0.45, 0.55, 0.52, 0.4]); ins.plot((ox['t'] - ox['t'][0]) / 60, ox['I'], color=C[1], lw=0.6)
ins.set_ylabel('I [A]', fontsize=8); ins.tick_params(labelsize=7)
for a in ax.flat: clean(a)
fig.tight_layout(); fig.savefig(os.path.join(I, 'fig01_data_overview.png'), dpi=150); plt.close(fig)

# ---------- Step 1: LHS curves ----------
d = np.load(os.path.join(O, 'db_baseline20.npz')); raw = pickle.load(open(os.path.join(O, 'db_baseline20_raw.pkl'), 'rb'))
D = np.load(os.path.join(O, 'db_core3000.npz'))
fig, ax = plt.subplots(1, 4, figsize=(18, 4.2))
hcol = plt.cm.Blues(0.35 + 0.6 * d['U'][:, 3])
for k, r in enumerate(raw):
    t, V, T = r
    ax[0].plot(t / 60, V, color=hcol[k], lw=1.2); ax[1].plot(t / 60, T - 273.15, color=hcol[k], lw=1.2)
ax[0].set(xlabel='time [min]', ylabel='voltage [V]', title='(a) 20 LHS samples: V(t), 1C')
ax[1].set(xlabel='time [min]', ylabel='cell temperature [°C]', title='(b) 20 LHS samples: T(t) (darker = larger h)')
sel = np.random.default_rng(0).choice(len(D['U']), 300, replace=False)
for i in sel:
    ax[2].plot(TAU, D['V'][i], color=C[0], lw=0.4, alpha=0.25); ax[3].plot(TAU, D['T'][i], color=C[1], lw=0.4, alpha=0.25)
ax[2].set(xlabel='normalised time τ = t / t_end', ylabel='voltage [V]', title='(c) training DB (300 of 3000): V(τ)')
ax[3].set(xlabel='normalised time τ = t / t_end', ylabel='ΔT [K]', title='(d) training DB: ΔT(τ)')
for a in ax: clean(a)
fig.tight_layout(); fig.savefig(os.path.join(I, 'fig02_lhs_simulations.png'), dpi=150); plt.close(fig)

# LHS design projection
X = D['X']
fig, ax = plt.subplots(1, 3, figsize=(13, 4))
lab = ['R_n [µm]', 'R_p [µm]', 'k_n multiplier [-]', 'h [W m⁻² K⁻¹]']; sc = [1e6, 1e6, 1, 1]
for k, (i, j) in enumerate([(0, 1), (2, 3), (0, 2)]):
    ax[k].scatter(X[:, i] * sc[i], X[:, j] * sc[j], s=4, color=C[0], alpha=0.5, lw=0)
    ax[k].set(xlabel=lab[i], ylabel=lab[j]); clean(ax[k])
    if i in (0, 1, 2): ax[k].set_xscale('log')
    if j in (0, 1, 2, 3): ax[k].set_yscale('log')
fig.suptitle('LHS design (3000 samples, log-uniform marginals)'); fig.tight_layout()
fig.savefig(os.path.join(I, 'fig03_lhs_design.png'), dpi=150); plt.close(fig)

# ---------- Step 2: ANN ----------
H = json.load(open(os.path.join(O, 'ann_core_history.json'))); M = json.load(open(os.path.join(O, 'ann_core_metrics.json')))
fig, ax = plt.subplots(1, 1, figsize=(8, 4.4)); ax = [ax]
h = H['full']; ax[0].semilogy(h['epoch'], h['train_mse'], color=C[0], lw=2, label=f'training (n={M["n_train"]})')
ax[0].semilogy(h['epoch'], h['val_mse'], color=C[1], lw=2, label=f'validation (n={M["n_val"]})')
hs = H['small']; ax[0].semilogy(hs['epoch'], hs['train_mse'], color=C[0], lw=1.2, ls='--', label='training, 20-sample DB')
ax[0].semilogy(hs['epoch'], hs['val_mse'], color=C[1], lw=1.2, ls='--', label='validation, 20-sample model')
ax[0].set(xlabel='epoch', ylabel='MSE (standardised outputs)', title='ANN learning curve (Adam, lr = 2e-3, batch 64)')
ax[0].legend(frameon=False); clean(ax[0])
fig.tight_layout(); fig.savefig(os.path.join(I, 'fig04_ann_loss.png'), dpi=150); plt.close(fig)

A = pickle.load(open(os.path.join(O, 'ann_core.pkl'), 'rb')); mm = A['mm']; ti = A['split']['test']
Vp, Tp, tp = mm.predict(D['U'][ti])
fig, ax = plt.subplots(1, 4, figsize=(18, 4.2))
ax[0].scatter(D['V'][ti].ravel(), Vp.ravel(), s=2, color=C[0], alpha=0.3, lw=0)
ax[1].scatter(D['T'][ti].ravel(), Tp.ravel(), s=2, color=C[1], alpha=0.3, lw=0)
ax[2].scatter(D['tend'][ti] / 60, tp / 60, s=8, color=C[2], lw=0)
for a, nm in zip(ax[:3], ['voltage [V]', 'ΔT [K]', 't_end [min]']):
    lo, hi = a.get_xlim(); a.plot([lo, hi], [lo, hi], color='#52514e', lw=1); a.set(xlabel='physics (PyBaMM) ' + nm, ylabel='ANN ' + nm)
ax[0].set_title(f'(a) V parity, RMSE {M["test"]["rmse_V_mV"]:.2f} mV'); ax[1].set_title(f'(b) ΔT parity, RMSE {M["test"]["rmse_T_K"]:.3f} K')
ax[2].set_title(f'(c) t_end parity, MAPE {M["test"]["tend_mape_pct"]:.3f} %')
eV = 1e3 * np.abs(Vp - D['V'][ti]); eT = np.abs(Tp - D['T'][ti])
ax[3].plot(TAU, np.median(eV, 0), color=C[0], lw=2, label='|ΔV| median [mV]')
ax[3].fill_between(TAU, np.percentile(eV, 5, 0), np.percentile(eV, 95, 0), color=C[0], alpha=0.2)
ax[3].plot(TAU, 10 * np.median(eT, 0), color=C[1], lw=2, label='10×|ΔT| median [K]')
ax[3].fill_between(TAU, 10 * np.percentile(eT, 5, 0), 10 * np.percentile(eT, 95, 0), color=C[1], alpha=0.2)
ax[3].set(xlabel='τ', ylabel='absolute error', title='(d) test error along the curve (5–95 % band)'); ax[3].legend(frameon=False)
for a in ax: clean(a)
fig.tight_layout(); fig.savefig(os.path.join(I, 'fig05_ann_parity.png'), dpi=150); plt.close(fig)
print('ok')
