"""Figures for real-data applications (NASA B0005, CALCE CS2_36, Oxford drive cycle) and cost comparison."""
import sys, os, json, pickle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from style import *

R = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); O = os.path.join(R, 'outputs'); I = os.path.join(R, 'report', 'images')
J = lambda f: json.load(open(os.path.join(O, f))); P = lambda f: pickle.load(open(os.path.join(O, f), 'rb'))

# ---------- NASA: fits for fresh and aged cycles ----------
N8 = J('nasa8_mmga.json'); C8 = P('nasa8_curves.pkl'); N6 = J('nasa_mmga.json'); C6 = P('nasa_curves.pkl')
fig, ax = plt.subplots(2, 2, figsize=(13, 8))
for k, cyc in enumerate([0, 85, 167]):
    e = C8[cyc]['exp']; m = C8[cyc]['free6']; m6 = C6[cyc]['free6']
    ax[0, 0].plot(e[0] / 60, e[1], 'o', ms=2.5, color=SEQ[k + 1], mfc='none', label=f'measured, cycle {cyc}')
    ax[0, 0].plot(m[0] / 60, m[1], color=C[k], lw=1.8, label=f'MMGA 8-p, cycle {cyc}')
    if cyc == 0: ax[0, 0].plot(m6[0] / 60, m6[1], color=C[k], lw=1.2, ls='--', label='MMGA 6-p (stoichiometry fixed), cycle 0')
    ax[0, 1].plot(e[0] / 60, e[2] - e[2][0], 'o', ms=2.5, color=SEQ[k + 1], mfc='none')
    ax[0, 1].plot(m[0] / 60, m[2] - m[2][0], color=C[k], lw=1.8)
ax[0, 0].set(xlabel='time [min]', ylabel='voltage [V]', title='(a) NASA B0005: voltage, 2 A CC discharge'); ax[0, 0].legend(frameon=False, fontsize=8)
ax[0, 1].set(xlabel='time [min]', ylabel='temperature rise ΔT [K]', title='(b) temperature rise (measured: surface; model: lumped)')
cyc = [c['cycle'] for c in N8['cycles']]
for key, ls, lab in [('free6', '-', '8-p free'), ('track3', '--', 'tracking (4 aging-p free)')]:
    ax[1, 0].plot(cyc, [c[key]['fit']['rmse_V_mV'] for c in N8['cycles']], ls, color=C[0], marker='o', ms=4, label='V RMSE [mV], ' + lab)
    ax[1, 0].plot(cyc, [10 * c[key]['fit']['rmse_dT_K'] for c in N8['cycles']], ls, color=C[1], marker='s', ms=4, label='10×ΔT RMSE [K], ' + lab)
ax[1, 0].plot(cyc, [c['free6']['fit']['rmse_V_mV'] for c in N6['cycles']], ':', color=C[0], marker='x', ms=4, label='V RMSE [mV], 6-p (stoich. fixed)')
ax[1, 0].set(xlabel='cycle', ylabel='error', title='(c) fit error over life'); ax[1, 0].legend(frameon=False, fontsize=7)
ax[1, 1].plot(cyc, [c['track3']['fit']['cap_meas_Ah'] for c in N8['cycles']], 'o', color='#52514e', mfc='none', label='measured (2 A to 2.7 V)')
ax[1, 1].plot(cyc, [c['free6']['fit']['cap_model_Ah'] for c in N8['cycles']], color=C[2], lw=1.2, ls=':', label='model capacity (8-p free)')
ax[1, 1].plot(cyc, [c['track3']['fit']['cap_model_Ah'] for c in N8['cycles']], color=C[0], lw=1.8, label='model capacity (tracking)')
ax[1, 1].set(xlabel='cycle', ylabel='capacity to 2.7 V [Ah]', title='(d) capacity'); ax[1, 1].legend(frameon=False)
for a in ax.flat: clean(a)
fig.tight_layout(); fig.savefig(os.path.join(I, 'fig11_nasa_fits.png'), dpi=150); plt.close(fig)

# aging-parameter trajectories
names = N8['names']; idx = {n: i for i, n in enumerate(names)}
fig, ax = plt.subplots(1, 4, figsize=(17, 3.8))
for k, (n, lab) in enumerate([('LAM_p', 'LAM_p (positive active-material fraction)'), ('LLI', 'LLI (initial negative stoich. factor)'),
                              ('R_f', 'R_f film resistance [Ω]'), ('k_n', 'k_n multiplier')]):
    ax[k].plot(cyc, [c['track3']['x'][idx[n]] for c in N8['cycles']], color=C[k], marker='o', ms=4, lw=1.8, label='tracking')
    ax[k].plot(cyc, [c['free6']['x'][idx[n]] for c in N8['cycles']], color=C[k], marker='x', ms=4, lw=0.8, ls=':', label='all 8 free')
    ax[k].set(xlabel='cycle', title=lab); clean(ax[k])
ax[0].legend(frameon=False)
fig.tight_layout(); fig.savefig(os.path.join(I, 'fig12_nasa_aging_params.png'), dpi=150); plt.close(fig)

# ---------- CS2 and Oxford ----------
fig, ax = plt.subplots(1, 3, figsize=(17, 4.2))
CS = J('cs2_7_mmga.json'); CC = P('cs2_7_curves.pkl'); CS5 = J('cs2_mmga.json'); CC5 = P('cs2_curves.pkl')
for k, c in enumerate(CS):
    cv = CC[c['cycle']]
    ax[0].plot(cv['exp'][0] / 60, cv['exp'][1], 'o', ms=3, color=SEQ[k + 1], mfc='none', label=f'measured, cycle {c["cycle"]}')
    ax[0].plot(cv['model'][0] / 60, cv['model'][1], color=C[k], lw=1.8, label=f'MMGA 7-p: {c["rmse_V_mV"]:.1f} mV')
cv5 = CC5[0]; ax[0].plot(cv5['model'][0] / 60, cv5['model'][1], color=C[0], lw=1.0, ls='--', label=f'MMGA 5-p (stoich. fixed): {CS5[0]["rmse_V_mV"]:.1f} mV')
ax[0].set(xlabel='time [min]', ylabel='voltage [V]', title='(a) CALCE CS2_36, 1C (1.1 A) CC discharge'); ax[0].legend(frameon=False, fontsize=7)
for k, c in enumerate(CS):
    cv = CC[c['cycle']]; ax[1].plot(cv['model'][0] / 60, cv['model'][2] - cv['model'][2][0], color=C[k], lw=1.8, label=f'cycle {c["cycle"]}')
ax[1].set(xlabel='time [min]', ylabel='predicted ΔT [K]', title='(b) CS2_36: model-predicted temperature (not measured)'); ax[1].legend(frameon=False)
try:
    OX = J('oxford_mmga.json'); OC = P('oxford_curves.pkl')
    tg = OC['tg'] / 60; s = OC['split']
    ax[2].plot(tg, OC['Vd'], color='#8a8984', lw=1.0, label='measured')
    ax[2].plot(tg[:s], OC['Vm'][:s], color=C[0], lw=1.2, label=f'identification window: {OX["rmse_V_ident_window_mV"]:.1f} mV')
    ax[2].plot(tg[s:], OC['Vm'][s:], color=C[1], lw=1.2, label=f'prediction window: {OX["rmse_V_prediction_window_mV"]:.1f} mV')
    ax[2].axvline(tg[s], color='#52514e', ls='--', lw=0.8)
    ax[2].set(xlabel='time [min]', ylabel='voltage [V]', title='(c) Oxford Artemis drive cycle (10 s-averaged)'); ax[2].legend(frameon=False, fontsize=8)
except Exception as ex:
    print('oxford missing', ex)
for a in ax: clean(a)
fig.tight_layout(); fig.savefig(os.path.join(I, 'fig13_cs2_oxford.png'), dpi=150); plt.close(fig)

# ---------- cost comparison ----------
B = J('step3_physga_baseline.json'); A = J('ann_core_metrics.json'); D = J('db_core3000.json')
fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
lab = ['physics-GA\n(40×26, case 0)', 'physics-GA\n(case 1)', 'MMGA online\n(case 0)', 'MMGA online\n(case 1)']
sims = [B[0]['physGA']['n_sims'], B[1]['physGA']['n_sims'], B[0]['MMGA']['n_sims_online'], B[1]['MMGA']['n_sims_online']]
tim = [B[0]['physGA']['time_s'], B[1]['physGA']['time_s'], B[0]['MMGA']['time_s'], B[1]['MMGA']['time_s']]
Js = [B[0]['physGA']['J_best'], B[1]['physGA']['J_best'], B[0]['MMGA']['J_phys'], B[1]['MMGA']['J_phys']]
col = [C[1], C[1], C[0], C[0]]
ax[0].bar(range(4), tim, color=col, width=0.6); ax[0].set_yscale('log'); ax[0].set_xticks(range(4)); ax[0].set_xticklabels(lab, fontsize=8)
for i in range(4): ax[0].text(i, tim[i] * 1.15, f'{tim[i]:.1f} s\n{sims[i]} sims\nJ={Js[i]:.3f}', ha='center', fontsize=8)
ax[0].set(ylabel='wall time per identification [s]', title='(a) per-identification cost (same target, same objective)', ylim=(1, 2000))
n = np.arange(1, 101)
ax[1].plot(n, n * B[0]['physGA']['time_s'], color=C[1], lw=2, label='physics-GA (1040 sims each)')
ax[1].plot(n, D['wall_time_s'] + A['train_time_s'] + n * B[0]['MMGA']['time_s'], color=C[0], lw=2, label='MMGA (offline DB + training, then online)')
ax[1].set(xlabel='number of identifications (cycles / cells)', ylabel='cumulative wall time [s]', title='(b) amortised cost incl. offline database + ANN training')
ax[1].legend(frameon=False)
for a in ax: clean(a)
fig.tight_layout(); fig.savefig(os.path.join(I, 'fig14_cost_comparison.png'), dpi=150); plt.close(fig)
print('ok')
