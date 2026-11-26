"""Figures for data overview and GP calibration."""
import sys, pickle, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import norm
sys.path.insert(0, 'code')
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False, 'figure.dpi': 130})
C1, C2, C3, C4 = '#2a6f97', '#e07a5f', '#81b29a', '#6d597a'
IMG = 'report/images/'
cal = pd.read_csv('data/tg_calibration.csv'); vit = pd.read_csv('outputs/vitrimer_calibrated.csv')
F = np.load('outputs/features.npz'); G = pickle.load(open('outputs/gp_calibration.pkl', 'rb'))
best, prim = G['best'], G['primary']

# ---- Fig 1 data overview
fig, ax = plt.subplots(1, 4, figsize=(15, 3.4))
ax[0].scatter(cal.tg_exp, cal.tg_md, s=12, c=C1, alpha=.7)
lim = [150, 650]; ax[0].plot(lim, lim, 'k--', lw=1); ax[0].set_xlim(lim); ax[0].set_ylim(lim)
ax[0].set_xlabel('Experimental Tg (K)'); ax[0].set_ylabel('MD Tg (K)')
ax[0].set_title('Calibration set (n=295)\nMAE %.1f K, bias %+.1f K' % ((cal.tg_md - cal.tg_exp).abs().mean(), (cal.tg_md - cal.tg_exp).mean()), fontsize=9)
ax[1].hist(cal.tg_md - cal.tg_exp, bins=30, color=C2, alpha=.8); ax[1].axvline(0, c='k', lw=1)
ax[1].set_xlabel('Tg_MD - Tg_exp (K)'); ax[1].set_ylabel('count'); ax[1].set_title('MD offset distribution', fontsize=9)
ax[2].hist(vit.tg, bins=50, color=C3, alpha=.8, label='vitrimer MD (n=8,424)')
ax[2].hist(cal.tg_md, bins=30, color=C1, alpha=.5, label='calibration MD')
ax[2].set_xlabel('MD Tg (K)'); ax[2].legend(fontsize=8); ax[2].set_title('MD Tg distributions', fontsize=9)
ax[3].hist(F['cal_nheavy'], bins=30, color=C1, alpha=.6, label='calibration repeat units', density=True)
ax[3].hist(F['vit_nheavy'], bins=30, color=C3, alpha=.6, label='vitrimer repeat units', density=True)
ax[3].set_xlabel('heavy atoms per repeat unit'); ax[3].legend(fontsize=8); ax[3].set_title('Domain gap in size', fontsize=9)
plt.tight_layout(); plt.savefig(IMG + 'fig1_data_overview.png'); plt.close()

# ---- Fig 2 parity + residual histogram (LOOCV)
y = cal.tg_exp.values
fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))
for name, col, lab in [('Raw MD', '#aaaaaa', 'raw MD'), ('Linear(MD)', C2, 'linear(MD)'), (best, C1, 'GP dTg (best)')]:
    mu = G['loo'][name][0]
    ax[0].scatter(y, mu, s=10, alpha=.6, c=col, label=f'{lab}: MAE {np.abs(mu - y).mean():.1f} K')
ax[0].plot([150, 650], [150, 650], 'k--', lw=1); ax[0].set_xlim(150, 650); ax[0].set_ylim(150, 650)
ax[0].set_xlabel('Experimental Tg (K)'); ax[0].set_ylabel('Predicted Tg (K, LOOCV)'); ax[0].legend(fontsize=8)
ax[0].set_title('Parity (leave-one-out)', fontsize=10)
mu, var = G['loo'][best]
ho = G['ho_mask']
ax[1].errorbar(y, mu, yerr=1.96 * np.sqrt(var), fmt='o', ms=3, c=C1, alpha=.4, elinewidth=.6)
ax[1].scatter(y[ho], mu[ho], c=C2, s=40, marker='*', zorder=5, label='epoxy/phenoxy resins')
ax[1].plot([150, 650], [150, 650], 'k--', lw=1); ax[1].set_xlim(150, 650); ax[1].set_ylim(150, 650)
ax[1].set_xlabel('Experimental Tg (K)'); ax[1].set_ylabel('Calibrated Tg (K)'); ax[1].legend(fontsize=8)
ax[1].set_title('Best GP with 95% intervals', fontsize=10)
b = np.linspace(-200, 200, 41)
ax[2].hist(G['loo']['Raw MD'][0] - y, bins=b, alpha=.5, color='#999999', label='raw MD')
ax[2].hist(mu - y, bins=b, alpha=.8, color=C1, label='GP dTg (best)')
ax[2].hist(G['loo'][prim][0] - y, bins=b, alpha=.4, color=C3, label='GP dTg Tanimoto-only')
ax[2].axvline(0, c='k', lw=1); ax[2].set_xlabel('Residual: predicted - experimental (K)'); ax[2].set_ylabel('count')
ax[2].legend(fontsize=8); ax[2].set_title('LOOCV residual histogram', fontsize=10)
plt.tight_layout(); plt.savefig(IMG + 'fig2_gp_parity_residuals.png'); plt.close()

# ---- Fig 3 reliability diagram + model comparison bars
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
lv = np.linspace(0.05, 0.95, 19)
for name, col in [(best, C1), (prim, C3), ('Linear(MD)', C2)]:
    m, v = G['loo'][name]; z = np.abs(y - m) / np.sqrt(v)
    emp = [(z <= norm.ppf(0.5 + l / 2)).mean() for l in lv]
    ax[0].plot(lv, emp, 'o-', ms=3, c=col, label=name.replace('GP dTg | ', 'GP dTg: '))
ax[0].plot([0, 1], [0, 1], 'k--', lw=1); ax[0].set_xlabel('nominal coverage'); ax[0].set_ylabel('empirical coverage (LOOCV)')
ax[0].legend(fontsize=7); ax[0].set_title('Reliability diagram', fontsize=10)
rows = G['rows']; names = [k for k in rows]
maes = [rows[k]['MAE'] for k in names]
o = np.argsort(maes)[::-1]
ax[1].barh([names[i].replace('GP dTg | ', 'GP dTg: ') for i in o], [maes[i] for i in o], color=[C1 if 'dTg' in names[i] else '#bbbbbb' for i in o])
ax[1].axvline(17.4, c=C2, ls=':', lw=1.2, label='previous composite GP (5-fold, 17.4 K)')
ax[1].set_xlabel('LOOCV MAE (K)'); ax[1].legend(fontsize=7); ax[1].set_title('Calibration model comparison', fontsize=10)
ax[1].tick_params(axis='y', labelsize=7)
plt.tight_layout(); plt.savefig(IMG + 'fig3_gp_reliability_comparison.png'); plt.close()

# ---- Fig 4 vitrimer calibration
fig, ax = plt.subplots(1, 3, figsize=(14, 3.8))
ax[0].hist(vit.tg, bins=50, alpha=.6, color='#999999', label='MD (raw)')
ax[0].hist(vit.tg_cal_best, bins=50, alpha=.7, color=C1, label='GP-calibrated')
ax[0].set_xlabel('Tg (K)'); ax[0].legend(fontsize=8)
ax[0].set_title('8,424 vitrimers: mean shift %.1f K' % (vit.tg_cal_best - vit.tg).mean(), fontsize=10)
ax[1].scatter(vit.tg, vit.tg_cal_best, s=2, alpha=.3, c=C1); ax[1].plot([300, 570], [300, 570], 'k--', lw=1)
ax[1].set_xlabel('MD Tg (K)'); ax[1].set_ylabel('calibrated Tg (K)'); ax[1].set_title('Calibration map (vitrimers)', fontsize=10)
ms = np.load('outputs/cal_maxsim_self.npy')
ax[2].hist(ms, bins=30, alpha=.6, density=True, color=C3, label='calibration (LOO nearest)')
ax[2].hist(vit.maxsim_cal, bins=30, alpha=.6, density=True, color=C4, label='vitrimer -> calibration set')
ax[2].set_xlabel('max MinMax similarity'); ax[2].legend(fontsize=8); ax[2].set_title('Domain gap in chemistry', fontsize=10)
plt.tight_layout(); plt.savefig(IMG + 'fig4_vitrimer_calibration.png'); plt.close()
print('ok')
