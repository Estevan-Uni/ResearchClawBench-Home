"""Data-overview figure (inputs only) and chi^2 / Δχ^2 summary table vs the digitised points."""
import os, json, csv, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import load_data, MODELS, NICE, COLORS, OUT, IMG, LABELS, SHARED

d = load_data()
fig = plt.figure(figsize=(16, 7))
gs = fig.add_gridspec(3, 5, width_ratios=[1, 1, 1, 1, 1.6], hspace=0.6, wspace=0.5)
allp = SHARED + ['f_EDE', 'log10_ac', 'w0', 'wa']
for k, p in enumerate(allp):
    ax = fig.add_subplot(gs[k // 4, k % 4])
    for i, m in enumerate(MODELS):
        if p in d[MODELS[m]]:
            mu, s = d[MODELS[m]][p]
            ax.errorbar(mu, -i, xerr=s, fmt='o', color=COLORS[m], capsize=3)
    if p == 'f_EDE': ax.axvline(0, color='gray', ls=':')
    if p == 'w0': ax.axvline(-1, color='gray', ls=':')
    if p == 'wa': ax.axvline(0, color='gray', ls=':')
    ax.set_yticks([0, -1, -2]); ax.set_yticklabels([NICE[m] for m in MODELS], fontsize=8)
    ax.set_title('$' + LABELS[p] + '$', fontsize=11); ax.set_ylim(-2.6, 0.6); ax.tick_params(axis='x', labelsize=8)
ax = fig.add_subplot(gs[2, 3]); ax.axis('off')
ax.text(0, 0.5, 'Table II/III means $\\pm 1\\sigma$\n(CMB+DESI)', fontsize=10, va='center')
for j, (key, lab) in enumerate([('sne_mu_points', r'Union3 $\Delta\mu$'), ('desi_dvrd_points', r'DESI $\Delta(D_V/r_d)$'),
                                ('desi_fap_points', r'DESI $\Delta F_\mathrm{AP}$')]):
    ax = fig.add_subplot(gs[j, 4]); pt = np.array(d[key])
    ax.errorbar(pt[:, 0], pt[:, 1], pt[:, 2], fmt='s' if j == 0 else 'o', color='k', mfc='w' if j == 0 else 'k', capsize=3)
    ax.axhline(0, color='gray', ls=':'); ax.set_title(lab + ' (digitised Fig. 6)', fontsize=10); ax.set_xlim(0, 2.5)
    if j == 2: ax.set_xlabel('$z$')
fig.suptitle('Input data overview: Table II/III CMB+DESI constraints (left) and digitised Fig. 6 points (right)', fontsize=12)
fig.savefig(os.path.join(IMG, 'figure1_data_overview.png'), dpi=130, bbox_inches='tight')

# chi2 summary
R = json.load(open(os.path.join(OUT, 'distance_fiducial_tests.json')))
sel = R['selected']; per = R[sel['fiducial']][sel['fap_mode']]['per_model']
rows = []
for m in MODELS:
    v = per[m]; bao = v['chi2_dvrd'] + v['chi2_fap']
    rows.append(dict(model=m, chi2_dvrd=v['chi2_dvrd'], chi2_fap=v['chi2_fap'], chi2_bao=bao, chi2_sn=v['chi2_mu'],
                     chi2_sn_offmarg=v['chi2_mu_offmarg'], chi2_tot=bao + v['chi2_mu'], chi2_tot_offmarg=bao + v['chi2_mu_offmarg']))
base = rows[0]
for r in rows:
    for k in ['chi2_bao', 'chi2_tot', 'chi2_tot_offmarg', 'chi2_sn_offmarg']:
        r['d' + k] = r[k] - base[k]
with open(os.path.join(OUT, 'chi2_summary.csv'), 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
for r in rows: print({k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items()})
# Planck18 point vs DESI(digitised) plane
A = json.load(open(os.path.join(OUT, 'acoustic_plane.json')))
print('acoustic', A)
