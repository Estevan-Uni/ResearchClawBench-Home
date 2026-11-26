import json, glob
import numpy as np
from plotstyle import plt, C, style
style()
R = {}
for fn in glob.glob('../outputs/results/rc_*.json'):
    r = json.load(open(fn)); R[r['name']] = r
rcs = [3.0, 4.0, 5.0, 6.0]
rows = []
for rc in rcs:
    for kind in ['sr', 'les']:
        k = f'rc_rc{rc}_{kind}_n40_s0'
        if k in R:
            t = R[k]['test']
            rows.append(dict(rc=rc, model=kind.upper(), ntrain=40, epochs=150, E_mae_meV_atom=t['E_mae_atom'] * 1e3, E_rmse_meV_atom=t['E_rmse_atom'] * 1e3,
                             F_mae=t['F_mae'], F_rmse=t['F_rmse'], train_time_min=R[k]['train_time'] / 60))
main = []
for kind in ['sr', 'les']:
    k = f'rc_rc5.0_{kind}_s0'
    if k in R:
        t = R[k]['test']
        main.append(dict(rc=5.0, model=kind.upper(), ntrain=80, epochs=200, E_mae_meV_atom=t['E_mae_atom'] * 1e3, E_rmse_meV_atom=t['E_rmse_atom'] * 1e3,
                         F_mae=t['F_mae'], F_rmse=t['F_rmse'], train_time_min=R[k]['train_time'] / 60))
k = 'rc_rc5.0_les_blind_n40_s0'
if k in R:
    t = R[k]['test']
    main.append(dict(rc=5.0, model='LES (species-blind)', ntrain=40, epochs=150, E_mae_meV_atom=t['E_mae_atom'] * 1e3, E_rmse_meV_atom=t['E_rmse_atom'] * 1e3,
                     F_mae=t['F_mae'], F_rmse=t['F_rmse'], train_time_min=R[k]['train_time'] / 60))
# robust force metric: atoms whose nearest neighbour is >= 1.0 A (excludes rare near-overlapping pairs)
from common import split
dl = np.load('../outputs/random_charges_labeled.npz')
_, te = split(100, 20, seed=0)
dnn = []
for i in te:
    p = dl['pos'][i]; dd = p[:, None] - p[None]; dd -= 15 * np.round(dd / 15); r = np.linalg.norm(dd, axis=-1); np.fill_diagonal(r, 99)
    dnn.append(r.min(1))
dnn = np.concatenate(dnn)
for row in main:
    nm = {'SR': 'rc_rc5.0_sr_s0', 'LES': 'rc_rc5.0_les_s0', 'LES (species-blind)': 'rc_rc5.0_les_blind_n40_s0'}[row['model']]
    pr = np.load(f'../outputs/preds/{nm}.npz')
    err = np.abs(pr['Fp'] - pr['Ft']).reshape(-1, 3)
    row['F_mae_dnn_ge_1A'] = float(err[dnn >= 1.0].mean()); row['frac_atoms_dnn_ge_1A'] = float((dnn >= 1.0).mean())
    row['F_median_ae'] = float(np.median(err))
for row in rows:
    pr = np.load(f"../outputs/preds/rc_rc{row['rc']}_{row['model'].lower()}_n40_s0.npz")
    err = np.abs(pr['Fp'] - pr['Ft']).reshape(-1, 3)
    row['F_mae_dnn_ge_1A'] = float(err[dnn >= 1.0].mean()); row['F_median_ae'] = float(np.median(err))
json.dump(dict(main=main, scan=rows), open('../outputs/results/summary_random_charges.json', 'w'), indent=1)
for r in main + rows:
    print(r)

fig, ax = plt.subplots(1, 4, figsize=(16, 3.8))
# (a) latent charges vs true
d = np.load('../outputs/preds/rc_rc5.0_les_s0.npz')
q, qt = d['q'].ravel(), d['qtrue'].ravel()
s = np.sign(np.corrcoef(q, qt)[0, 1]); qs = s * q
bins = np.linspace(-1.6, 1.6, 81)
ax[0].hist(qs[qt > 0], bins, color=C['blue'], label='true q = +1 e', edgecolor=C['surface'], lw=0.3)
ax[0].hist(qs[qt < 0], bins, color=C['orange'], label='true q = −1 e', edgecolor=C['surface'], lw=0.3)
if glob.glob('../outputs/preds/rc_rc5.0_les_blind_n40_s0.npz'):
    db = np.load('../outputs/preds/rc_rc5.0_les_blind_n40_s0.npz'); qb = db['q'].ravel()
    ax[0].hist(qb, bins, histtype='step', color=C['ink2'], lw=1.5, label='species-blind LES')
for v in (-1, 1):
    ax[0].axvline(v, color=C['ink2'], lw=1, ls='--')
ax[0].set_xlabel('latent charge $q_{\\rm LES}$ (e, sign-fixed)'); ax[0].set_ylabel('atoms (20 test frames)')
ax[0].set_title('(a) recovered charges'); ax[0].legend(fontsize=8, loc='upper center')
stats = dict(sign=float(s), q_plus_mean=float(qs[qt > 0].mean()), q_plus_std=float(qs[qt > 0].std()),
             q_minus_mean=float(qs[qt < 0].mean()), q_minus_std=float(qs[qt < 0].std()),
             corr=float(np.corrcoef(qs, qt)[0, 1]), rmse_vs_true=float(np.sqrt(np.mean((qs - qt) ** 2))))
if glob.glob('../outputs/preds/rc_rc5.0_les_blind_n40_s0.npz'):
    stats['blind_corr'] = float(np.corrcoef(qb, db['qtrue'].ravel())[0, 1]); stats['blind_q_std'] = float(qb.std())
# (b) force parity
for kind, col, lab in [('sr', C['orange'], 'SR'), ('les', C['blue'], 'LES')]:
    dd = np.load(f'../outputs/preds/rc_rc5.0_{kind}_s0.npz')
    ax[1].scatter(dd['Ft'].ravel(), dd['Fp'].ravel(), s=3, alpha=0.4, color=col, label=lab, rasterized=True)
lim = 30
ax[1].plot([-lim, lim], [-lim, lim], color=C['ink2'], lw=1)
ax[1].set_xlim(-lim, lim); ax[1].set_ylim(-lim, lim)
ax[1].set_xlabel('reference force (eV/Å)'); ax[1].set_ylabel('predicted force (eV/Å)')
ax[1].set_title('(b) test forces, $r_c$ = 5 Å'); ax[1].legend(markerscale=4)
# (c) error vs cutoff
for kind, col, mk in [('SR', C['orange'], 's'), ('LES', C['blue'], 'o')]:
    rr = [r for r in rows if r['model'] == kind]
    ax[2].plot([r['rc'] for r in rr], [r['F_mae'] for r in rr], marker=mk, color=col, ms=6, lw=1.2, ls='--', alpha=0.6, label=kind + ' (all atoms)')
    ax[2].plot([r['rc'] for r in rr], [r['F_mae_dnn_ge_1A'] for r in rr], marker=mk, color=col, ms=8, label=kind + ' ($d_{nn}\\geq$1 Å)')
ax[2].set_yscale('log'); ax[2].legend(fontsize=7.5)
ax[2].set_xlabel('short-range cutoff $r_c$ (Å)'); ax[2].set_ylabel('test force MAE (eV/Å)')
ax[2].set_title('(c) force error vs cutoff (40 train)')
ax2 = ax[3]
for kind, col, mk in [('SR', C['orange'], 's'), ('LES', C['blue'], 'o')]:
    rr = [r for r in rows if r['model'] == kind]
    ax2.plot([r['rc'] for r in rr], [r['E_mae_meV_atom'] for r in rr], marker=mk, color=col, ms=8, label=kind)
ax2.set_yscale('log')
ax2.set_xlabel('short-range cutoff $r_c$ (Å)'); ax2.set_ylabel('test energy MAE (meV/atom)')
ax2.set_title('(d) energy error vs cutoff (40 train)'); ax2.legend()
plt.tight_layout(); plt.savefig('../report/images/fig2_random_charges.png')

# BEC
import os
if os.path.exists('../outputs/preds/rc_rc5.0_les_s0_bec.npy'):
    Z = np.load('../outputs/preds/rc_rc5.0_les_s0_bec.npy') * s
    qt3 = d['qtrue'][:3].ravel()
    diag = np.einsum('naa->na', Z.reshape(-1, 3, 3)); off = Z.reshape(-1, 3, 3)[:, [0, 0, 1, 1, 2, 2], [1, 2, 0, 2, 0, 1]]
    stats['bec_diag_plus'] = float(diag[qt3 > 0].mean()); stats['bec_diag_minus'] = float(diag[qt3 < 0].mean())
    stats['bec_diag_std'] = float(np.concatenate([diag[qt3 > 0] - diag[qt3 > 0].mean(), diag[qt3 < 0] - diag[qt3 < 0].mean()]).std())
    stats['bec_offdiag_absmean'] = float(np.abs(off).mean())
    stats['bec_trace_vs_q_corr'] = float(np.corrcoef(diag.mean(1), (qs.reshape(-1, 128)[:3]).ravel())[0, 1])
    fig, ax = plt.subplots(1, 2, figsize=(8.5, 3.6))
    bb = np.linspace(-0.08, 0.08, 41)
    ax[0].hist(diag[qt3 > 0].ravel() - 1, bb, histtype='step', lw=2, color=C['blue'], label='true q = +1 e')
    ax[0].hist(diag[qt3 < 0].ravel() + 1, bb, histtype='step', lw=2, color=C['orange'], label='true q = −1 e')
    ax[0].set_xlabel('$Z^*_{i,\\alpha\\alpha} - q_i^{\\rm true}$ (e)'); ax[0].set_ylabel('count'); ax[0].legend(fontsize=8)
    ax[0].set_title('(a) BEC diagonal minus true charge')
    ax[1].hist(off.ravel(), np.linspace(-0.04, 0.04, 41), color=C['aqua'], edgecolor=C['surface'], lw=0.3)
    ax[1].set_xlabel('BEC off-diagonal $Z^*_{i,\\alpha\\beta}$ (e)'); ax[1].set_ylabel('count')
    ax[1].set_title('(b) off-diagonal (exact: 0)')
    plt.tight_layout(); plt.savefig('../report/images/fig2b_random_charges_bec.png')
json.dump(stats, open('../outputs/results/random_charges_charge_stats.json', 'w'), indent=1)
json.dump(dict(main=main, scan=rows), open('../outputs/results/summary_random_charges.json', 'w'), indent=1)
print(stats)
