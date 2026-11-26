import json, glob
import numpy as np
from plotstyle import plt, C, style
from les import KE
style()
Q_TRUE = np.array([0.5, 1/6, 1/6, 1/6, -0.5, -1/6, -1/6, -1/6])
R = {json.load(open(f))['name']: json.load(open(f)) for f in glob.glob('../outputs/results/dimer_*_rc*.json')}
rows = []
for kind in ['sr', 'les']:
    for tag in ['plain', 'tagged']:
        ks = [f'dimer_{kind}_{tag}_corrected_rc4.0_s{s}' for s in range(3) if f'dimer_{kind}_{tag}_corrected_rc4.0_s{s}' in R]
        if not ks: continue
        E = np.array([R[k]['test']['E_mae_atom'] * 1e3 for k in ks]); F = np.array([R[k]['test']['F_mae'] * 1e3 for k in ks])
        Etr = np.array([R[k]['train']['E_mae_atom'] * 1e3 for k in ks])
        # binding-curve error for R in [6, 12] (beyond rc-range of the SR model)
        be = []
        for k in ks:
            d = np.load(f'../outputs/preds/{k}.npz'); m = (d['Rs'] >= 3.0) & (d['Rs'] <= 12.0)
            be.append(np.mean(np.abs(d['Eml'][m] - d['Eref'][m])) * 1e3)
        rows.append(dict(model=kind.upper(), species=tag, labels='E+F(corrected)', n_seeds=len(ks), E_mae_meV_atom=E.mean(), E_mae_std=E.std(),
                         E_train_mae=Etr.mean(), F_mae_meV_A=F.mean(), F_mae_std=F.std(), bind_mae_meV=np.mean(be), bind_mae_std=np.std(be)))
for k, lab in [('dimer_les_tagged_energy_rc4.0_s0', 'E only'), ('dimer_sr_tagged_energy_rc4.0_s0', 'E only'),
               ('dimer_les_tagged_raw_rc4.0_s0', 'E+F(raw, non-conservative)')]:
    if k in R:
        d = np.load(f'../outputs/preds/{k}.npz'); m = (d['Rs'] >= 3.0) & (d['Rs'] <= 12.0)
        rows.append(dict(model=k.split('_')[1].upper(), species='tagged', labels=lab, n_seeds=1, E_mae_meV_atom=R[k]['test']['E_mae_atom'] * 1e3, E_mae_std=0,
                         E_train_mae=R[k]['train']['E_mae_atom'] * 1e3,
                         F_mae_meV_A=R[k]['test']['F_mae'] * 1e3, F_mae_std=0, bind_mae_meV=np.mean(np.abs(d['Eml'][m] - d['Eref'][m])) * 1e3, bind_mae_std=0))
for r in rows: print(r)
json.dump(rows, open('../outputs/results/summary_dimer.json', 'w'), indent=1)

fig, ax = plt.subplots(1, 4, figsize=(16.5, 3.9))
ref = np.load('../outputs/preds/dimer_les_tagged_corrected_rc4.0_s0.npz')
ax[0].plot(ref['Rs'], ref['Eref'], color=C['ink'], lw=2.5, label='exact (Coulomb)')
sty = {('sr', 'plain'): (C['orange'], '--', 'SR'), ('sr', 'tagged'): (C['yellow'], ':', 'SR (tagged)'),
       ('les', 'plain'): (C['magenta'], '-.', 'LES (untagged species)'), ('les', 'tagged'): (C['blue'], '-', 'LES (tagged species)')}
for (kind, tag), (col, ls, lab) in sty.items():
    for s in range(3):
        k = f'dimer_{kind}_{tag}_corrected_rc4.0_s{s}'
        try:
            d = np.load(f'../outputs/preds/{k}.npz')
        except FileNotFoundError:
            continue
        ax[0].plot(d['Rs'], d['Eml'], color=col, ls=ls, lw=1.8, alpha=1 if s == 0 else 0.45, label=lab if s == 0 else None)
ax[0].axvspan(12.1, 16, color=C['grid'], alpha=0.6, lw=0)
ax[0].text(12.3, -0.035, 'beyond\ntraining\nrange', fontsize=8, color=C['ink2'])
ax[0].set_xlabel('C–C separation R (Å)'); ax[0].set_ylabel('binding energy (eV)')
ax[0].set_title('(a) binding curve, rigid CH$_3^+$···CH$_3^-$'); ax[0].legend(fontsize=7.5, loc='lower right')
ax[0].set_ylim(-0.42, 0.08)
# (b) latent charges
qs = ref['q']; scale = np.sqrt(KE / 1.0)  # data use k_e = 1; model uses k_e = 14.4 -> q_phys = q_LES * sqrt(14.4)
labels = ['C$^{(+)}$', 'H$^{(+)}$', 'C$^{(-)}$', 'H$^{(-)}$']
groups = [qs[:, 0], qs[:, 1:4].ravel(), qs[:, 4], qs[:, 5:8].ravel()]
true = [0.5, 1/6, -0.5, -1/6]
sgn = np.sign(np.corrcoef(qs.ravel(), np.tile(Q_TRUE, len(qs)))[0, 1])
for g, (vals, t) in enumerate(zip(groups, true)):
    ax[1].scatter(np.full(len(vals), g) + np.random.default_rng(g).uniform(-0.15, 0.15, len(vals)), sgn * vals * scale, s=14,
                  color=C['blue'], alpha=0.7, edgecolor='none')
    ax[1].plot([g - 0.3, g + 0.3], [t, t], color=C['ink'], lw=2)
try:
    dp = np.load('../outputs/preds/dimer_les_plain_corrected_rc4.0_s0.npz')
    qp = dp['q']
    gp = [qp[:, 0], qp[:, 1:4].ravel(), qp[:, 4], qp[:, 5:8].ravel()]
    for g, vals in enumerate(gp):
        ax[1].scatter(np.full(len(vals), g + 0.38), vals * scale * sgn, s=10, color=C['magenta'], marker='^', alpha=0.7,
                      label='LES untagged' if g == 0 else None)
except FileNotFoundError:
    pass
ax[1].scatter([], [], color=C['blue'], s=14, label='LES tagged')
ax[1].plot([], [], color=C['ink'], lw=2, label='exact')
ax[1].set_xticks(range(4)); ax[1].set_xticklabels(labels)
ax[1].set_ylabel('$q_{\\rm LES}\\sqrt{k_e/k_e^{\\rm data}}$ (e)'); ax[1].set_title('(b) latent charges (test frames)')
ax[1].legend(fontsize=7.5)
# (c) dipole
mu_ml = ref['mu_ml'] * scale * sgn; mu_t = ref['mu_true']
ax[2].scatter(mu_t.ravel(), mu_ml.ravel(), s=22, color=C['blue'], edgecolor=C['surface'])
lim = [mu_t.min() - 0.1, mu_t.max() + 0.1]
ax[2].plot(lim, lim, color=C['ink2'], lw=1)
ax[2].set_xlabel('exact dipole component (e·Å)'); ax[2].set_ylabel('LES dipole component (e·Å)')
ax[2].set_title('(c) dipole moment, test frames')
# (d) BEC diag
Z = ref['Z'] * scale * sgn
diag = np.einsum('naa->na', Z.reshape(-1, 3, 3)).reshape(len(ref['Z']), 8, 3)
for g, idx in enumerate([[0], [1, 2, 3], [4], [5, 6, 7]]):
    vals = diag[:, idx, :].ravel()
    ax[3].scatter(np.full(len(vals), g) + np.random.default_rng(g).uniform(-0.15, 0.15, len(vals)), vals, s=8, color=C['aqua'], alpha=0.7)
    ax[3].plot([g - 0.3, g + 0.3], [true[g], true[g]], color=C['ink'], lw=2)
ax[3].set_xticks(range(4)); ax[3].set_xticklabels(labels)
ax[3].set_ylabel('$Z^*_{i,\\alpha\\alpha}$ (e)'); ax[3].set_title('(d) BEC diagonal (exact: q$_i$)')
plt.tight_layout(); plt.savefig('../report/images/fig3_dimer.png')
mu_err = np.abs(mu_ml - mu_t)
stats = dict(sign=float(sgn), charge_scale=float(scale), q_mean_scaled=[float(np.mean(g) * scale * sgn) for g in groups],
             q_std_scaled=[float(np.std(g) * scale * sgn) for g in groups],
             mol_charge_cation=float((qs[:, :4].sum(1) * scale * sgn).mean()), mol_charge_anion=float((qs[:, 4:].sum(1) * scale * sgn).mean()),
             dipole_mae=float(mu_err.mean()), dipole_r2=float(1 - ((mu_ml - mu_t) ** 2).sum() / ((mu_t - mu_t.mean()) ** 2).sum()),
             bec_diag_mean=[float(diag[:, idx, :].mean()) for idx in [[0], [1, 2, 3], [4], [5, 6, 7]]],
             bec_offdiag_absmean=float(np.abs(Z.reshape(-1, 3, 3)[:, [0, 0, 1, 1, 2, 2], [1, 2, 0, 2, 0, 1]]).mean()))
try:
    stats['untagged_q_mean_scaled'] = [float(np.mean(g) * scale * sgn) for g in gp]
    stats['untagged_mol_charges'] = [float((qp[:, :4].sum(1) * scale).mean()), float((qp[:, 4:].sum(1) * scale).mean())]
except NameError:
    pass
json.dump(stats, open('../outputs/results/dimer_charge_stats.json', 'w'), indent=1)
print(stats)

# quadrupole moment (traceless, about the centre of mass) from latent vs exact charges
from io_utils import read_extxyz
from common import split
raw = read_extxyz('../data/charged_dimer.xyz')
_, te = split(len(raw), 12, seed=0)
Qml, Qt = [], []
for k, i in enumerate(te):
    p = raw[i]['pos'] - raw[i]['pos'].mean(0)
    def quad(q):
        return 0.5 * (q[:, None, None] * (3 * p[:, :, None] * p[:, None, :] - (p ** 2).sum(1)[:, None, None] * np.eye(3))).sum(0)
    Qml.append(quad(ref['q'][k] * scale * sgn)); Qt.append(quad(Q_TRUE))
Qml, Qt = np.array(Qml), np.array(Qt)
stats['quadrupole_mae'] = float(np.abs(Qml - Qt).mean()); stats['quadrupole_range'] = float(np.abs(Qt).max())
stats['quadrupole_r2'] = float(1 - ((Qml - Qt) ** 2).sum() / ((Qt - Qt.mean()) ** 2).sum())
json.dump(stats, open('../outputs/results/dimer_charge_stats.json', 'w'), indent=1)
print('quadrupole', stats['quadrupole_mae'], stats['quadrupole_range'], stats['quadrupole_r2'])
