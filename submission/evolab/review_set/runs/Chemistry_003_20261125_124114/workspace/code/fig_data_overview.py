import numpy as np
from plotstyle import plt, C, style
from io_utils import read_extxyz
style()
fig, ax = plt.subplots(1, 4, figsize=(15, 3.5))
# random charges
d = np.load('../outputs/random_charges_labeled.npz')
P, q, L = d['pos'], d['q'][0], float(d['L'])
rpp, rpm = [], []
for p in P[:20]:
    dd = p[:, None] - p[None]; dd -= L * np.round(dd / L); r = np.linalg.norm(dd, axis=-1)
    iu = np.triu_indices(len(p), 1); s = np.outer(q, q)[iu]
    rpp.append(r[iu][s > 0]); rpm.append(r[iu][s < 0])
bins = np.linspace(0, 7.5, 61)
ax[0].hist(np.concatenate(rpp), bins, histtype='step', color=C['blue'], lw=2, label='like charges')
ax[0].hist(np.concatenate(rpm), bins, histtype='step', color=C['orange'], lw=2, label='opposite charges')
ax[0].set_xlabel('pair distance r (Å)'); ax[0].set_ylabel('count (20 frames)'); ax[0].legend()
ax[0].set_title('(a) random charges: pair distances')
ax[1].hist(d['e_coul'] / 128, 25, color=C['blue'], edgecolor=C['surface'])
ax[1].set_xlabel('Coulomb (Ewald) energy per atom (eV)'); ax[1].set_ylabel('frames')
ax[1].set_title('(b) random charges: $E_{\\rm Coul}$')
# dimer
dm = read_extxyz('../data/charged_dimer.xyz')
R = [np.linalg.norm(f['pos'][0] - f['pos'][4]) for f in dm]; E = [f['energy'] for f in dm]
ax[2].scatter(R, E, s=22, color=C['blue'], edgecolor=C['surface'], lw=0.8)
ax[2].set_xlabel('C–C separation R (Å)'); ax[2].set_ylabel('total energy (eV)')
ax[2].set_title('(c) charged dimer CH$_3^+$···CH$_3^-$')
# ag3
ag = read_extxyz('../data/ag3_chargestates.xyz')
for Qs, col, mk, lab in [(1, C['blue'], 'o', 'Ag$_3^+$'), (-1, C['orange'], 'x', 'Ag$_3^-$')]:
    fs = [f for f in ag if f['total_charge'] == Qs]
    rb = [np.mean(np.linalg.norm(f['pos'][[0, 0, 1]] - f['pos'][[1, 2, 2]], axis=-1)) for f in fs]
    ax[3].scatter(rb, [f['energy'] for f in fs], s=40 if mk == 'o' else 30, marker=mk, color=col,
                  facecolors='none' if mk == 'o' else None, lw=1.5, label=lab)
ax[3].set_xlabel('mean Ag–Ag bond length (Å)'); ax[3].set_ylabel('energy (eV)'); ax[3].legend()
ax[3].set_title('(d) Ag$_3$ charge states (as provided)')
plt.tight_layout(); plt.savefig('../report/images/fig1_data_overview.png'); print('ok')
