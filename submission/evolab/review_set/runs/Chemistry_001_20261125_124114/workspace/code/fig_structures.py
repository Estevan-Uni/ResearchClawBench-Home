"""Figure 6: structural overlays of sampled complexes; Figure 7: reverse-diffusion trajectory."""
import json, os, sys
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(__file__))
from fig_style import style, C, plt
from biomol import load_2l3r
from model import AF3Lite
from common import to_torch, sample, noise_schedule
from metrics import kabsch, apply, evaluate

style()
run = sys.argv[1] if len(sys.argv) > 1 else 'full'
tag = sys.argv[2] if len(sys.argv) > 2 else 'model'
R = f'/workspace/outputs/runs/{run}'
fnp = load_2l3r(ref_seed=0)
ref = fnp['coords']
mol = fnp['atom_moltype']; lig = np.where(mol == 3)[0]; prot = np.where(mol == 0)[0]
ca = fnp['token_center_atom'][fnp['moltype'] == 0]
X = np.load(f'{R}/samples_{tag}.npy')
ev = json.load(open(f'{R}/eval_{tag}.json'))['main']['metrics']
# frame: principal axes of native complex
cen = ref.mean(0); _, _, Vt = np.linalg.svd(ref - cen)
proj = lambda Y: (Y - cen) @ Vt.T
al = []
for x in X:
    Rm, t = kabsch(x[ca], ref[ca]); al.append(apply(Rm, t, x))
al = np.array(al)
order = np.argsort([m['lig_rmsd_pocketaligned'] for m in ev])
best = order[0]
fig, ax = plt.subplots(1, 3, figsize=(16, 5))
for a, (i, j), ttl in zip(ax[:2], [(0, 1), (0, 2)], ['PC1–PC2', 'PC1–PC3']):
    Pn = proj(ref)
    a.plot(Pn[ca, i], Pn[ca, j], color=C['muted'], lw=2.5, alpha=0.8, label='native protein (Cα)')
    Pb = proj(al[best])
    a.plot(Pb[ca, i], Pb[ca, j], color=C['blue'], lw=1.2, label=f'predicted protein (best sample)')
    for k in order[1:]:
        Pk = proj(al[k]); a.scatter(Pk[lig, i], Pk[lig, j], s=3, color=C['orange'], alpha=0.12, lw=0)
    a.scatter(Pn[lig, i], Pn[lig, j], s=16, color=C['ink'], marker='x', lw=0.8, label='native peptide atoms', zorder=4)
    a.scatter(Pb[lig, i], Pb[lig, j], s=14, color=C['orange'], edgecolor=C['surface'], lw=0.4, zorder=5, label='predicted peptide (best)')
    a.scatter([], [], s=10, color=C['orange'], alpha=0.4, label='peptide, other samples')
    a.set_aspect('equal'); a.set_title(f'{ttl} projection, Cα-superposed'); a.set_xlabel('Å'); a.set_ylabel('Å')
ax[0].legend(fontsize=7.5, loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=3)
# per-residue CA deviation across samples
dev = np.linalg.norm(al[:, ca] - ref[ca][None], axis=-1)
a = ax[2]
resn = np.arange(161) + 125
a.fill_between(resn, dev.min(0), dev.max(0), color=C['blue'], alpha=0.2, lw=0, label='range over samples')
a.plot(resn, np.median(dev, 0), color=C['blue'], lw=1.5, label='median over samples')
# interface residues
D = np.linalg.norm(ref[prot][:, None] - ref[lig][None], axis=-1).min(1)
a2t = fnp['atom_to_token']
iface = np.unique(a2t[prot[D < 4.5]])
a.scatter(resn[iface], np.zeros(len(iface)) - 0.05 * max(dev.max(), 1), marker='|', s=60, color=C['orange'], label='peptide-contact residues')
a.set_xlabel('UHRF1 residue number'); a.set_ylabel('Cα deviation after superposition (Å)')
a.set_title('Per-residue Cα error'); a.legend(fontsize=8)
plt.tight_layout(); plt.savefig(f'/workspace/report/images/fig6_structure_overlay.png', dpi=150); plt.close()
# per-residue peptide atom error (best sample, pocket frame = CA frame here)
np.save(f'{R}/ca_dev_{tag}.npy', dev)

# ------------------------------------------------ trajectory
ck = torch.load(f'{R}/{tag}.pt', weights_only=False)
a_ = ck['args']
model = AF3Lite(n_pairformer=a_.get('n_pairformer', 2), canon=a_.get('canon', False), canon_mode=a_.get('canon_mode', 'pca'),
                use_trunk=not a_['no_trunk'], use_bonds=not a_['no_bonds'], use_ref=not a_['no_ref'])
model.load_state_dict(ck['model']); model.eval()
f = to_torch(fnp)
with torch.no_grad():
    tr = model.trunk(f)
    n_steps = 100
    xf, traj = sample(model, f, tr, n_samples=1, n_steps=n_steps, seed=5, return_traj=True)
c = noise_schedule(n_steps).numpy()
snap = [0, 20, 40, 60, 75, 99]
fig, ax = plt.subplots(1, len(snap), figsize=(17, 3.4))
lddts = []
for a, k in zip(ax, snap):
    x = traj[k][0].numpy()
    Rm, t = kabsch(x[ca], ref[ca]); P = proj(apply(Rm, t, x))
    hi = c[k + 1] > 20
    a.plot(P[ca, 0], P[ca, 1], color=C['blue'], lw=0.25 if hi else 0.9, alpha=0.6 if hi else 1)
    a.scatter(P[lig, 0], P[lig, 1], s=5, color=C['orange'], lw=0)
    m = evaluate(x, fnp)
    a.set_title(f'step {k+1}: σ={c[k+1]:.2g} Å\nlDDT {m["lddt_all"]:.2f}, pep. RMSD {m["lig_rmsd_pocketaligned"]:.1f} Å', fontsize=8.5)
    a.set_aspect('equal'); a.set_xticks([]); a.set_yticks([]); a.grid(False)
    lim = max(40.0, float(np.percentile(np.abs(P[:, :2]), 99)))
    a.set_xlim(-lim, lim); a.set_ylim(-lim, lim)
    a.text(0.03, 0.03, f'box ±{lim:.0f} Å', transform=a.transAxes, fontsize=7.5, color=C['ink2'])
plt.tight_layout(); plt.savefig('/workspace/report/images/fig7_diffusion_trajectory.png', dpi=150); plt.close()
print('done')
