"""Figure 1: data overview of the 2L3R complex and of the unified tokenisation."""
import sys, os, json
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from fig_style import style, C, plt
from biomol import load_2l3r, parse_sdf, ligand_reference_conformer, RESTYPES
from metrics import kabsch, apply, rmsd

style()
f = load_2l3r(ref_seed=0)
X = f['coords']; mol = f['atom_moltype']
prot = np.where(mol == 0)[0]; lig = np.where(mol == 3)[0]
ca = f['token_center_atom'][f['moltype'] == 0]
a2t = f['atom_to_token']

# interface statistics
D = np.linalg.norm(X[prot][:, None] - X[lig][None], axis=-1)
res_min = np.full(161, np.inf)
for i, d in zip(prot, D.min(1)):
    res_min[a2t[i]] = min(res_min[a2t[i]], d)
iface = np.where(res_min < 4.5)[0]
seq = [RESTYPES[r] for r in f['restype'][:161]]
stats = dict(n_tokens=int(len(f['restype'])), n_atoms=int(len(X)), n_protein_tokens=161, n_ligand_tokens=int(len(lig)),
             n_protein_heavy_atoms=int(len(prot)), n_ligand_heavy_atoms=int(len(lig)), n_ligand_bonds=int(len(f['lig_bonds'])),
             interface_residues=[int(f['residue_index'][i]) + 125 for i in iface],
             interface_residue_names=[seq[i] for i in iface],
             n_interface_residues=int(len(iface)),
             protein_radius_of_gyration=float(np.sqrt(((X[ca] - X[ca].mean(0)) ** 2).sum(-1).mean())),
             ligand_radius_of_gyration=float(np.sqrt(((X[lig] - X[lig].mean(0)) ** 2).sum(-1).mean())))
# reference-conformer quality
sdf = parse_sdf('/workspace/data/sample/2l3r/2l3r_ligand.sdf')
ref_rmsds = []
for s in range(1):
    pos, _ = ligand_reference_conformer(sdf, seed=s)
    R, t = kabsch(pos, sdf['coords']); ref_rmsds.append(rmsd(apply(R, t, pos), sdf['coords']))
stats_old = json.load(open('/workspace/outputs/data_stats.json')) if os.path.exists('/workspace/outputs/data_stats.json') else {}
stats['ligand_refconf_rmsd_to_native_5seeds'] = stats_old.get('ligand_refconf_rmsd_to_native_5seeds', ref_rmsds)
# protein template sanity: RMSD of template residue atoms vs own residue (after superposition)
tm = []
for t in range(161):
    at = np.where(a2t == t)[0]
    if f['ref_mask'][at].min() < 1 or len(at) < 4:
        continue
    R, tt = kabsch(f['ref_pos'][at], X[at]); tm.append(rmsd(apply(R, tt, f['ref_pos'][at]), X[at]))
stats['protein_template_rmsd_mean'] = float(np.mean(tm)); stats['protein_template_rmsd_median'] = float(np.median(tm))
json.dump(stats, open('/workspace/outputs/data_stats.json', 'w'), indent=1)
print(json.dumps(stats, indent=1))

fig, ax = plt.subplots(1, 4, figsize=(17, 5.0), gridspec_kw=dict(width_ratios=[1.15, 1, 1, 1]))
# (a) projection of complex on principal axes
Xc = X - X.mean(0); U, S, Vt = np.linalg.svd(Xc, full_matrices=False); P = Xc @ Vt.T
a = ax[0]
a.plot(P[ca, 0], P[ca, 1], color=C['blue'], lw=1.5, alpha=0.9, label='UHRF1 TTD (CA trace, 161 res.)')
a.scatter(P[lig, 0], P[lig, 1], s=14, color=C['orange'], edgecolor=C['surface'], lw=0.5, zorder=3, label='H3K9me3 peptide (90 heavy atoms)')
a.scatter(P[ca[iface], 0], P[ca[iface], 1], s=22, facecolor='none', edgecolor=C['ink'], lw=1, zorder=4, label='interface residues (<4.5 Å)')
a.set_aspect('equal'); a.set_xlabel('PC1 (Å)'); a.set_ylabel('PC2 (Å)'); a.set_title('a  2L3R complex (NMR model 1)')
a.legend(loc='upper center', bbox_to_anchor=(0.5, -0.18), fontsize=8, ncol=1)
# (b) contact map protein CA + ligand atoms
a = ax[1]
cen = X[f['token_center_atom']]
Dm = np.linalg.norm(cen[:, None] - cen[None], axis=-1)
im = a.imshow(Dm, cmap='Blues_r', vmin=0, vmax=30)
a.axhline(160.5, color=C['orange'], lw=1); a.axvline(160.5, color=C['orange'], lw=1)
a.set_title('b  token-centre distance map'); a.set_xlabel('token index'); a.set_ylabel('token index'); a.grid(False)
a.text(205, 15, 'ligand\natom tokens', color=C['ink'], fontsize=8, ha='center')
plt.colorbar(im, ax=a, fraction=0.046, label='distance (Å)')
# (c) per-residue minimum distance to ligand
a = ax[2]
a.bar(np.arange(161) + 125, np.minimum(res_min, 20), color=[C['orange'] if r < 4.5 else C['blue'] for r in res_min], width=0.9)
a.axhline(4.5, color=C['ink2'], lw=1, ls='--'); a.text(286, 5.0, '4.5 Å', fontsize=8, color=C['ink2'], ha='right')
a.set_xlabel('UHRF1 residue number'); a.set_ylabel('min. heavy-atom distance to peptide (Å)')
a.set_title(f'c  interface: {len(iface)} contact residues')
# (d) reference conformer vs native ligand
a = ax[3]
pos, _ = ligand_reference_conformer(sdf, seed=0)
R, t = kabsch(pos, sdf['coords']); Pr = apply(R, t, pos)
L = sdf['coords']; Lc = L.mean(0); U2, S2, V2 = np.linalg.svd(L - Lc)
pn = (L - Lc) @ V2.T; pr = (Pr - Lc) @ V2.T
for (i, j, o) in sdf['bonds']:
    a.plot(pn[[i, j], 0], pn[[i, j], 1], color=C['orange'], lw=1.6)
    a.plot(pr[[i, j], 0], pr[[i, j], 1], color=C['muted'], lw=1.2)
a.plot([], [], color=C['orange'], label='native peptide pose')
a.plot([], [], color=C['muted'], label=f'reference conformer (RMSD {ref_rmsds[0]:.1f} Å)')
a.set_aspect('equal'); a.legend(fontsize=8, loc='upper center', bbox_to_anchor=(0.5, -0.25)); a.set_title('d  ligand reference conformer')
a.set_xlabel('PC1 (Å)'); a.set_ylabel('PC2 (Å)')
plt.tight_layout()
plt.savefig('/workspace/report/images/fig1_data_overview.png', dpi=150)
print('saved')
