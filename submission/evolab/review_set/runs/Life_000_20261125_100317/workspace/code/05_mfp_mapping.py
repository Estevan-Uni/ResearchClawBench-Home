"""Minimal data-mining check: map mussel-foot-protein sequences (paper_002, Lee et al. 2011, Table 1)
onto the six residue classes used for the hydrogel monomers, and compare with the hydrogel library.
ASSUMED mapping (documented in report): nucleophilic S,T,C -> HEA; hydrophobic A,V,L,I,M -> BA;
acidic D,E -> CBEA; cationic K,R,H -> ATAC; aromatic Y(incl. Dopa),F,W -> PEA; amide N,Q -> AAm;
G, P, O (hydroxyproline) unassigned (excluded before renormalisation)."""
import sys; sys.path.insert(0, 'code')
from style import *
import pandas as pd, numpy as np
from sklearn.decomposition import PCA
SEQ = {  # (sequence, repeat number) transcribed from paper_002 Table 1
 'mfp-1': [('AKPSYOOTYK', 75)],
 'mfp-2': [('TDKAYKPNPCVVSKPCKNRGKCIWNGKAYRCKCAYGYGGRHC', 11)],
 'mfp-3': [('ADYYGPNYGPPRRYGGGNYNRYNRYGRRYGGYKGWNNGWNRGRRGKYW', 1)],
 'mfp-4': [('HVHTHRVLHK', 36), ('DDHVNDIAQTA', 16)],
 'mfp-5': [('SSEEYKGGYYPGNAYHYSGGSYHGSGYHGGYKGKYYGKAKKYYYKYKNSGKYKYLKKARKYHRKGYKYYGGSS', 1)],
 'mfp-6': [('GGGNYRGYCSNKGCRSGYIFYDNRGFCKYGSSSYKYDCGNYACLPRNPYGRVKYYCTKKYSCPDDFYYYNNKGYYYYNDKDYGCFNCGSYNGCCLRSGY', 1)],
 'tmp-1': [('YGYGNIYGYNGYGNGKTNIIVNKSGYGYGNGYGYGNGYIYYG', 5), ('YGNGKTTKIVVNKGNGYGYD', 3)],
}
CLS = {**{a: 'HEA' for a in 'STC'}, **{a: 'BA' for a in 'AVLIM'}, **{a: 'CBEA' for a in 'DE'},
       **{a: 'ATAC' for a in 'KRH'}, **{a: 'PEA' for a in 'YFW'}, **{a: 'AAm' for a in 'NQ'}}
rows = []; runs = []
for name, parts in SEQ.items():
    full = ''.join(s * n for s, n in parts)
    cl = [CLS.get(a, None) for a in full]
    cnt = pd.Series([c for c in cl if c]).value_counts()
    comp = {m: cnt.get(m, 0) / cnt.sum() for m in MON}
    rows.append(dict(protein=name, length=len(full), frac_unassigned=1 - cnt.sum() / len(full), **comp))
    # run lengths of identical classes (unassigned residues break runs)
    prev, L = None, 0
    for c in cl + [None]:
        if c is not None and c == prev: L += 1
        else:
            if prev is not None: runs.append(dict(protein=name, cls=prev, run=L))
            prev, L = c, 1 if c is not None else 0
mfp = pd.DataFrame(rows); mfp.to_csv('outputs/mfp_class_composition.csv', index=False); print(mfp.round(3).to_string())
pd.DataFrame(runs).to_csv('outputs/mfp_class_runs.csv', index=False)
d = pd.read_csv('outputs/initial_180.csv'); allm = pd.read_csv('outputs/all_measured_formulations.csv')
key = pd.read_csv('outputs/key_formulations.csv')
pca = PCA(2).fit(d[MON].values)
Z = pca.transform(d[MON].values); Zm = pca.transform(mfp[MON].values); Zml = pca.transform(allm[allm.source == 'ML'][MON].values)
Zk = pca.transform(key[MON].values)
fig, axs = plt.subplots(1, 2, figsize=(12.5, 4.4), gridspec_kw={'width_ratios': [1.25, 1]})
ax = axs[0]
sc = ax.scatter(Z[:, 0], Z[:, 1], c=d.Fa_glass, cmap='Blues', s=18, edgecolor=INK2, lw=0.3, label='DM library (n=180)')
ax.scatter(Zml[:, 0], Zml[:, 1], marker='x', color=CAT[1], s=14, lw=0.8, label='ML designs (rounds 1–3)')
ax.scatter(Zm[:, 0], Zm[:, 1], marker='D', color=CAT[6], s=40, edgecolor='white', lw=0.8, label='mussel foot proteins (paper_002)')
for (x, yv), n in zip(Zm, mfp.protein): ax.text(x + 0.015, yv + 0.015, n, fontsize=7.5, color=CAT[6])
for (x, yv), n in zip(Zk[[0, 2]], ['G-042 (G-max)', 'R1-max']):
    ax.scatter(x, yv, marker='*', s=160, color=CAT[7], edgecolor=INK, lw=0.6, zorder=5); ax.text(x + 0.02, yv - 0.04, n, fontsize=8, color=INK)
plt.colorbar(sc, ax=ax, label='$F_a$ of DM gels (kPa)')
ev = pca.explained_variance_ratio_
ax.set_xlabel(f'PC1 ({ev[0]*100:.0f}%): ' + ', '.join(f'{m} {v:+.2f}' for m, v in zip(MON, pca.components_[0]) if abs(v) > 0.3))
ax.set_ylabel(f'PC2 ({ev[1]*100:.0f}%): ' + ', '.join(f'{m} {v:+.2f}' for m, v in zip(MON, pca.components_[1]) if abs(v) > 0.45))
ax.legend(fontsize=7, loc='lower left'); ax.set_title('a  Composition space (PCA fitted on the 180 DM gels)', loc='left', fontsize=9.5)
ax = axs[1]
tab = pd.concat([mfp.set_index('protein')[MON],
                 pd.DataFrame([d[MON].mean()], index=['DM library mean']),
                 pd.DataFrame([d.loc[d.Fa_glass.idxmax(), MON]], index=['G-042']).astype(float),
                 pd.DataFrame([key.iloc[2][MON].astype(float)], index=['R1-max'])])
left = np.zeros(len(tab))
for m in MON:
    ax.barh(range(len(tab))[::-1], tab[m], left=left, color=MONCOL[m], height=0.72, label=m, edgecolor='white', lw=1)
    left += tab[m].values
ax.set_yticks(range(len(tab))[::-1]); ax.set_yticklabels(tab.index); ax.set_xlim(0, 1); ax.set_xlabel('class / monomer fraction (unassigned G, P, O excluded)')
ax.legend(ncol=6, fontsize=7, loc='upper center', bbox_to_anchor=(0.5, -0.13)); ax.grid(False)
ax.set_title('b  Residue-class composition vs hydrogel feeds', loc='left', fontsize=9.5)
save(fig, 'fig06_mfp_mapping_pca')
tab.to_csv('outputs/mfp_vs_hydrogel_composition_table.csv')
pd.DataFrame(pca.components_, columns=MON, index=['PC1', 'PC2']).assign(evr=ev).to_csv('outputs/pca_loadings_DM.csv')
