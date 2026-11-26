"""EDA of the 180 DM-designed gels: ranking, composition effects, phase separation, swelling, mechanics, steel."""
import sys; sys.path.insert(0, 'code')
from style import *
import pandas as pd, numpy as np
from scipy import stats
d = pd.read_csv('outputs/initial_180.csv')
allm = pd.read_csv('outputs/all_measured_formulations.csv')
y = d.Fa_glass
# ---------- Fig 2: sample-indexed bar chart + distribution ----------
fig, axs = plt.subplots(1, 2, figsize=(12, 3.6), gridspec_kw={'width_ratios': [3.2, 1]})
ax = axs[0]
idx = np.arange(1, 181)
ax.bar(idx, y, width=0.8, color=CAT[0], lw=0)
im = int(y.idxmax())
ax.bar(idx[im], y[im], width=0.8, color=CAT[1])
ax.annotate(f'G-max = {d.ID[im]}\n{y[im]:.1f} kPa\nHEA/BA/CBEA/ATAC/PEA/AAm =\n' + '/'.join(f'{d.loc[im, m]:.2f}' for m in MON),
            xy=(idx[im], y[im]), xytext=(idx[im] + 18, y[im] - 10), fontsize=7.5, arrowprops=dict(arrowstyle='-', color=INK2), va='top')
ax.axhline(y.median(), color=INK2, lw=1, ls='--'); ax.text(1, y.median() + 2, f'median {y.median():.1f} kPa', ha='left', fontsize=7.5, color=INK2)
top5 = y.nlargest(6).index[1:]
for i in top5: ax.text(idx[i], y[i] + 2, d.ID[i][2:], ha='center', fontsize=6, color=INK2, rotation=90)
ax.set_xlim(0, 181); ax.set_xlabel('DM-designed hydrogel (G-001 … G-180)'); ax.set_ylabel('Underwater adhesive strength on glass, $F_a$ (kPa)')
ax.set_title('a  Initial screen of the 180 protein-derived (DM) hydrogels', loc='left', fontsize=9.5)
ax = axs[1]
bins = np.linspace(0, 360, 25)
ax.hist(y, bins=bins, color=CAT[0], alpha=0.85, label=f'DM library (n=180)', edgecolor='white', lw=0.5)
ml = allm[allm.source == 'ML']
ax.hist(ml.Fa_glass, bins=bins, color=CAT[1], alpha=0.65, label=f'ML designs (n={len(ml)})', edgecolor='white', lw=0.5)
ax.set_xlabel('$F_a$ (kPa)'); ax.set_ylabel('count'); ax.legend(fontsize=7.5)
ax.set_title('b  Distribution', loc='left', fontsize=9.5)
save(fig, 'fig02_initial_screen_bar')

# ---------- Spearman table ----------
props = MON + ['Q', 'PhaseSep', 'Modulus', 'tand', "G''", 'XlogP3', 'Log_Slope']
d['Log_Slope'] = pd.to_numeric(d['Log_Slope'], errors='coerce')
rows = []
for p in props:
    s = d[[p, 'Fa_glass']].dropna()
    r, pv = stats.spearmanr(s[p], s.Fa_glass)
    rows.append(dict(variable=p, n=len(s), spearman_rho=r, p=pv))
sp = pd.DataFrame(rows); sp.to_csv('outputs/spearman_initial180.csv', index=False); print(sp.round(4).to_string())

# ---------- Fig 3: composition effects ----------
fig, axs = plt.subplots(1, 3, figsize=(13, 3.8), gridspec_kw={'width_ratios': [1.5, 1.4, 1]})
ax = axs[0]
q = pd.qcut(y, 3, labels=['low', 'mid', 'high'])
pos = 0; ticks = []
for j, m in enumerate(MON):
    for k, lev in enumerate(['low', 'mid', 'high']):
        b = ax.boxplot(d.loc[q == lev, m], positions=[j * 4 + k], widths=0.8, patch_artist=True, showfliers=False,
                       medianprops=dict(color=INK))
        b['boxes'][0].set(facecolor=CAT[[0, 3, 1][k]], alpha=0.8, edgecolor=INK2)
    ticks.append(j * 4 + 1)
ax.set_xticks(ticks); ax.set_xticklabels(MON); ax.set_ylabel('mole fraction in feed')
from matplotlib.patches import Patch
ax.legend([Patch(color=CAT[c]) for c in [0, 3, 1]], [f'low $F_a$ tercile', 'mid', 'high'], fontsize=7.5, loc='upper right')
ax.set_title('a  Composition by $F_a$ tercile (n=60 each)', loc='left', fontsize=9.5)
ax = axs[1]
spp = sp.set_index('variable').loc[MON + ['PhaseSep', 'Q', 'tand', 'Modulus', 'XlogP3']]
cols = [CAT[1] if r > 0 else CAT[0] for r in spp.spearman_rho]
ax.barh(range(len(spp))[::-1], spp.spearman_rho, color=cols, height=0.7)
for i, (r, p) in enumerate(zip(spp.spearman_rho, spp.p)):
    ax.text(r + (0.02 if r >= 0 else -0.02), len(spp) - 1 - i, f'{r:+.2f}{"*" if p < 0.05 else ""}', va='center', ha='left' if r >= 0 else 'right', fontsize=7.5)
ax.set_yticks(range(len(spp))[::-1]); ax.set_yticklabels(['PhaseSep' if v == 'PhaseSep' else ('tanδ' if v == 'tand' else v) for v in spp.index])
ax.axvline(0, color=INK2, lw=0.8); ax.set_xlim(-0.6, 0.7); ax.set_xlabel('Spearman ρ with $F_a$ (n=180; * p<0.05)')
ax.set_title('b  Univariate associations', loc='left', fontsize=9.5)
ax = axs[2]
sc = ax.scatter(d.BA + d.PEA, y, c=d.ATAC, cmap='Blues', s=16, edgecolor=INK2, lw=0.3, vmin=0, vmax=0.4)
plt.colorbar(sc, ax=ax, label='ATAC fraction')
ax.set_xlabel('BA + PEA (hydrophobic + aromatic)'); ax.set_ylabel('$F_a$ (kPa)')
ax.set_title('c  Hydrophobic/aromatic content', loc='left', fontsize=9.5)
save(fig, 'fig03_composition_effects')

# ---------- Fig 4: phase separation gate & swelling ----------
d['hasATAC'] = d.ATAC > 0.005
ps = d.groupby('hasATAC').PhaseSep.agg(['mean', 'size'])
ps_stats = {'PS_frac_with_ATAC': ps.loc[True, 'mean'], 'n_with_ATAC': int(ps.loc[True, 'size']),
            'PS_frac_without_ATAC': ps.loc[False, 'mean'], 'n_without_ATAC': int(ps.loc[False, 'size'])}
ct = pd.crosstab(d.hasATAC, d.PhaseSep); ps_stats['fisher_p'] = stats.fisher_exact(ct.values)[1]
s1 = d[d.PhaseSep == 1]; s0 = d[d.PhaseSep == 0]
ps_stats['median_Fa_PS1'] = s1.Fa_glass.median(); ps_stats['median_Fa_PS0'] = s0.Fa_glass.median()
ps_stats['n_PS1'] = len(s1); ps_stats['n_PS0'] = len(s0)
ps_stats['mwu_p'] = stats.mannwhitneyu(s1.Fa_glass, s0.Fa_glass).pvalue
ps_stats['rho_BA+PEA_inPS1'] = stats.spearmanr(s1.BA + s1.PEA, s1.Fa_glass)[0]
ps_stats['rho_BA+PEA_inPS0'] = stats.spearmanr(s0.BA + s0.PEA, s0.Fa_glass)[0]
ps_stats['rho_ATAC_Q'] = stats.spearmanr(d.ATAC, d.Q, nan_policy='omit')[0]
qq = d.dropna(subset=['Q'])
ps_stats['median_Fa_Q_le2'] = qq[qq.Q <= 2].Fa_glass.median(); ps_stats['median_Fa_Q_gt2'] = qq[qq.Q > 2].Fa_glass.median()
ps_stats['n_Q_gt2'] = int((qq.Q > 2).sum()); ps_stats['max_Fa_Q_gt2'] = qq[qq.Q > 2].Fa_glass.max()
# ATAC optimum
bins_a = [-0.001, 0.005, 0.06, 0.15, 0.25, 1]
d['ATACbin'] = pd.cut(d.ATAC, bins_a, labels=['0', '0.005–0.06', '0.06–0.15', '0.15–0.25', '>0.25'])
ab = d.groupby('ATACbin', observed=False).Fa_glass.agg(['median', 'size', 'max']); ab.to_csv('outputs/atac_bins.csv')
print(ab)
pd.Series(ps_stats).to_csv('outputs/phase_separation_stats.csv'); print(pd.Series(ps_stats))
fig, axs = plt.subplots(1, 4, figsize=(14, 3.5))
ax = axs[0]
ax.bar([0, 1], [ps.loc[False, 'mean'], ps.loc[True, 'mean']], color=[CAT[0], CAT[1]], width=0.6)
for i, k in enumerate([False, True]): ax.text(i, ps.loc[k, 'mean'] + 0.02, f"{ps.loc[k, 'mean']*100:.0f}%\n(n={int(ps.loc[k, 'size'])})", ha='center', fontsize=8)
ax.set_xticks([0, 1]); ax.set_xticklabels(['no ATAC', 'ATAC > 0']); ax.set_ylim(0, 1.15); ax.set_ylabel('fraction phase-separated')
ax.set_title('a  ATAC gates phase separation', loc='left', fontsize=9.5)
ax = axs[1]
for k, (sub, c) in enumerate([(s0, CAT[0]), (s1, CAT[1])]):
    xx = np.random.default_rng(k).normal(k, 0.07, len(sub))
    ax.scatter(xx, sub.Fa_glass, s=9, color=c, alpha=0.7)
    ax.plot([k - 0.25, k + 0.25], [sub.Fa_glass.median()] * 2, color=INK, lw=2)
ax.set_xticks([0, 1]); ax.set_xticklabels([f'homogeneous\n(n={len(s0)})', f'phase-separated\n(n={len(s1)})']); ax.set_ylabel('$F_a$ (kPa)')
ax.set_title(f"b  $F_a$ by morphology (MWU p={ps_stats['mwu_p']:.1e})", loc='left', fontsize=9.5)
ax = axs[2]
ax.boxplot([d.loc[d.ATACbin == l, 'Fa_glass'] for l in ab.index], tick_labels=ab.index, patch_artist=True, showfliers=True,
           boxprops=dict(facecolor=CAT[2] + '88', edgecolor=INK2), medianprops=dict(color=INK), flierprops=dict(markersize=3))
for i, n in enumerate(ab['size']): ax.text(i + 1, 152, f'n={n}', ha='center', fontsize=7, color=INK2)
ax.set_xlabel('ATAC mole fraction'); ax.set_ylabel('$F_a$ (kPa)'); ax.tick_params(axis='x', labelsize=7, rotation=30)
ax.set_title('c  Cationic optimum', loc='left', fontsize=9.5)
ax = axs[3]
sc = ax.scatter(qq.Q, qq.Fa_glass, c=qq.ATAC, cmap='Blues', s=14, edgecolor=INK2, lw=0.3)
ax.axvline(2, color=CAT[7], ls='--', lw=1); ax.text(2.05, 140, 'Q = 2', color=INK2, fontsize=8)
plt.colorbar(sc, ax=ax, label='ATAC')
ax.set_xlabel('equilibrium swelling ratio Q'); ax.set_ylabel('$F_a$ (kPa)')
ax.set_title(f"d  Swelling penalty (ρ(ATAC,Q)={ps_stats['rho_ATAC_Q']:.2f})", loc='left', fontsize=9.5)
save(fig, 'fig04_phase_separation_swelling')

# ---------- Fig 5: mechanics and steel ----------
mech = {}
fig, axs = plt.subplots(1, 3, figsize=(12, 3.5))
for ax, (col, lab, logx) in zip(axs[:2], [('Modulus', 'storage modulus (kPa)', True), ('tand', 'tan δ', False)]):
    s = d[[col, 'Fa_glass', 'PhaseSep']].dropna()
    r, p = stats.spearmanr(s[col], s.Fa_glass); mech[col] = (r, p, len(s))
    ax.scatter(s[col], s.Fa_glass, c=[CAT[1] if v == 1 else CAT[0] for v in s.PhaseSep], s=14, alpha=0.8, edgecolor='white', lw=0.3)
    if logx:
        import matplotlib.ticker as mt; ax.set_xscale('log'); ax.xaxis.set_minor_formatter(mt.NullFormatter())
    ax.set_xlabel(lab); ax.set_ylabel('$F_a$ (kPa)'); ax.set_title(f"{'ab'[list(axs).index(ax)]}  ρ={r:.2f} (p={p:.1e}, n={len(s)})", loc='left', fontsize=9.5)
axs[0].scatter([], [], c=CAT[1], label='phase-separated'); axs[0].scatter([], [], c=CAT[0], label='homogeneous'); axs[0].legend(fontsize=7.5)
ax = axs[2]
s = d[['Fa_steel', 'Fa_glass', 'ID']].dropna()
r, p = stats.spearmanr(s.Fa_steel, s.Fa_glass); rp, pp = stats.pearsonr(s.Fa_steel, s.Fa_glass); mech['steel_spearman'] = (r, p, len(s)); mech['steel_pearson'] = (rp, pp, len(s))
ax.scatter(s.Fa_glass, s.Fa_steel, color=CAT[2], s=20, edgecolor=INK2, lw=0.3)
lim = [0, max(s.Fa_glass.max(), s.Fa_steel.max()) * 1.1]; ax.plot(lim, lim, color=INK2, ls=':', lw=1); ax.text(lim[1]*0.7, lim[1]*0.75, 'y = x', fontsize=7.5, color=INK2)
ax.set_xlabel('$F_a$ glass (kPa)'); ax.set_ylabel('$F_a$ stainless steel (kPa)')
ax.set_title(f'c  Glass vs steel (n={len(s)}, ρ={r:.2f}, p={p:.2g})', loc='left', fontsize=9.5)
save(fig, 'fig05_mechanics_steel')
s.assign(ratio=s.Fa_steel / s.Fa_glass).to_csv('outputs/steel_vs_glass.csv', index=False)
mech['steel_median_ratio'] = float((s.Fa_steel / s.Fa_glass).median()); mech['steel_IDs'] = ','.join(s.ID)
mech['steel_glass_range'] = (s.Fa_glass.min(), s.Fa_glass.max()); mech['steel_range'] = (s.Fa_steel.min(), s.Fa_steel.max())
pd.Series({k: str(v) for k, v in mech.items()}).to_csv('outputs/mechanics_steel_stats.csv'); print(mech)
