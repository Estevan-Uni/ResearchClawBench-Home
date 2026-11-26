"""Optimisation-campaign analysis: strategies, EI vs predicted-mean lists, convergence, replicates, literature baseline."""
import sys; sys.path.insert(0, 'code')
from style import *
import pandas as pd, numpy as np
from scipy import stats
L = pd.read_csv('outputs/ml_designs_all_lists.csv')
A = pd.read_csv('outputs/all_measured_formulations.csv')
d = pd.read_csv('outputs/initial_180.csv'); P = pd.read_csv('outputs/pilot_P_series.csv')
key = pd.read_csv('outputs/key_formulations.csv')
G = d.Fa_glass.max(); DMmed = d.Fa_glass.median()
def fam(m):
    if m in ['RFR-RFR', 'GP-RFR', 'ENU-RFR']: return 'RFR'
    if m in ['SM-ETR', 'SM-GBM']: return 'ETR/GBM'
    return 'GP'
L['EI_surrogate'] = L.method.map(fam)
g = L[L.gelled]
st = g.groupby(['round', 'sheet', 'label']).agg(n=('Fa_glass', 'size'), median=('Fa_glass', 'median'), mean=('Fa_glass', 'mean'),
                                                max=('Fa_glass', 'max'), frac_gt_Gmax=('Fa_glass', lambda s: (s > G).mean()),
                                                frac_gt_200=('Fa_glass', lambda s: (s > 200).mean())).reset_index()
nog = L[~L.gelled].groupby('label').size(); st['n_no_gelation'] = st.label.map(nog).fillna(0).astype(int)
st['EI_surrogate'] = st.label.str.replace(r'-\drd-(ei|pred)$', '', regex=True).map(fam)
st.to_csv('outputs/strategy_stats.csv', index=False); print(st.round(1).to_string())
tests = {}
r1 = g[g['round'] == 1]
tests['R1_EI_vs_PRED_median'] = (r1[r1.sheet == 'EI'].Fa_glass.median(), r1[r1.sheet == 'PRED'].Fa_glass.median(),
                                 stats.mannwhitneyu(r1[r1.sheet == 'EI'].Fa_glass, r1[r1.sheet == 'PRED'].Fa_glass).pvalue)
e1 = r1[r1.sheet == 'EI']
tests['R1_EI_GPsurrogate_vs_RFRsurrogate'] = (e1[e1.EI_surrogate == 'GP'].Fa_glass.median(), e1[e1.EI_surrogate == 'RFR'].Fa_glass.median(),
                                              stats.mannwhitneyu(e1[e1.EI_surrogate == 'GP'].Fa_glass, e1[e1.EI_surrogate == 'RFR'].Fa_glass).pvalue)
for r in [2, 3]:
    s = g[g['round'] == r]
    tests[f'R{r}_EI_vs_PRED_median'] = (s[s.sheet == 'EI'].Fa_glass.median(), s[s.sheet == 'PRED'].Fa_glass.median(),
                                        stats.mannwhitneyu(s[s.sheet == 'EI'].Fa_glass, s[s.sheet == 'PRED'].Fa_glass).pvalue)
    tests[f'R{r}_RFRGP_vs_GPGP_median'] = (s[s.method == 'RFR-GP'].Fa_glass.median(), s[s.method == 'GP-GP'].Fa_glass.median(),
                                           stats.mannwhitneyu(s[s.method == 'RFR-GP'].Fa_glass, s[s.method == 'GP-GP'].Fa_glass).pvalue)
allEI = g[g.sheet == 'EI'].Fa_glass; allPR = g[g.sheet == 'PRED'].Fa_glass
tests['all_EI_vs_PRED_median'] = (allEI.median(), allPR.median(), stats.mannwhitneyu(allEI, allPR).pvalue)
tests['ML_unique_vs_DM_median'] = (A[A.source == 'ML'].Fa_glass.median(), DMmed, stats.mannwhitneyu(A[A.source == 'ML'].Fa_glass, d.Fa_glass).pvalue)
tests['frac_ML_unique_gt_Gmax'] = float((A[A.source == 'ML'].Fa_glass > G).mean())
pd.Series({k: str(tuple(np.round(v, 4)) if isinstance(v, tuple) else round(v, 4)) for k, v in tests.items()}).to_csv('outputs/campaign_tests.csv'); print(tests)

# ---------- Fig 12: strategy strip plot ----------
order = (st.sort_values(['round', 'sheet', 'median'], ascending=[True, True, False]))
fig, ax = plt.subplots(figsize=(14, 4.6))
famcol = {'GP': CAT[0], 'RFR': CAT[1], 'ETR/GBM': CAT[2]}
for i, (_, row) in enumerate(order.iterrows()):
    s = g[g.label == row.label]
    if row.sheet != s.sheet.iloc[0]: s = s[s.sheet == row.sheet]
    s = g[(g.label == row.label) & (g.sheet == row.sheet)]
    xx = i + np.random.default_rng(i).normal(0, 0.08, len(s))
    ax.scatter(xx, s.Fa_glass, s=16, color=famcol[row.EI_surrogate], alpha=0.8, edgecolor='white', lw=0.3, marker='o' if row.sheet == 'EI' else 'D')
    ax.plot([i - 0.3, i + 0.3], [row['median']] * 2, color=INK, lw=2)
    if row.n_no_gelation: ax.text(i, 8, f'{row.n_no_gelation}×NG', ha='center', fontsize=6.5, color=CAT[7])
labs = [f"{r.label.replace('-2rd', '').replace('-3rd', '')}\n[{r.sheet}]" for _, r in order.iterrows()]
ax.set_xticks(range(len(order))); ax.set_xticklabels(labs, rotation=70, fontsize=7)
ax.axhline(G, color=CAT[7], ls='--', lw=1); ax.text(len(order) - 0.5, G + 4, f'G-max (G-042) {G:.1f} kPa', ha='right', fontsize=7.5, color=INK2, bbox=dict(fc='white', ec='none', pad=1))
ax.axhline(DMmed, color=INK2, ls=':', lw=1); ax.text(len(order) - 0.5, DMmed + 4, f'DM-library median {DMmed:.1f} kPa', ha='right', fontsize=7.5, color=INK2, bbox=dict(fc='white', ec='none', pad=1))
bnd = order.groupby('round').size().cumsum().values
for b in bnd[:-1]: ax.axvline(b - 0.5, color=INK2, lw=0.8)
for r, (a0, a1) in zip([1, 2, 3], zip(np.r_[0, bnd[:-1]], bnd)): ax.text((a0 + a1 - 1) / 2, 375, f'Round {r}', ha='center', fontsize=9, color=INK)
ax.set_ylim(0, 395); ax.set_ylabel('measured $F_a$ on glass (kPa)')
for k, c in famcol.items(): ax.scatter([], [], color=c, label=f'EI/ranking surrogate: {k}')
ax.scatter([], [], marker='o', color='grey', label='EI-ranked list'); ax.scatter([], [], marker='D', color='grey', label='predicted-mean (PRED) list')
ax.legend(fontsize=7.5, ncol=5, loc='upper left', bbox_to_anchor=(0, 1.13))
save(fig, 'fig12_strategy_comparison')

# ---------- Fig 13: convergence ----------
fig, axs = plt.subplots(1, 3, figsize=(14, 4))
ax = axs[0]
rounds = [0, 1, 2, 3]
for m in MON:
    mu = [A[A['round'] == r].nlargest(10, 'Fa_glass')[m].mean() for r in rounds]
    ax.plot(rounds, mu, 'o-', color=MONCOL[m], lw=2, label=m)
ax.set_xticks(rounds); ax.set_xticklabels(['initial\n184', 'R1', 'R2', 'R3']); ax.set_ylabel('mean mole fraction, top-10 gels of the round')
ax.legend(fontsize=7.5, ncol=2); ax.set_title('a  Composition convergence of the best gels', loc='left', fontsize=9.5)
ax = axs[1]
for r in rounds:
    s = A[A['round'] == r].Fa_glass
    ax.scatter(np.full(len(s), r) + np.random.default_rng(r).normal(0, 0.08, len(s)), s, s=10, color=[ '#c3c2b7', CAT[0], CAT[1], CAT[2]][r], alpha=0.7)
best = [A[A['round'] <= r].Fa_glass.max() for r in rounds]
ax.plot(rounds, best, 'k-', lw=1.5, label='best so far'); ax.plot(rounds, [A[A['round'] == r].Fa_glass.median() for r in rounds], 's--', color=INK2, label='round median')
ax.set_xticks(rounds); ax.set_xticklabels(['initial\n184', 'R1', 'R2', 'R3']); ax.set_ylabel('$F_a$ (kPa)'); ax.legend(fontsize=7.5)
ax.set_title('b  Performance trajectory', loc='left', fontsize=9.5)
ax = axs[2]
# BA-PEA-ATAC ternary for gels with HEA+CBEA+AAm <= 0.05
s = A[(A.HEA + A.CBEA + A.AAm) <= 0.05].copy(); tot = s.BA + s.PEA + s.ATAC
a, b, c = s.BA / tot, s.PEA / tot, s.ATAC / tot
tx = b + c / 2; ty = c * np.sqrt(3) / 2
ax.plot([0, 1, 0.5, 0], [0, 0, np.sqrt(3) / 2, 0], color=INK2, lw=1)
sc = ax.scatter(tx, ty, c=s.Fa_glass, cmap='Blues', s=30, edgecolor=INK2, lw=0.4, vmin=0, vmax=360)
ib = s.Fa_glass.idxmax(); ax.scatter(tx[ib], ty[ib], marker='*', s=180, color=CAT[3], edgecolor=INK, zorder=5)
ax.text(tx[ib] + 0.03, ty[ib], 'R1-max', fontsize=8)
for t in [0.2, 0.4, 0.6, 0.8]:
    ax.plot([t / 2, 1 - t / 2], [t * np.sqrt(3) / 2] * 2, color=GRID, lw=0.6, zorder=0)
ax.text(-0.04, -0.05, 'BA', fontsize=9); ax.text(1.0, -0.05, 'PEA', fontsize=9); ax.text(0.47, 0.89, 'ATAC', fontsize=9)
ax.text(0.62, 0.3, 'ATAC gridlines 0.2…0.8', fontsize=6.5, color=INK2)
ax.set_aspect('equal'); ax.axis('off'); plt.colorbar(sc, ax=ax, label='$F_a$ (kPa)', shrink=0.8)
ax.set_title(f'c  BA–PEA–ATAC sub-simplex\n(HEA+CBEA+AAm ≤ 0.05, n={len(s)})', loc='left', fontsize=9.5)
save(fig, 'fig13_convergence_ternary')
conv = pd.DataFrame({m: [A[A['round'] == r].nlargest(10, 'Fa_glass')[m].mean() for r in rounds] for m in MON}, index=rounds)
conv['best_so_far'] = best; conv['round_median'] = [A[A['round'] == r].Fa_glass.median() for r in rounds]; conv.to_csv('outputs/convergence_top10.csv'); print(conv.round(3))

# ---------- replicates: P-series vs matching formulations ----------
ref = pd.concat([d[['ID'] + MON + ['Fa_glass']].assign(src='DM'), L[L.HEA.notna() & L.gelled].assign(ID=lambda x: x.sheet + ':' + x.label + '#' + x.pos.astype(str))[['ID'] + MON + ['Fa_glass']].assign(src='ML lists')])
rep = []
for _, p in P.iterrows():
    dist = np.abs(ref[MON].values - p[MON].values.astype(float)).sum(1) / 2
    j = int(np.argmin(dist)); rr = ref.iloc[j]
    rep.append(dict(P_ID=p.ID, match_ID=rr.ID, match_src=rr.src, TV_distance=dist[j], P_Fa10s=p.Fa_glass, P_Fa60s=p.Fa_glass60, match_Fa=rr.Fa_glass))
rep = pd.DataFrame(rep); rep.to_csv('outputs/replicate_matches.csv', index=False); print(rep.round(3).to_string())
ex = rep[rep.TV_distance < 0.005]
print('exact replicate pairs', len(ex), 'ratio P10s/match', (ex.P_Fa10s / ex.match_Fa).round(2).tolist())
P['ratio60_10'] = P.Fa_glass60 / P.Fa_glass; print('60s/10s ratios median', P.ratio60_10.median(), P.ratio60_10.min(), P.ratio60_10.max())

# ---------- Fig 14: literature baseline + replicates ----------
fig, axs = plt.subplots(1, 2, figsize=(13, 4.4), gridspec_kw={'width_ratios': [1.5, 1]})
ax = axs[0]
p8 = P[P.ID == 'P-008'].iloc[0]
items = [('DM-library median (n=180)', DMmed / 1000, CAT[0]), ('G-max: G-042 (DM)', G / 1000, CAT[0]),
         ('Best pilot GPRFR-2', 304.6 / 1000, CAT[6]),
         ('R1-max old-SM-GP#1 (list value)', 353.3 / 1000, CAT[1]), ('R1-max composition, P-008 10 s', p8.Fa_glass / 1000, CAT[1]),
         ('R1-max composition, P-008 60 s', p8.Fa_glass60 / 1000, CAT[1]),
         ('R2-max GP-GP-2rd#5', key.iloc[3].Fa_glass_kPa / 1000, CAT[1]), ('R3-max GP-GP-3rd#7', key.iloc[4].Fa_glass_kPa / 1000, CAT[1]),
         ('Mussel plaque, measured\n(M. edulis; paper_002 refs 16,17)', 0.3, CAT[3]), ('Mussel plaque, estimated from tenacity\n(M. californianus; paper_002)', 6.0, CAT[3])]
yy = np.arange(len(items))[::-1]
ax.barh(yy, [v for _, v, _ in items], color=[c for _, _, c in items], height=0.65)
for yv, (n, v, c) in zip(yy, items): ax.text(v * 1.08, yv, f'{v*1000:.0f} kPa' if v < 1 else f'{v:.1f} MPa', va='center', fontsize=7.5)
ax.set_yticks(yy); ax.set_yticklabels([n for n, _, _ in items], fontsize=7.5); ax.set_xscale('log'); ax.set_xlim(0.02, 20)
ax.axvline(1.0, color=CAT[7], ls='--', lw=1.2); ax.text(1.05, yy[0] + 0.1, 'task target 1 MPa', color=CAT[7], fontsize=8)
ax.set_xlabel('adhesive strength (MPa, log scale)'); ax.grid(axis='y', visible=False)
ax.set_title('a  Measured screening values vs literature baselines', loc='left', fontsize=9.5)
ax = axs[1]
ax.scatter(P.Fa_glass, P.Fa_glass60, s=30, color=CAT[2], edgecolor=INK2, lw=0.4)
for _, p in P.iterrows(): ax.text(p.Fa_glass + 2, p.Fa_glass60 + 2, p.ID[2:], fontsize=7, color=INK2)
lim = [0, 370]; ax.plot(lim, lim, ':', color=INK2); ax.set_xlim(lim); ax.set_ylim(0, 300)
for _, r in ex.iterrows():
    pp = P[P.ID == r.P_ID].iloc[0]
    ax.annotate('', xy=(r.match_Fa, pp.Fa_glass60), xytext=(pp.Fa_glass, pp.Fa_glass60), arrowprops=dict(arrowstyle='->', color=CAT[1], lw=1))
    ax.text(r.match_Fa, pp.Fa_glass60 - 12, r.match_ID.split(':')[-1], fontsize=6.5, color=CAT[1], ha='right')
ax.set_xlabel('$F_a$ at 10 s contact (kPa)'); ax.set_ylabel('$F_a$ at 60 s contact (kPa)')
ax.set_title('b  Pilot P-series: contact time & replicates', loc='left', fontsize=9.5)
ax.text(5, 285, 'arrows: same composition measured\nin another file (10 s / list value)', fontsize=7, color=CAT[1], va='top')
save(fig, 'fig14_literature_replicates')
