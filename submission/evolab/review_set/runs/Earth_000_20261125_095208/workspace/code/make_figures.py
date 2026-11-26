"""All report figures (PNG) from outputs/ CSVs."""
import sys, os, json, numpy as np, pandas as pd, warnings
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')
sys.path.insert(0, 'code'); import glambie_lib as L

IMG = 'report/images'; os.makedirs(IMG, exist_ok=True)
OUT = 'outputs'; CFG = os.path.join(OUT, 'configs')
# validated default categorical palette (fixed order)
C = dict(D='#2a78d6', A='#eb6834', G='#1baf7a', H='#eda100', comb='#222222', off='#8a8a8a', V='#6250d6', R='#e34948', M='#e87ba4')
SRC_COL = {'glaciological': '#2a78d6', 'demdiff': '#6250d6', 'altimetry': '#eb6834', 'gravimetry': '#1baf7a', 'combined': '#eda100'}
plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True,
                     'grid.color': '#e6e6e6', 'grid.linewidth': 0.6, 'axes.edgecolor': '#888888', 'lines.linewidth': 1.6,
                     'legend.frameon': False, 'savefig.dpi': 160, 'figure.dpi': 110})
S = json.load(open(os.path.join(OUT, 'summary_stats.json')))
G = pd.read_csv(os.path.join(OUT, 'global_series_main.csv')).set_index('year')
inv = pd.read_csv(os.path.join(OUT, 'inventory.csv'))
R = pd.read_csv(os.path.join(OUT, 'regional_summary.csv'))
HY = {r: pd.read_csv(os.path.join(CFG, 'main', f'{r}_hydro.csv')).set_index('hy') for r in L.REGIONS}
SHORT = ['ALA', 'WNA', 'ACN', 'ACS', 'GRL', 'ISL', 'SJM', 'SCA', 'RUA', 'ASN', 'CEU', 'CAU', 'ASC', 'ASW', 'ASE', 'TRP', 'SAN', 'NZL', 'ANT']

def save(fig, name):
    fig.savefig(os.path.join(IMG, name), bbox_inches='tight', facecolor='white'); plt.close(fig)

# ------------------------------------------------------------ 1 data overview
fig, ax = plt.subplots(1, 2, figsize=(11, 4.6), gridspec_kw=dict(width_ratios=[1, 1.25]))
cnt = inv.groupby(['rid', 'source']).size().unstack(fill_value=0)[list(SRC_COL)]
left = np.zeros(19)
for s in SRC_COL:
    ax[0].barh(np.arange(19), cnt[s].values, left=left, color=SRC_COL[s], label=f'{s} ({cnt[s].sum()})', edgecolor='white', linewidth=1)
    left += cnt[s].values
ax[0].set_yticks(np.arange(19)); ax[0].set_yticklabels([f'{i+1} {L.LABELS[i]}' for i in range(19)]); ax[0].invert_yaxis()
ax[0].set_xlabel('number of input datasets'); ax[0].set_title(f'(a) {len(inv)} input files by region and method', loc='left')
ax[0].legend(loc='lower right', fontsize=8); ax[0].grid(axis='y', visible=False)
years = np.arange(1995, 2025)
for s in SRC_COL:
    sub = inv[inv.source == s]
    n = [((sub.t0 <= y + 0.5) & (sub.t1 >= y + 0.5)).sum() for y in years]
    ax[1].plot(years, n, color=SRC_COL[s], label=s)
ax[1].set_xlim(1995, 2024.5); ax[1].set_ylabel('datasets covering the year (all regions)')
ax[1].set_title('(b) temporal coverage of the inputs', loc='left'); ax[1].legend(fontsize=8, loc='upper left')
ax[1].axvspan(2017.4, 2018.5, color='#f2f2f2', zorder=0); ax[1].text(2017.95, 2, 'GRACE /\nGRACE-FO gap', ha='center', fontsize=7, color='#555')
save(fig, 'fig01_data_overview.png')

# ------------------------------------------------------------ 2 workflow for one region (Iceland)
reg = '6_iceland'; rid = 6
cfg = dict(L.DEFAULT_CFG)
ds = L.load_region(reg)
fig, axs = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
k = L.RGI_AREA[rid - 1] / L.area_model(reg)(L.YEARS + L.hy_offset(rid) + 0.5)
panels = {'D': (axs[0, 0], ['glaciological', 'demdiff', 'combined'], '(a) DEM differencing + glaciological (+ hybrids)'),
          'A': (axs[0, 1], ['altimetry'], '(b) altimetry'), 'G': (axs[1, 0], ['gravimetry'], '(c) gravimetry')}
t = HY[reg].loc[:2023]
for key, (ax, srcs, title) in panels.items():
    for d in ds:
        if d['src'] not in srcs: continue
        a = L.annual_series(d, cfg)
        lab = d['name'].split(f"_{d['src']}_")[1][:28]
        if a is not None and len(a) > 2:
            ax.plot(a.index, a.v * k[np.searchsorted(L.YEARS, a.index)], color=SRC_COL[d['src']], lw=0.8, alpha=0.55)
        else:
            for t0, t1, v, e in L.trend_records(d, pentad=True):
                ax.hlines(v / (t1 - t0), t0 + 0.25, t1 + 0.25, color=SRC_COL[d['src']], lw=1.2, alpha=0.8)
    col = L.GROUP_COL[key]
    ax.fill_between(t.index, t[col + '_mwe'] - t[col + '_mwe_errors'], t[col + '_mwe'] + t[col + '_mwe_errors'], color=C[key], alpha=0.15, lw=0)
    ax.plot(t.index, t[col + '_mwe'], color=C[key], lw=2.2, label='group estimate (this study)')
    o = L.official_hy(reg)
    if col + '_mwe' in o:
        ax.plot(o.index, o[col + '_mwe'], color=C['off'], ls='--', lw=1.3, label='official group (GlaMBIE)')
    ax.set_title(title, loc='left'); ax.set_ylabel('m w.e. per hydrological year'); ax.axhline(0, color='#999', lw=0.6)
    ax.legend(fontsize=7, loc='lower left')
ax = axs[1, 1]
for key in ['D', 'A', 'G']:
    col = L.GROUP_COL[key]; ax.plot(t.index, t[col + '_mwe'], color=C[key], lw=1, label=f'{key} group')
ax.fill_between(t.index, t.combined_mwe - t.combined_mwe_errors, t.combined_mwe + t.combined_mwe_errors, color='#555', alpha=0.15, lw=0)
ax.plot(t.index, t.combined_mwe, color=C['comb'], lw=2.2, label='combined (this study)')
o = L.official_hy(reg); ax.plot(o.index, o.combined_mwe, color=C['off'], ls='--', lw=1.3, label='official combined')
ax.set_title('(d) unweighted mean of data groups', loc='left'); ax.legend(fontsize=7, loc='lower left', ncol=2); ax.axhline(0, color='#999', lw=0.6)
for a_ in axs[1]: a_.set_xlabel('hydrological year (ending 30 Sep)')
fig.suptitle('Iceland: thin lines = annualised input datasets, horizontal bars = trend datasets (rate per year)', fontsize=9, x=0.01, ha='left')
save(fig, 'fig02_workflow_iceland.png')

# ------------------------------------------------------------ 3 regional small multiples
fig, axs = plt.subplots(5, 4, figsize=(13, 13), sharex=True)
for i, r in enumerate(L.REGIONS):
    ax = axs.flat[i]; t = HY[r].loc[2000:2023]; o = L.official_hy(r)
    for key in ['D', 'A', 'G']:
        col = L.GROUP_COL[key]
        if col + '_mwe' in t and t[col + '_mwe'].notna().any():
            ax.plot(t.index, t[col + '_mwe'], color=C[key], lw=0.9, alpha=0.9)
    ax.fill_between(t.index, t.combined_mwe - t.combined_mwe_errors, t.combined_mwe + t.combined_mwe_errors, color='#555', alpha=0.15, lw=0)
    ax.plot(t.index, t.combined_mwe, color=C['comb'], lw=1.8)
    ax.plot(o.index, o.combined_mwe, color=C['off'], ls='--', lw=1.1)
    ax.axhline(0, color='#999', lw=0.5)
    rr = R.iloc[i]
    ax.set_title(f'{i+1} {L.LABELS[i]}\n{rr.rate_mwe:.2f}±{rr.rate_mwe_err:.2f} m w.e./yr', loc='left', fontsize=8)
ax = axs.flat[19]; ax.axis('off')
for key, lab in [('D', 'DEM diff. + glaciological'), ('A', 'altimetry'), ('G', 'gravimetry')]:
    ax.plot([], [], color=C[key], label=lab)
ax.plot([], [], color=C['comb'], lw=1.8, label='combined (this study) ±1σ'); ax.plot([], [], color=C['off'], ls='--', label='official GlaMBIE combined')
ax.legend(loc='center', fontsize=9)
fig.text(0.0, 0.5, 'specific mass change (m w.e. per hydrological year)', rotation=90, va='center')
save(fig, 'fig03_regional_small_multiples.png')

# ------------------------------------------------------------ 4 global
fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
x = G.index.values
ax[0].bar(x, G['gt'], color=C['D'], width=0.7, label='this study (±1σ)', edgecolor='white', linewidth=1)
ax[0].errorbar(x, G['gt'], yerr=G.gt_err, fmt='none', ecolor='#333', lw=0.8, capsize=2)
ax[0].plot(x, G.official_gt, 'o', color=C['off'], ms=4, label='official GlaMBIE')
for y in [2023, 2019, 2022]:
    ax[0].annotate(str(y), (y, G['gt'][y] - G.gt_err[y]), xytext=(0, -10), textcoords='offset points', ha='center', fontsize=7)
ax[0].plot([2000, 2011.4], [S['global']['rate_2000_2011']] * 2, color=C['A'], lw=2, label='12-yr means')
ax[0].plot([2011.6, 2023.4], [S['global']['rate_2012_2023']] * 2, color=C['A'], lw=2)
ax[0].set_ylabel('global glacier mass change (Gt per calendar year)'); ax[0].set_title('(a) annual mass change', loc='left'); ax[0].legend(fontsize=8, loc='lower left')
ax[0].set_ylim(-720, 50)
ax[1].fill_between(x, G.cum_gt - G.cum_err, G.cum_gt + G.cum_err, color=C['D'], alpha=0.2, lw=0)
ax[1].plot(x, G.cum_gt, color=C['D'], lw=2, label='this study (±1σ, quadrature)')
ax[1].plot(x, G.official_cum_gt, '--', color=C['off'], label='official GlaMBIE')
ax[1].set_ylabel('cumulative mass change since 1 Jan 2000 (Gt)'); ax[1].set_title('(b) cumulative mass change', loc='left')
ax[1].text(2023, G.cum_gt[2023] + 400, f"{G.cum_gt[2023]:.0f} ± {G.cum_err[2023]:.0f} Gt\n= {G.cum_slr_mm[2023]:.1f} ± {G.cum_err[2023]/362.5:.1f} mm SLE", ha='right', fontsize=8)
ax[1].legend(fontsize=8, loc='lower left')
save(fig, 'fig04_global_series.png')

# ------------------------------------------------------------ 5 regional rates & relative loss
fig, ax = plt.subplots(1, 3, figsize=(15, 5.4), gridspec_kw=dict(wspace=0.55))
o = np.arange(19)
ax[0].barh(o, R.rate_gt, xerr=R.rate_gt_err, color=C['D'], edgecolor='white', error_kw=dict(lw=0.8, ecolor='#333'))
ax[0].plot(R.official_rate_gt, o, 'o', color=C['off'], ms=3.5, label='official')
ax[0].set_yticks(o); ax[0].set_yticklabels([f'{i+1} {L.LABELS[i]}' for i in range(19)]); ax[0].invert_yaxis()
ax[0].set_xlabel('mass change rate 2000–2023 (Gt/yr)'); ax[0].set_title('(a) total mass change rate', loc='left'); ax[0].legend(fontsize=8)
for i, v in enumerate(R.share_pct):
    ax[0].text(1, i, f'{v:.1f}%', va='center', ha='left', fontsize=7)
ax[0].set_xlim(right=12); ax[0].text(1, -1.1, 'share', fontsize=7)
ax[1].barh(o, R.rate_mwe, xerr=R.rate_mwe_err, color=C['V'], edgecolor='white', error_kw=dict(lw=0.8, ecolor='#333'))
ax[1].plot(R.official_rate_mwe, o, 'o', color=C['off'], ms=3.5)
ax[1].set_yticks(o); ax[1].set_yticklabels(SHORT); ax[1].invert_yaxis()
ax[1].set_xlabel('specific mass change 2000–2023 (m w.e./yr)'); ax[1].set_title('(b) specific rate', loc='left')
Rs = R.sort_values('loss_pct', ascending=True)
ax[2].barh(np.arange(19), Rs.loss_pct, xerr=Rs.loss_pct_err, color=C['A'], edgecolor='white', error_kw=dict(lw=0.8, ecolor='#333'))
ax[2].set_yticks(np.arange(19)); ax[2].set_yticklabels([L.LABELS[i - 1] for i in Rs.rid])
ax[2].set_xlabel('cumulative 2000–2023 loss (% of ~2000 ice mass)'); ax[2].set_title('(c) relative loss (mass: Zemp et al. 2019 vol. × 0.9)', loc='left')
for i, v in enumerate(Rs.loss_pct): ax[2].text(v + 1.5, i, f'{v:.0f}%', va='center', fontsize=7)
save(fig, 'fig05_regional_rates_relative_loss.png')

# ------------------------------------------------------------ 6 forest plot (method consistency)
F = pd.read_csv(os.path.join(OUT, 'method_forest.csv'))
F = F[F.window != '2000-2023 (D only)']
regs = [r for r in L.REGIONS if r in set(F.region)]
fig, axs = plt.subplots(1, 2, figsize=(12, 7.5), gridspec_kw=dict(wspace=0.45))
for ax, wn in zip(axs, ['altimetry period', 'GRACE period']):
    yy = 0; ticks, labs = [], []
    for r in regs:
        sub = F[(F.region == r) & (F.window == wn)]
        if sub.empty: continue
        for j, (_, row) in enumerate(sub.iterrows()):
            ax.errorbar(row.rate, yy - 0.22 * (j - 1), xerr=row.err_corr, fmt='o', color=C[row.group], ms=5, lw=1.2, capsize=2)
        ticks.append(yy); labs.append(f'{L.LABELS[L.rid_of(r)-1]}\n{sub.y0.iloc[0]}–{sub.y1.iloc[0]}'); yy -= 1
    ax.set_yticks(ticks); ax.set_yticklabels(labs, fontsize=7); ax.axvline(0, color='#999', lw=0.6)
    ax.set_xlabel('mean specific mass change over common years (m w.e./yr)')
    ax.set_title(f'({"ab"[axs.tolist().index(ax)]}) common {wn} (HY)', loc='left'); ax.tick_params(labelleft=True)
for key, lab in [('D', 'DEM diff. + glaciological'), ('A', 'altimetry'), ('G', 'gravimetry')]:
    axs[1].errorbar([], [], xerr=[], fmt='o', color=C[key], label=lab)
axs[1].legend(fontsize=8, loc='lower right')
fig.text(0.01, -0.01, 'Error bars: mean annual group uncertainty (fully correlated in time, conservative).', fontsize=7)
save(fig, 'fig06_method_forest.png')

# ------------------------------------------------------------ 7 z-test & bias vs D
Z = pd.read_csv(os.path.join(OUT, 'zscores.csv')); B = pd.read_csv(os.path.join(OUT, 'method_bias_vs_D.csv'))
fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))
bins = np.linspace(-5, 5, 41)
ax[0].hist(Z.z, bins=bins, density=True, color=C['D'], alpha=0.7, edgecolor='white')
xx = np.linspace(-5, 5, 200); ax[0].plot(xx, np.exp(-xx ** 2 / 2) / np.sqrt(2 * np.pi), color=C['comb'], lw=1.5, label='N(0,1)')
ax[0].set_title(f"(a) z of annual between-group differences\nn={len(Z)}, SD={Z.z.std():.2f}", loc='left'); ax[0].legend(); ax[0].set_xlabel('z')
sd = Z.groupby('region').z.std().reindex([r for r in L.REGIONS if r in set(Z.region)])
ax[1].bar(range(len(sd)), sd.values, color=C['V'], edgecolor='white'); ax[1].axhline(1, color=C['comb'], lw=1)
ax[1].set_xticks(range(len(sd))); ax[1].set_xticklabels([SHORT[L.rid_of(r) - 1] for r in sd.index], rotation=90)
ax[1].set_title('(b) SD of z by region', loc='left')
for gk, mk in [('A', 'o'), ('G', 's')]:
    sub = B[B.group == gk]
    ax[2].errorbar([SHORT[L.rid_of(r) - 1] for r in sub.region], sub.bias, yerr=sub.bias_err, fmt=mk, color=C[gk], label=f'{gk} − D', capsize=2)
ax[2].axhline(0, color='#999', lw=0.7); ax[2].set_ylabel('mean bias vs DEM+glac group (m w.e./yr)')
ax[2].set_title('(c) group bias relative to D (common years)', loc='left'); ax[2].legend(); ax[2].tick_params(axis='x', rotation=90)
save(fig, 'fig07_uncertainty_ztest_bias.png')

# ------------------------------------------------------------ 8 sensitivity
sens = pd.read_csv(os.path.join(OUT, 'sensitivity.csv')).set_index('config').drop(index=['main'])
fig, ax = plt.subplots(1, 3, figsize=(14, 4.4), sharey=True)
yy = np.arange(len(sens))
ax[0].barh(yy, sens.d_global_rate, color=C['D'], edgecolor='white'); ax[0].axvline(0, color='#333', lw=0.8)
ax[0].axvspan(-S['global']['rate_err'], S['global']['rate_err'], color='#ddd', zorder=0, label='±1σ of main rate')
ax[0].set_yticks(yy); ax[0].set_yticklabels(sens.index); ax[0].set_xlabel('Δ 2000–2023 mean rate vs main (Gt/yr)'); ax[0].legend(fontsize=7, loc='lower right')
ax[0].set_title('(a) mean rate', loc='left')
ax[1].barh(yy, 100 * (sens.ratio_late_early - 1), color=C['A'], edgecolor='white')
ax[1].axvline(100 * (S['sensitivity']['main']['ratio_late_early'] - 1), color='#333', lw=0.8, label='main')
ax[1].axvline(S['official']['increase_pct'], color=C['off'], ls='--', lw=1, label='official')
ax[1].set_xlabel('2012–23 vs 2000–11 increase (%)'); ax[1].legend(fontsize=7, loc='lower right', framealpha=1, frameon=True); ax[1].set_title('(b) acceleration', loc='left')
ax[2].barh(yy, sens.cal_rmsd_gt, color=C['G'], edgecolor='white')
ax[2].axvline(S['sensitivity']['main']['cal_rmsd_gt'], color='#333', lw=0.8)
ax[2].set_xlabel('RMSD vs official global calendar series (Gt)'); ax[2].set_title('(c) reproduction error', loc='left')
save(fig, 'fig08_sensitivity.png')

# ------------------------------------------------------------ 9 ice sheets
ice = pd.read_csv(os.path.join(OUT, 'ice_sheet_comparison.csv'))
fig, ax = plt.subplots(figsize=(9, 4.2))
xx = np.arange(len(ice)); w = 0.26
ax.bar(xx - w, -ice.glaciers, w, yerr=ice.glaciers_e, color=C['D'], label='glaciers (this study)', edgecolor='white', error_kw=dict(lw=0.8))
ax.bar(xx, -ice.gris, w, yerr=ice.gris_e_1s, color=C['A'], label='Greenland Ice Sheet', edgecolor='white', error_kw=dict(lw=0.8))
ax.bar(xx + w, -ice.ais, w, yerr=ice.ais_e_1s, color=C['G'], label='Antarctic Ice Sheet', edgecolor='white', error_kw=dict(lw=0.8))
ax.set_xticks(xx); ax.set_xticklabels(ice.period); ax.set_ylabel('mass loss rate (Gt/yr, ±1σ)')
for i, rw in ice.iterrows():
    ax.text(i - w, -rw.glaciers + rw.glaciers_e + 8, f'{rw.share_of_landice_pct:.0f}%', ha='center', fontsize=7)
ax.set_title('Glacier vs ice-sheet mass loss; % = glacier share of land-ice loss (ice sheets: Hugonnet et al. 2021, Table 1)', loc='left', fontsize=9)
ax.legend(fontsize=8, loc='upper left')
save(fig, 'fig09_ice_sheet_comparison.png')

# ------------------------------------------------------------ 10 projections
P = S['projections']
fig, ax = plt.subplots(1, 2, figsize=(13, 5))
cum = G.cum_slr_mm; cum15 = cum - cum[2014]
ax[0].fill_between(G.index, (G.cum_gt - G.cum_err) / -362.5, (G.cum_gt + G.cum_err) / -362.5, color=C['comb'], alpha=0.15, lw=0)
ax[0].plot(G.index, cum, color=C['comb'], lw=2.2, label='observed (this study), from 2000')
obs15 = cum[2014]
scen = [('rounce2023', 'SSP1-2.6', C['D']), ('rounce2023', 'SSP2-4.5', C['V']), ('rounce2023', 'SSP5-8.5', C['R']),
        ('marzeion2020', 'RCP2.6', C['G']), ('marzeion2020', 'RCP8.5', C['A']), ('hock2019', 'RCP2.6', C['H']), ('hock2019', 'RCP8.5', C['M'])]
for j, (m, s, col) in enumerate(scen):
    v, e = P[m]['slr_2100_mm'][s]
    xe = 2100 + 1.6 * j
    ax[0].plot([2015, 2100], [obs15, obs15 + v], ls=':', color=col, lw=1)
    ax[0].errorbar(xe, obs15 + v, yerr=e, fmt='o', color=col, capsize=2, ms=5, label=f"{m[:-4].capitalize()} {m[-4:]} {s}: {v}±{e} mm ({P[m]['ci']})")
# observed-rate extrapolations
r_late = -S['global']['rate_2012_2023'] / 362.5
yy = np.arange(2023, 2101)
ax[0].plot(yy, cum[2023] + r_late * (yy - 2023), color=C['comb'], ls='--', lw=1.2, label=f'constant 2012–2023 rate ({r_late:.2f} mm/yr)')
acc = -S['global']['accel_gt_yr2'] / 362.5
ax[0].plot(yy, cum[2023] + r_late * (yy - 2023) + 0.5 * acc * ((yy - 2017.5) ** 2 - (2023 - 2017.5) ** 2), color=C['comb'], ls='-.', lw=1,
           label=f'continued OLS acceleration ({acc*10:.2f} mm/yr per decade)')
ax[0].set_xlim(1998, 2113); ax[0].set_ylabel('cumulative glacier sea-level contribution (mm SLE)')
ax[0].set_title('(a) cumulative SLE: observed vs 2015–2100 projections\n(projections anchored at observed end-2014 value)', loc='left', fontsize=9)
ax[0].legend(fontsize=6.5, loc='upper left')
# rates
rate = -G['gt'] / 362.5
ax[1].bar(G.index, rate, color=C['D'], width=0.95, edgecolor='white', label='observed annual (this study)')
ax[1].errorbar(G.index, rate, yerr=G.gt_err / 362.5, fmt='none', ecolor='#333', lw=0.7)
ax[1].plot([2015, 2023], [-S['global']['rate_2015_2019'] * 0 + S['proj_compare']['obs_rate_2015_2023_mm']] * 2, color=C['comb'], lw=2, label=f"observed mean 2015–2023 ({S['proj_compare']['obs_rate_2015_2023_mm']:.2f} mm/yr)")
pr = P['rounce2023']
ax[1].errorbar(2035, pr['peak_rate_1p5C']['mm_yr'], yerr=pr['peak_rate_1p5C']['err'], fmt='D', color=C['D'], capsize=3, label='Rounce +1.5°C peak rate ~2035 (95% CI)')
ax[1].errorbar(2099, pr['end_century_rate_mm_yr'][0], yerr=pr['end_century_rate_mm_yr'][1], fmt='v', color=C['G'], capsize=3, label='Rounce end-century rate +1.5°C')
ax[1].errorbar(2101, pr['end_century_rate_mm_yr'][2], yerr=pr['end_century_rate_mm_yr'][3], fmt='^', color=C['R'], capsize=3, label='Rounce end-century rate +4°C')
ax[1].plot([2015, 2100], [P['marzeion2020']['rate_rcp26_peak_mm_yr']] * 2, color=C['G'], ls=':', lw=1.2, label='Marzeion RCP2.6 peak (~1 mm/yr, first half)')
ax[1].scatter([2103, 2103], P['hock2019']['rate_2100_mm_yr'], marker='s', color=C['H'], zorder=3, label='Hock 2019 rates in 2100 (RCP2.6 / RCP8.5)')
for m, s, col in [('rounce2023', 'SSP1-2.6', C['D']), ('rounce2023', 'SSP5-8.5', C['R'])]:
    ax[1].hlines(P[m]['slr_2100_mm'][s][0] / 85, 2015, 2100, colors=col, linestyles='--', lw=1, label=f'mean 2015–2100 rate, Rounce {s}')
ax[1].set_xlim(1998, 2106); ax[1].set_ylabel('glacier sea-level contribution rate (mm SLE/yr)')
ax[1].set_title('(b) observed annual rates vs projected rates', loc='left', fontsize=9); ax[1].legend(fontsize=6.5, loc='upper left')
save(fig, 'fig10_observations_vs_projections.png')

# ------------------------------------------------------------ 11 validation
V = pd.read_csv(os.path.join(OUT, 'validation_regional_hy.csv'))
cw = pd.read_csv(os.path.join(OUT, 'calendar_weights.csv'))
og = pd.read_csv('data/glambie/results/calendar_years/0_global.csv'); og.index = np.round(og.start_dates).astype(int)
fig, ax = plt.subplots(1, 3, figsize=(14, 4.3))
allm, allo = [], []
for r in L.REGIONS:
    t = HY[r]; o = L.official_hy(r)
    j = pd.concat([t.combined_mwe, o.combined_mwe.rename('o')], axis=1).dropna(); allm += list(j.combined_mwe); allo += list(j.o)
ax[0].scatter(allo, allm, s=6, color=C['D'], alpha=0.6)
lim = [-4.5, 2]; ax[0].plot(lim, lim, color='#333', lw=0.8); ax[0].set_xlim(lim); ax[0].set_ylim(lim)
ax[0].set_xlabel('official combined (m w.e., HY)'); ax[0].set_ylabel('this study (m w.e., HY)')
ax[0].set_title(f'(a) 19 regions × HY, r={np.corrcoef(allo, allm)[0,1]:.3f}', loc='left')
d = G['gt'] - og.combined_gt
ax[1].bar(G.index, d, color=C['V'], edgecolor='white'); ax[1].fill_between(G.index, -og.combined_gt_errors, og.combined_gt_errors, color='#ddd', zorder=0, label='official ±1σ')
ax[1].set_ylabel('this study − official (Gt)'); ax[1].set_title(f"(b) global calendar years, RMSD {S['validation']['cal_rmsd']:.0f} Gt", loc='left'); ax[1].legend(fontsize=8)
ax[2].scatter(cw.w_official_implied, cw.w_sinewave, color=C['A'], s=20)
for _, rw in cw.iterrows():
    if L.rid_of(rw.region) != 16: ax[2].annotate(SHORT[L.rid_of(rw.region) - 1], (rw.w_official_implied, rw.w_sinewave), fontsize=6, xytext=(2, 2), textcoords='offset points')
ax[2].plot([0, 0.8], [0, 0.8], color='#333', lw=0.8); ax[2].set_xlabel('weight implied by official HY/CY pairs'); ax[2].set_ylabel('weight from glaciological sine-wave product')
ax[2].set_title('(c) calendar-conversion weight w', loc='left')
save(fig, 'fig11_validation.png')
print('figures done')
