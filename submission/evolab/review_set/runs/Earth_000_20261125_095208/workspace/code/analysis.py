"""Post-processing of the main reconciliation: global & regional statistics, ice-sheet comparison,
relative loss, method consistency, uncertainty z-test, hybrid validation, sensitivity, validation
against official results, projection comparison.  Writes outputs/summary_stats.json and CSV tables."""
import sys, os, json, numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, 'code'); import glambie_lib as L

OUT = 'outputs'; CFG = os.path.join(OUT, 'configs')
S = {}
yrs = np.arange(2000, 2024)

def load_cfg(name):
    G = pd.read_csv(os.path.join(CFG, name, 'global_calendar.csv')).set_index('year')
    cy = {r: pd.read_csv(os.path.join(CFG, name, f'{r}_calendar.csv')).set_index('year') for r in L.REGIONS}
    hy = {r: pd.read_csv(os.path.join(CFG, name, f'{r}_hydro.csv')).set_index('hy') for r in L.REGIONS}
    return G, cy, hy

G, CY, HY = load_cfg('main')
og = pd.read_csv('data/glambie/results/calendar_years/0_global.csv'); og.index = np.round(og.start_dates).astype(int)

def period_rate(series, err, a, b):
    s = series.loc[a:b]; e = err.loc[a:b]
    return s.mean(), np.sqrt((e ** 2).sum()) / len(s)

# ------------------------------------------------------------------ 1 global
g, ge = G['gt'], G['gt_err']
tot = g.sum(); tot_e = np.sqrt((ge ** 2).sum()); tot_e_corr = ge.sum()
S['global'] = dict(total_gt=tot, total_err_quad=tot_e, total_err_fullcorr=tot_e_corr,
                   rate_gt=tot / 24, rate_err=tot_e / 24,
                   total_slr_mm=-tot / L.GT_PER_MM, total_slr_err=tot_e / L.GT_PER_MM,
                   rate_slr_mm=-tot / 24 / L.GT_PER_MM, rate_slr_err=tot_e / 24 / L.GT_PER_MM,
                   mwe_rate=G['mwe'].mean(), mwe_total=G['mwe'].sum(),
                   mwe_total_err=np.sqrt((G['mwe_err'] ** 2).sum()))
r1, e1 = period_rate(g, ge, 2000, 2011); r2, e2 = period_rate(g, ge, 2012, 2023)
inc = r2 / r1 - 1; inc_e = abs(r2 / r1) * np.sqrt((e1 / r1) ** 2 + (e2 / r2) ** 2)
S['global'].update(rate_2000_2011=r1, err_2000_2011=e1, rate_2012_2023=r2, err_2012_2023=e2,
                   increase_pct=100 * inc, increase_pct_err=100 * inc_e)
for a, b in [(2000, 2004), (2005, 2009), (2010, 2014), (2015, 2019), (2020, 2023), (2000, 2019), (2000, 2018), (2003, 2018)]:
    r, e = period_rate(g, ge, a, b); S['global'][f'rate_{a}_{b}'] = r; S['global'][f'err_{a}_{b}'] = e
# OLS acceleration (Gt/yr^2) with residual-based and error-weighted SE
X = np.column_stack([np.ones(24), yrs - yrs.mean()])
beta, res, *_ = np.linalg.lstsq(X, g.values, rcond=None)
sig2 = np.sum((g.values - X @ beta) ** 2) / 22
se = np.sqrt(sig2 / np.sum((yrs - yrs.mean()) ** 2))
W = 1 / ge.values ** 2
bw = np.polyfit(yrs, g.values, 1, w=np.sqrt(W))[0]
S['global'].update(accel_gt_yr2=beta[1], accel_se=se, accel_wls=bw,
                   accel_gt_yr_per_decade=10 * beta[1], accel_se_decade=10 * se)
# official for reference
ogt = og.combined_gt
o1, o2 = ogt.loc[2000:2011].mean(), ogt.loc[2012:2023].mean()
ob = np.polyfit(yrs, ogt.values, 1)[0]
S['official'] = dict(total_gt=ogt.sum(), total_err_quad=np.sqrt((og.combined_gt_errors ** 2).sum()), rate_gt=ogt.mean(),
                     rate_2000_2011=o1, rate_2012_2023=o2, increase_pct=100 * (o2 / o1 - 1), accel_gt_yr2=ob,
                     gt_2023=ogt.loc[2023], total_mwe=og.combined_mwe.sum())
rank = g.sort_values().index.tolist()
S['global']['record_years'] = [dict(year=int(y), gt=g[y], err=ge[y], slr_mm=-g[y] / L.GT_PER_MM) for y in rank[:5]]
S['global']['least_negative_years'] = [dict(year=int(y), gt=g[y]) for y in rank[-3:]]
S['global']['official_rank_2023'] = int(ogt.rank().loc[2023])
S['global']['gt_2023'] = g[2023]; S['global']['err_2023'] = ge[2023]
S['global']['mwe_2023'] = G['mwe'][2023]
S['global']['n_years_below_minus300'] = int((g < -300).sum())
S['global']['years_below_minus300_after_2012'] = int((g.loc[2012:] < -300).sum())
S['global']['area_2000'] = G['area'][2000]; S['global']['area_2023'] = G['area'][2023]
# Hugonnet 2021 comparison (paper_004 abstract: 2000-2019 267 +/- 16 Gt/yr)
S['global']['hugonnet_2000_2019'] = -267.0
G2 = G.copy(); G2['cum_gt'] = g.cumsum(); G2['cum_err'] = np.sqrt((ge ** 2).cumsum()); G2['cum_slr_mm'] = -G2['cum_gt'] / L.GT_PER_MM
G2['official_gt'] = ogt; G2['official_err'] = og.combined_gt_errors; G2['official_cum_gt'] = ogt.cumsum()
G2.to_csv(os.path.join(OUT, 'global_series_main.csv'))

# ------------------------------------------------------------------ 2 ice sheets (Hugonnet et al. 2021, paper_004 Table 1, p.3)
ice = pd.DataFrame([
    dict(period='2000-2004', a=2000, b=2004, gris=-94, gris_e=65, ais=-36, ais_e=118, hug=-227, hug_e=25),
    dict(period='2005-2009', a=2005, b=2009, gris=-206, gris_e=56, ais=-93, ais_e=104, hug=-257, hug_e=22),
    dict(period='2010-2014', a=2010, b=2014, gris=-267, gris_e=57, ais=-214, ais_e=94, hug=-284, hug_e=23),
    dict(period='2015-2018', a=2015, b=2018, gris=-152, gris_e=64, ais=-157, ais_e=87, hug=-292, hug_e=24),
    dict(period='2000-2018', a=2000, b=2018, gris=-181, gris_e=31, ais=-121, ais_e=104, hug=-264, hug_e=16),
    dict(period='2003-2018', a=2003, b=2018, gris=-205, gris_e=32, ais=-143, ais_e=104, hug=-272, hug_e=16)])
# 95% CI in Hugonnet Table 1 -> 1 sigma
for c in ['gris_e', 'ais_e', 'hug_e']:
    ice[c + '_1s'] = ice[c] / 1.96
gl = [period_rate(g, ge, a, b) for a, b in zip(ice.a, ice.b)]
ice['glaciers'] = [x[0] for x in gl]; ice['glaciers_e'] = [x[1] for x in gl]
ice['ratio_gris'] = ice.glaciers / ice.gris; ice['ratio_ais'] = ice.glaciers / ice.ais
ice['ratio_gris_e'] = ice.ratio_gris * np.sqrt((ice.glaciers_e / ice.glaciers) ** 2 + (ice.gris_e_1s / ice.gris) ** 2)
ice['ratio_ais_e'] = ice.ratio_ais * np.sqrt((ice.glaciers_e / ice.glaciers) ** 2 + (ice.ais_e_1s / ice.ais) ** 2)
ice['share_of_landice_pct'] = 100 * ice.glaciers / (ice.glaciers + ice.gris + ice.ais)
ice.to_csv(os.path.join(OUT, 'ice_sheet_comparison.csv'), index=False)
S['ice_sheets'] = ice.set_index('period')[['glaciers', 'glaciers_e', 'gris', 'gris_e_1s', 'ais', 'ais_e_1s', 'ratio_gris', 'ratio_gris_e', 'ratio_ais', 'ratio_ais_e', 'share_of_landice_pct']].to_dict('index')
S['ice_sheets']['source'] = 'Hugonnet et al. 2021 (related_work/paper_004), Table 1 p.3: IMBIE minus half of peripheral glaciers; 95% CI converted to 1 sigma; AIS end date June 2017'

# ------------------------------------------------------------------ 3 regional
rows = []
for k, r in enumerate(L.REGIONS):
    c = CY[r]; h = HY[r]
    t = c['gt'].sum(); te = np.sqrt((c.gt_err ** 2).sum())
    mass0 = L.RGI_VOL[k] * 0.9          # Gt (volume km3 x 0.9 t/m3), Zemp et al. 2019 Table 1
    rows.append(dict(region=r, label=L.LABELS[k], rid=k + 1, area_rgi=L.RGI_AREA[k],
                     rate_gt=t / 24, rate_gt_err=te / 24, total_gt=t, total_gt_err=te,
                     rate_mwe=c['mwe'].mean(), rate_mwe_err=np.sqrt((c.mwe_err ** 2).sum()) / 24,
                     rate_gt_2000_2011=c['gt'].loc[2000:2011].mean(), rate_gt_2012_2023=c['gt'].loc[2012:2023].mean(),
                     rate_mwe_2000_2011=c['mwe'].loc[2000:2011].mean(), rate_mwe_2012_2023=c['mwe'].loc[2012:2023].mean(),
                     rate_gt_2015_2023=c['gt'].loc[2015:2023].mean(),
                     share_pct=100 * t / tot, mass_2000_gt=mass0, loss_pct=-100 * t / mass0,
                     loss_pct_err=100 * te / mass0, mass_2015_gt=mass0 + c['gt'].loc[2000:2014].sum(),
                     official_rate_gt=L.official_cy(r).combined_gt.mean(), official_rate_mwe=L.official_cy(r).combined_mwe.mean(),
                     official_rate_gt_err=np.sqrt((L.official_cy(r).combined_gt_errors ** 2).sum()) / 24,
                     n_groups_max=int(h.n_groups.max())))
R = pd.DataFrame(rows)
R['loss_rank'] = R.loss_pct.rank(ascending=False).astype(int)
R['dev_from_official_sigma'] = (R.rate_gt - R.official_rate_gt) / np.sqrt(R.rate_gt_err ** 2 + R.official_rate_gt_err ** 2)
R['pct_per_yr_2015_2023'] = -100 * R.rate_gt_2015_2023 / R.mass_2015_gt
R.to_csv(os.path.join(OUT, 'regional_summary.csv'), index=False)
S['regional'] = R.set_index('region').to_dict('index')
can = R[R.rid.isin([3, 4])]
S['canadian_arctic'] = dict(share_pct=can.share_pct.sum(), rate_gt=can.rate_gt.sum(),
                            rate_gt_err=np.sqrt((can.rate_gt_err ** 2).sum()))
S['n_regions_within_1sigma_official'] = int((R.dev_from_official_sigma.abs() < 1).sum())
S['top_contributors'] = R.sort_values('total_gt').label.head(5).tolist()
S['global_mass_2000_gt'] = R.mass_2000_gt.sum(); S['global_loss_pct'] = -100 * tot / R.mass_2000_gt.sum()
S['global_loss_pct_err'] = 100 * tot_e / R.mass_2000_gt.sum()

# ------------------------------------------------------------------ 4 method consistency
grp = {'D': 'demdiff_and_glaciological', 'A': 'altimetry', 'G': 'gravimetry'}
frows, brows, zrows = [], [], []
for k, r in enumerate(L.REGIONS):
    h = HY[r].loc[2000:2023]
    avail = {g_: h[c + '_mwe'] for g_, c in grp.items() if c + '_mwe' in h}
    errs = {g_: h[c + '_mwe_errors'] for g_, c in grp.items() if c + '_mwe' in h}
    windows = {}
    if 'A' in avail:
        windows['altimetry period'] = avail['A'].dropna().index
    if 'G' in avail:
        windows['GRACE period'] = avail['G'].dropna().index
    windows['2000-2023 (D only)'] = avail['D'].dropna().index
    for wn, idx in windows.items():
        for g_ in avail:
            s = avail[g_].reindex(idx); e = errs[g_].reindex(idx)
            if s.notna().sum() < len(idx) or len(idx) < 3:
                continue
            frows.append(dict(region=r, label=L.LABELS[k], window=wn, y0=int(idx.min()), y1=int(idx.max()), n=len(idx),
                              group=g_, rate=s.mean(), err=np.sqrt((e ** 2).sum()) / len(idx), err_corr=e.mean()))
    for g_ in ['A', 'G']:
        if g_ not in avail:
            continue
        j = pd.concat([avail[g_], avail['D'], errs[g_], errs['D']], axis=1, keys=['x', 'd', 'ex', 'ed']).dropna()
        if len(j) < 3:
            continue
        brows.append(dict(region=r, label=L.LABELS[k], group=g_, n=len(j), y0=int(j.index.min()), y1=int(j.index.max()),
                          bias=(j.x - j.d).mean(), bias_err=np.sqrt(((j.ex ** 2 + j.ed ** 2)).sum()) / len(j),
                          rmsd=np.sqrt(((j.x - j.d) ** 2).mean()), r=np.corrcoef(j.x, j.d)[0, 1],
                          r_anom=np.corrcoef(j.x - j.x.mean(), j.d - j.d.mean())[0, 1]))
    # z-scores of differences between group pairs (annual)
    keys = list(avail)
    for i1 in range(len(keys)):
        for i2 in range(i1 + 1, len(keys)):
            a_, b_ = keys[i1], keys[i2]
            j = pd.concat([avail[a_], avail[b_], errs[a_], errs[b_]], axis=1, keys=['a', 'b', 'ea', 'eb']).dropna()
            for y, row in j.iterrows():
                zrows.append(dict(region=r, pair=f'{a_}-{b_}', year=y, z=(row.a - row.b) / np.sqrt(row.ea ** 2 + row.eb ** 2)))
F = pd.DataFrame(frows); F.to_csv(os.path.join(OUT, 'method_forest.csv'), index=False)
B = pd.DataFrame(brows); B.to_csv(os.path.join(OUT, 'method_bias_vs_D.csv'), index=False)
Z = pd.DataFrame(zrows); Z.to_csv(os.path.join(OUT, 'zscores.csv'), index=False)
S['zscore'] = dict(n=len(Z), sd=Z.z.std(), mean=Z.z.mean(), frac_abs_gt2=float((Z.z.abs() > 2).mean()),
                   by_pair=Z.groupby('pair').z.agg(['count', 'mean', 'std']).to_dict('index'))
S['zscore_by_region'] = Z.groupby('region').z.std().to_dict()
polar = [3, 4, 5, 7, 9, 19]
Bp = B[(B.group == 'A')]
S['method_bias'] = dict(
    alt_polar_mean_bias=B[(B.group == 'A') & B.region.map(L.rid_of).isin(polar)].bias.mean(),
    alt_nonpolar_mean_bias=B[(B.group == 'A') & ~B.region.map(L.rid_of).isin(polar)].bias.mean(),
    alt_mean_bias=B[B.group == 'A'].bias.mean(), grav_mean_bias=B[B.group == 'G'].bias.mean(),
    alt_median_r=B[B.group == 'A'].r.median(), grav_median_r=B[B.group == 'G'].r.median(),
    rows=B.set_index(['region', 'group'])[['bias', 'bias_err', 'rmsd', 'r', 'n']].round(3).reset_index().to_dict('records'))

# gravimetry outside the 7 official regions (GRACE leakage in small regions / HMA): compare grav_all vs D
cache = {r: L.load_region(r) for r in L.REGIONS}
grows = []
for r in L.REGIONS:
    if L.rid_of(r) in L.GRAV_REGIONS_OFFICIAL: continue
    t, _ = L.region_table(r, {'grav': 'all'}, cache[r]); t = t.set_index('hy').loc[2003:2022]
    if 'gravimetry_mwe' not in t: continue
    j = t[['gravimetry_mwe', 'demdiff_and_glaciological_mwe', 'gravimetry_mwe_errors']].dropna()
    if len(j) < 3: continue
    grows.append(dict(region=r, label=L.LABELS[L.rid_of(r) - 1], grav_rate=j.gravimetry_mwe.mean(), D_rate=j.demdiff_and_glaciological_mwe.mean(),
                      diff=(j.gravimetry_mwe - j.demdiff_and_glaciological_mwe).mean(), r=np.corrcoef(j.gravimetry_mwe, j.demdiff_and_glaciological_mwe)[0, 1],
                      grav_annual_err=j.gravimetry_mwe_errors.mean(), area=L.RGI_AREA[L.rid_of(r) - 1]))
GX = pd.DataFrame(grows); GX.to_csv(os.path.join(OUT, 'gravimetry_excluded_regions.csv'), index=False)
S['grav_excluded'] = GX.round(3).to_dict('records')

# glaciological representativeness: glaciological sample (Zemp sine-wave product) vs geodetic (Hugonnet) 2000-2019
rep = []
for r in L.REGIONS:
    ds = cache[r]
    hug = [d for d in ds if d['src'] == 'demdiff' and 'Hugonnet' in d['author']]
    gl = [d for d in ds if d['src'] == 'glaciological' and 'Zemp' in d['author']]
    wg = [d for d in ds if d['src'] == 'glaciological' and 'WGMS' in d['author']]
    if not hug: continue
    hrate = hug[0]['data'].v.sum() / 20
    row = dict(region=r, label=L.LABELS[L.rid_of(r) - 1], geodetic_rate=hrate)
    if gl:
        a, _ = L.annualise_monthly(gl[0])
        sel = a.loc[2001:2020] if L.rid_of(r) <= 15 else a.loc[2000:2019]
        row['glac_sinewave_rate'] = sel.v.mean()
    if wg:
        a = L.annual_from_annual(wg[0]); row['wgms_beta_mean'] = a.loc[2001:2020].v.mean() if a is not None else np.nan
        # inter-annual agreement of WGMS anomalies with the altimetry/gravimetry groups (independent variability)
        h = HY[r]
        for g_, c in [('A', 'altimetry'), ('G', 'gravimetry')]:
            if c + '_mwe' in h and a is not None:
                j = pd.concat([a.v, h[c + '_mwe']], axis=1).dropna()
                if len(j) >= 5: row[f'r_wgms_{g_}'] = np.corrcoef(j.iloc[:, 0], j.iloc[:, 1])[0, 1]
    rep.append(row)
REP = pd.DataFrame(rep); REP['glac_minus_geod'] = REP.glac_sinewave_rate - REP.geodetic_rate
REP.to_csv(os.path.join(OUT, 'glaciological_representativeness.csv'), index=False)
S['glac_rep'] = dict(mean_diff=REP.glac_minus_geod.mean(), mean_abs_diff=REP.glac_minus_geod.abs().mean(),
                     wgms_beta_mean_of_means=REP.wgms_beta_mean.mean(), wgms_beta_range=[REP.wgms_beta_mean.min(), REP.wgms_beta_mean.max()],
                     median_r_wgms_A=REP.get('r_wgms_A', pd.Series(dtype=float)).median(),
                     median_r_wgms_G=REP.get('r_wgms_G', pd.Series(dtype=float)).median())

# ------------------------------------------------------------------ 5 hybrid validation (hybrids withheld)
_, _, HYX = load_cfg('hyb_exclude')
hv = []
for r in L.REGIONS:
    hx = HYX[r]
    k_ = L.RGI_AREA[L.rid_of(r) - 1] / hx.area
    for d in cache[r]:
        if d['src'] != 'combined': continue
        a = L.annual_from_annual(d) if d['kind'] == 'annual' else None
        if a is None or len(a) < 3: continue
        j = pd.concat([a.v * k_.reindex(a.index), hx.combined_mwe], axis=1, keys=['h', 'c']).loc[2000:2023].dropna()
        if len(j) < 3: continue
        hv.append(dict(region=r, dataset=d['name'].split('_combined_')[1], n=len(j), diff=(j.h - j.c).mean(), r=np.corrcoef(j.h, j.c)[0, 1]))
HV = pd.DataFrame(hv); HV.to_csv(os.path.join(OUT, 'hybrid_validation.csv'), index=False)
S['hybrid_validation'] = dict(n_series=len(HV), mean_diff=HV['diff'].mean(), median_r=HV.r.median(),
                              by_dataset=HV.groupby(HV.dataset.str.split('_').str[-3]).agg(diff=('diff', 'mean'), r=('r', 'median')).round(3).to_dict('index'))

# ------------------------------------------------------------------ 6 sensitivity table
met = pd.read_csv(os.path.join(OUT, 'config_metrics.csv')).set_index('config')
sens = met[['global_rate', 'global_total_gt', 'rate_2000_2011', 'rate_2012_2023', 'ratio_late_early', 'gt_2023', 'cal_rmsd_gt', 'cal_r', 'reg_mean_rmsd_gt', 'err_ratio_global']].copy()
for c in ['global_rate', 'global_total_gt', 'rate_2000_2011', 'rate_2012_2023', 'gt_2023']:
    sens['d_' + c] = sens[c] - sens.loc['main', c]
sens.to_csv(os.path.join(OUT, 'sensitivity.csv'))
S['sensitivity'] = sens.round(3).to_dict('index')

# ------------------------------------------------------------------ 7 validation vs official (regional HY)
from compare_hy import compare
V = compare({}, cache); V.to_csv(os.path.join(OUT, 'validation_regional_hy.csv'), index=False)
Vc = V[V.group == 'combined']
S['validation'] = dict(combined_mean_abs_bias=Vc.bias.abs().mean(), combined_median_rmsd=Vc.rmsd.median(), combined_median_r=Vc.r.median(),
                       combined_median_err_ratio=Vc.err_ratio.median(),
                       group_err_ratio_median=V[V.group != 'combined'].groupby('group').err_ratio.median().to_dict(),
                       D_bias_CEU=float(V[(V.region == '11_central_europe') & (V.group == 'demdiff_and_glaciological')].bias.iloc[0]),
                       D_bias_NZL=float(V[(V.region == '18_new_zealand') & (V.group == 'demdiff_and_glaciological')].bias.iloc[0]),
                       cal_rmsd=met.loc['main', 'cal_rmsd_gt'], cal_r=met.loc['main', 'cal_r'], cal_maxabs=met.loc['main', 'cal_maxabs_gt'],
                       cal_diff_2020=g[2020] - ogt[2020], cal_diff_2023=g[2023] - ogt[2023],
                       err_ratio_global=float((ge / og.combined_gt_errors).mean()),
                       multi_year_err_check=dict(official_reported_annual_quadrature=S['official']['total_err_quad']))
cw = pd.read_csv(os.path.join(OUT, 'calendar_weights.csv'))
S['calendar_weights'] = cw.round(3).to_dict('records')
S['calendar_weights_rmsd_sine_vs_official'] = float(np.sqrt(((cw.w_sinewave - cw.w_official_implied) ** 2).mean()))

# ------------------------------------------------------------------ 8 projections (numbers transcribed from related_work, with source)
P = dict(
    rounce2023=dict(src='Rounce et al. 2023 (paper_000), p.1 abstract & text; Table 1 p.5', base_year=2015,
                    slr_2100_mm={'+1.5C': [90, 26], '+2C': [99, 31], '+3C': [125, 39], '+4C': [154, 44], 'SSP1-2.6': [98, 38], 'SSP2-4.5': [116, 51], 'SSP5-8.5': [166, 83], 'RCP2.6': [90, 36], 'RCP4.5': [114, 44], 'RCP8.5': [163, 53]},
                    mass_loss_pct_2100={'+1.5C': [26, 6], '+4C': [41, 11], 'SSP1-2.6': [28, 9], 'SSP2-4.5': [32, 12], 'SSP5-8.5': [44, 20], 'RCP2.6': [26, 8], 'RCP8.5': [43, 13]},
                    peak_rate_1p5C=dict(mm_yr=1.29, err=0.59, year=2035), end_century_rate_mm_yr=[0.70, 0.45, 2.23, 1.08],
                    slr_2100_uncorrected_mm={'RCP2.6': [106, 37], 'RCP4.5': [132, 47], 'RCP8.5': [187, 61], 'SSP1-2.6': [115, 42], 'SSP2-4.5': [135, 57], 'SSP5-8.5': [192, 97]},
                    ci='95%'),
    marzeion2020=dict(src='Marzeion et al. 2020 (paper_001), Table 2 p.21; text p.13-14', base_year=2015,
                      slr_2100_mm={'RCP2.6': [79, 56], 'RCP4.5': [119, 66], 'RCP6.0': [111, 75], 'RCP8.5': [159, 86]},
                      mass_loss_pct_2100={'RCP2.6': [18, 13], 'RCP4.5': [27, 15], 'RCP6.0': [25, 17], 'RCP8.5': [36, 20]},
                      rate_rcp26_peak_mm_yr=1.0, rate_rcp26_2100=0.7, diseq_2005_2014_kg_m2_yr=[-533, 585],
                      mass_2015_mm_sle=439, consensus_mass_mm_sle=[382, 99], ci='90%'),
    hock2019=dict(src='Hock et al. 2019 (paper_003), abstract p.1; Table 3 p.6; text p.8-9; values recalculated in Marzeion 2020 Table 2', base_year=2015,
                  slr_2100_mm={'RCP2.6': [89, 42], 'RCP4.5': [124, 58], 'RCP6.0': [140, 52], 'RCP8.5': [180, 74]},
                  mass_loss_pct_2100={'RCP2.6': [18, 7], 'RCP4.5': [27, 8], 'RCP6.0': [24, 6], 'RCP8.5': [36, 11]},
                  rate_2100_mm_yr=[0.7, 3.2], regional_loss_pct_2100={'ACN': {'RCP2.6': [12, 8], 'RCP8.5': [23, 15]}, 'CEU': {'RCP2.6': [69, 19], 'RCP8.5': [93, 10]}},
                  ci='1 SD'))
S['projections'] = P
# observation-based comparisons
obs_rate_2015_2023 = g.loc[2015:2023].mean()
S['proj_compare'] = dict(
    obs_rate_2015_2023_gt=obs_rate_2015_2023, obs_rate_2015_2023_mm=-obs_rate_2015_2023 / L.GT_PER_MM,
    obs_cum_2015_2023_mm=-g.loc[2015:2023].sum() / L.GT_PER_MM,
    obs_mass_2015_gt=R.mass_2015_gt.sum(), obs_pct_lost_2015_2023=-100 * g.loc[2015:2023].sum() / R.mass_2015_gt.sum(),
    mean_rate_2015_2100_mm={f'{m}:{s}': v[0] / 85 for m, d in P.items() for s, v in d['slr_2100_mm'].items()},
    years_to_reach_by_obs_rate={f'{m}:{s}': v[0] / (-obs_rate_2015_2023 / L.GT_PER_MM) for m, d in P.items() for s, v in d['slr_2100_mm'].items()},
    obs_specific_2005_2014=G['mwe'].loc[2005:2014].mean())
# regional observation vs model statements
reg_model = [
    ('1_alaska', 'Rounce: peak 0.33-0.44 mm SLE/yr (2030-2060)', 0.33 * L.GT_PER_MM, 0.44 * L.GT_PER_MM, 'rate'),
    ('11_central_europe', 'Hock: 69+/-19% (RCP2.6) to 93+/-10% (RCP8.5) lost 2015-2100', 69, 93, 'pct'),
    ('3_arctic_canada_north', 'Hock: 12+/-8% (RCP2.6) to 23+/-15% (RCP8.5) lost 2015-2100', 12, 23, 'pct'),
    ('2_western_canada_us', 'Rounce: 60-100% lost by 2100; Marzeion RCP8.5 >85%', 60, 100, 'pct'),
    ('15_south_asia_east', 'Rounce: 60-100% lost by 2100', 60, 100, 'pct'),
    ('8_scandinavia', 'Rounce: 60-100% lost by 2100', 60, 100, 'pct'),
    ('10_north_asia', 'Rounce: 60-100% lost by 2100', 60, 100, 'pct'),
    ('16_low_latitudes', 'Rounce: 60-100%; Marzeion RCP8.5 >85%', 60, 100, 'pct'),
    ('12_caucasus_middle_east', 'Rounce: 60-100%; Marzeion RCP8.5 >85%', 60, 100, 'pct'),
    ('18_new_zealand', 'Rounce: 60-100% lost by 2100', 60, 100, 'pct'),
    ('5_greenland_periphery', 'Rounce: 12% of 2015-2100 glacier SLR', 12, 12, 'share'),
    ('19_antarctic_and_subantarctic', 'Rounce: 10% of 2015-2100 glacier SLR', 10, 10, 'share'),
    ('3_arctic_canada_north', 'Rounce: 10% of 2015-2100 glacier SLR', 10, 10, 'share'),
    ('4_arctic_canada_south', 'Rounce: 9% of 2015-2100 glacier SLR', 9, 9, 'share'),
]
rm = []
Ri = R.set_index('region')
for reg, txt, lo, hi, kind in reg_model:
    x = Ri.loc[reg]
    if kind == 'rate':
        obs = -x.rate_gt_2015_2023; obs_s = f'{obs:.0f} Gt/yr (2015-2023)'; mod = f'{lo:.0f}-{hi:.0f} Gt/yr'
        verdict = 'observed below projected peak' if obs < lo else 'observed within/above'
        implied = np.nan
    elif kind == 'pct':
        pct_yr = x.pct_per_yr_2015_2023; implied = pct_yr * 85  # % of 2015 mass lost by 2100 if 2015-2023 absolute rate persisted
        obs = implied
        if implied >= 100:
            obs_s = f'{pct_yr:.2f} %/yr of 2015 mass (2015-2023); constant rate depletes ice by ~{2015 + 100 / pct_yr:.0f}'
        else:
            obs_s = f'{pct_yr:.2f} %/yr of 2015 mass (2015-2023); constant rate -> {implied:.0f}% by 2100'
        mod = f'{lo}-{hi}% by 2100'
        verdict = ('below model range: models imply acceleration' if implied < lo else
                   ('within model range' if implied <= hi else 'at/above upper end: consistent with projected near-complete loss'))
    else:
        obs = x.share_pct; obs_s = f'{obs:.1f}% of 2000-2023 loss'; mod = f'{lo}% of 2015-2100 SLR'; implied = np.nan
        verdict = 'model share larger' if lo > obs + 2 else ('model share smaller' if lo < obs - 2 else 'similar')
    rm.append(dict(region=reg, label=x.label, model_statement=txt, model_range=mod, observed=obs_s, verdict=verdict))
RM = pd.DataFrame(rm); RM.to_csv(os.path.join(OUT, 'regional_obs_vs_models.csv'), index=False)
S['regional_obs_vs_models'] = RM.to_dict('records')

class NpEnc(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, (np.integer,)): return int(o)
        if isinstance(o, (np.floating,)): return float(o)
        if isinstance(o, np.ndarray): return o.tolist()
        return super().default(o)
def clean(o):
    if isinstance(o, dict): return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, list): return [clean(v) for v in o]
    if isinstance(o, (float, np.floating)): return None if not np.isfinite(o) else round(float(o), 4)
    return o
json.dump(clean(S), open(os.path.join(OUT, 'summary_stats.json'), 'w'), indent=1, cls=NpEnc)
print(json.dumps(clean({k: S[k] for k in ['global', 'official', 'canadian_arctic', 'zscore', 'method_bias', 'glac_rep', 'hybrid_validation', 'validation', 'proj_compare']}), indent=1, cls=NpEnc)[:9000])
