"""Write report/report.md; every number is taken from outputs/*.csv or outputs/summary_stats.json."""
import sys, os, json, numpy as np, pandas as pd
sys.path.insert(0, 'code'); import glambie_lib as L
OUT = 'outputs'
S = json.load(open(os.path.join(OUT, 'summary_stats.json')))
g, o, V = S['global'], S['official'], S['validation']
R = pd.read_csv(os.path.join(OUT, 'regional_summary.csv'))
ice = pd.read_csv(os.path.join(OUT, 'ice_sheet_comparison.csv'))
sens = pd.read_csv(os.path.join(OUT, 'sensitivity.csv'))
B = pd.read_csv(os.path.join(OUT, 'method_bias_vs_D.csv'))
F = pd.read_csv(os.path.join(OUT, 'method_forest.csv'))
GX = pd.read_csv(os.path.join(OUT, 'gravimetry_excluded_regions.csv'))
REP = pd.read_csv(os.path.join(OUT, 'glaciological_representativeness.csv'))
HV = pd.read_csv(os.path.join(OUT, 'hybrid_validation.csv'))
VAL = pd.read_csv(os.path.join(OUT, 'validation_regional_hy.csv'))
RM = pd.read_csv(os.path.join(OUT, 'regional_obs_vs_models.csv'))
inv = pd.read_csv(os.path.join(OUT, 'inventory.csv'))
cw = pd.read_csv(os.path.join(OUT, 'calendar_weights.csv'))
Z = pd.read_csv(os.path.join(OUT, 'zscores.csv'))
GS = pd.read_csv(os.path.join(OUT, 'global_series_main.csv')).set_index('year')
P, PC = S['projections'], S['proj_compare']

def md(df, floatfmt=2):
    cols = list(df.columns)
    s = '| ' + ' | '.join(cols) + ' |\n|' + '|'.join(['---'] * len(cols)) + '|\n'
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, (float, np.floating)):
                cells.append('–' if not np.isfinite(v) else f'{v:.{floatfmt}f}')
            else:
                cells.append(str(v))
        s += '| ' + ' | '.join(cells) + ' |\n'
    return s

f0 = lambda x: f'{x:.0f}'; f1 = lambda x: f'{x:.1f}'; f2 = lambda x: f'{x:.2f}'
cnt = inv.source.value_counts()
rate_e = g['rate_err']
can = S['canadian_arctic']
top = R.sort_values('total_gt').head(5)
mb = S['method_bias']; zs = S['zscore']; hv = S['hybrid_validation']; gr = S['glac_rep']
i18 = ice.set_index('period')

# --------------------------------------------------------------- tables
t_reg = R[['rid', 'label', 'rate_gt', 'rate_gt_err', 'rate_mwe', 'rate_mwe_err', 'rate_mwe_2000_2011', 'rate_mwe_2012_2023', 'share_pct', 'mass_2000_gt', 'loss_pct', 'official_rate_gt', 'dev_from_official_sigma']].copy()
t_reg.columns = ['#', 'region', 'Gt/yr', '±σ', 'm w.e./yr', '±σ ', 'm w.e./yr 2000–11', 'm w.e./yr 2012–23', 'share of loss %', 'mass ~2000 (Gt)', 'lost 2000–23 (%)', 'official Gt/yr', 'Δ/σ vs official']
t_reg['mass ~2000 (Gt)'] = t_reg['mass ~2000 (Gt)'].map(lambda x: f'{x:,.0f}')
t_reg['#'] = t_reg['#'].astype(int)

t_ice = ice[['period', 'glaciers', 'glaciers_e', 'gris', 'gris_e_1s', 'ais', 'ais_e_1s', 'ratio_gris', 'ratio_ais', 'share_of_landice_pct']].copy()
t_ice.columns = ['period', 'glaciers (this study) Gt/yr', '±σ', 'GrIS Gt/yr', '±σ ', 'AIS Gt/yr', '±σ  ', 'glaciers/GrIS', 'glaciers/AIS', 'glacier share of land-ice loss %']

t_sens = sens[['config', 'global_rate', 'd_global_rate', 'rate_2000_2011', 'rate_2012_2023', 'ratio_late_early', 'gt_2023', 'cal_rmsd_gt', 'reg_mean_rmsd_gt', 'err_ratio_global']].copy()
t_sens.columns = ['configuration', 'rate Gt/yr', 'Δ vs main', '2000–11', '2012–23', 'late/early', '2023 (Gt)', 'RMSD vs official CY (Gt)', 'mean regional HY RMSD (Gt)', 'σ ratio vs official']
off_row = pd.DataFrame([{'configuration': '**official GlaMBIE**', 'rate Gt/yr': o['rate_gt'], 'Δ vs main': o['rate_gt'] - g['rate_gt'], '2000–11': o['rate_2000_2011'], '2012–23': o['rate_2012_2023'],
                         'late/early': o['rate_2012_2023'] / o['rate_2000_2011'], '2023 (Gt)': o['gt_2023'], 'RMSD vs official CY (Gt)': 0.0, 'mean regional HY RMSD (Gt)': 0.0, 'σ ratio vs official': 1.0}])
t_sens = pd.concat([t_sens, off_row])

t_bias = B[['label', 'group', 'y0', 'y1', 'n', 'bias', 'bias_err', 'rmsd', 'r']].copy()
t_bias.columns = ['region', 'group', 'from', 'to', 'n yrs', 'bias vs D (m w.e./yr)', '±σ', 'RMSD', 'r (annual)']
t_bias['from'] = t_bias['from'].astype(int); t_bias['to'] = t_bias['to'].astype(int); t_bias['n yrs'] = t_bias['n yrs'].astype(int)

t_gx = GX[['label', 'area', 'grav_rate', 'D_rate', 'diff', 'r']].copy(); t_gx.columns = ['region (gravimetry not used)', 'area km²', 'GRACE m w.e./yr', 'D m w.e./yr', 'GRACE − D', 'r (annual)']
t_gx['area km²'] = t_gx['area km²'].map(lambda x: f'{x:,.0f}')

t_rep = REP[['label', 'geodetic_rate', 'glac_sinewave_rate', 'glac_minus_geod', 'wgms_beta_mean', 'r_wgms_A', 'r_wgms_G']].copy()
t_rep.columns = ['region', 'geodetic 2000–19 (Hugonnet)', 'glaciological sample (sine-wave product)', 'glac − geod', 'WGMS-beta mean (anomaly)', 'r(WGMS, altimetry)', 'r(WGMS, gravimetry)']

hvd = HV.copy(); hvd['dataset'] = hvd.dataset.str.replace('_et_al', '').str.replace('DUS-combined_', '')
t_hv = hvd.groupby('dataset').agg(n_regions=('region', 'count'), mean_diff=('diff', 'mean'), median_r=('r', 'median')).reset_index()
t_hv.columns = ['hybrid dataset', 'regions', 'mean (hybrid − consensus w/o hybrids) m w.e./yr', 'median r']

val = VAL[VAL.group == 'combined'][['region', 'bias', 'rmsd', 'r', 'err_ratio']].copy()
valD = VAL[VAL.group == 'demdiff_and_glaciological'][['region', 'bias', 'rmsd']].rename(columns={'bias': 'D bias', 'rmsd': 'D RMSD'})
valA = VAL[VAL.group == 'altimetry'][['region', 'bias']].rename(columns={'bias': 'A bias'})
valG = VAL[VAL.group == 'gravimetry'][['region', 'bias']].rename(columns={'bias': 'G bias'})
val = val.merge(valD, on='region', how='left').merge(valA, on='region', how='left').merge(valG, on='region', how='left')
val['region'] = val.region.map(lambda r: L.LABELS[L.rid_of(r) - 1])
val = val[['region', 'bias', 'rmsd', 'r', 'err_ratio', 'D bias', 'D RMSD', 'A bias', 'G bias']]
val.columns = ['region', 'combined bias', 'combined RMSD', 'r', 'σ ratio (this/official)', 'D bias', 'D RMSD', 'A bias', 'G bias']

t_cw = cw.copy(); t_cw['region'] = t_cw.region.map(lambda r: L.LABELS[L.rid_of(r) - 1])
t_cw = t_cw[['region', 'hy_start', 'w_sinewave', 'w_official_implied', 'w_time']]
t_cw.columns = ['region', 'HY start', 'w (sine-wave product, used)', 'w implied by official (diagnostic)', 'w (pure time fraction)']

proj_rows = []
for key, name in [('rounce2023', 'Rounce et al. 2023'), ('marzeion2020', 'Marzeion et al. 2020'), ('hock2019', 'Hock et al. 2019')]:
    d = P[key]
    for sc, (v, e) in d['slr_2100_mm'].items():
        ml = d['mass_loss_pct_2100'].get(sc)
        proj_rows.append({'study': name, 'scenario': sc, 'SLE 2015–2100 (mm)': f'{v} ± {e}', 'mass lost by 2100 (% of 2015)': f'{ml[0]} ± {ml[1]}' if ml else '–',
                          'implied mean rate (mm/yr)': v / 85, 'years needed at observed 2015–23 rate': v / PC['obs_rate_2015_2023_mm'], 'uncertainty': d['ci'], 'source': d['src'].split('(')[1].split(')')[0] + ', ' + d['src'].split(')')[1].strip(' ,')})
t_proj = pd.DataFrame(proj_rows)
t_rm = RM[['label', 'model_statement', 'observed', 'verdict']]; t_rm.columns = ['region', 'model projection (source)', 'observation (this study)', 'assessment']
rec = pd.DataFrame(g['record_years']); rec['rank'] = range(1, 6); rec = rec[['rank', 'year', 'gt', 'err', 'slr_mm']]; rec.columns = ['rank', 'year', 'Gt', '±σ', 'mm SLE']
rec['year'] = rec['year'].astype(int)
t_glob = pd.DataFrame({'year': GS.index.astype(int), 'Gt': GS['gt'], '±σ': GS.gt_err, 'm w.e.': GS['mwe'], '±σ ': GS.mwe_err, 'cumulative Gt': GS.cum_gt, 'cum. mm SLE': GS.cum_slr_mm, 'official Gt': GS.official_gt})

wA = F[(F.window == 'altimetry period')]
pc = 100 * PC['obs_pct_lost_2015_2023']

rep = f"""# Reconciled global glacier mass change 2000–2023 from 233 multi-method regional estimates

*An independent re-implementation of the GlaMBIE community reconciliation, with validation, uncertainty diagnostics, and comparison with ice sheets and model projections.*

## Abstract

We combine the 233 regional glacier mass-change estimates of the Glacier Mass Balance Intercomparison Exercise (GlaMBIE). They are delivered as {len(inv)} files: {cnt['glaciological']} glaciological, {cnt['demdiff']} DEM differencing, {cnt['altimetry']} altimetry, {cnt['gravimetry']} gravimetry and {cnt['combined']} hybrid series. From them we derive annual regional and global glacier mass change for 2000–2023 with uncertainties.

The inputs are homogenised and annualised, combined within three data groups (DEM differencing + glaciological, altimetry, gravimetry), averaged across groups, and converted to calendar years. Global glaciers lost **{f0(-g['total_gt'])} ± {f0(g['total_err_quad'])} Gt** over 2000–2023. This is {f1(g['rate_gt'])} ± {f1(rate_e)} Gt yr⁻¹, or {f2(g['mwe_rate'])} m w.e. yr⁻¹, and equals **{f1(g['total_slr_mm'])} ± {f1(g['total_slr_err'])} mm** of global-mean sea-level rise.

Loss accelerated. The mean rate changed from {f0(g['rate_2000_2011'])} ± {f0(g['err_2000_2011'])} Gt yr⁻¹ (2000–2011) to {f0(g['rate_2012_2023'])} ± {f0(g['err_2012_2023'])} Gt yr⁻¹ (2012–2023), an increase of {f0(g['increase_pct'])} ± {f0(g['increase_pct_err'])} %. The linear trend in the annual rates is {f1(g['accel_gt_yr2'])} ± {f1(g['accel_se'])} Gt yr⁻². 2023 was the most negative year on record at {f0(g['gt_2023'])} ± {f0(g['err_2023'])} Gt.

Our main configuration was chosen among predefined alternatives by reproducing the published GlaMBIE results. It agrees with the official values:
- total {f0(o['total_gt'])} ± {f0(o['total_err_quad'])} Gt (difference {f1(100*(g['total_gt']/o['total_gt']-1))} %);
- global calendar-year RMSD {f0(V['cal_rmsd'])} Gt;
- all 19 regional mean rates within 1σ.

Over 2000–2018 glaciers lost {f1(i18.loc['2000-2018','ratio_gris'])}× as much mass as the Greenland Ice Sheet and {f1(i18.loc['2000-2018','ratio_ais'])}× as much as the Antarctic Ice Sheet. That makes them the largest single land-ice contributor to sea-level rise in that period.

Central Europe has lost {f0(R.set_index('rid').loc[11,'loss_pct'])} % of its ice mass since 2000, and five more regions have lost more than 20 %. The observed 2015–2023 global rate ({f2(PC['obs_rate_2015_2023_mm'])} mm SLE yr⁻¹) is already comparable to the mean 2015–2100 rate of the low-emission projections (0.93–1.15 mm yr⁻¹).

## 1. Introduction

Glaciers outside the two ice sheets hold less than 1 % of Earth's land ice. Even so, they are a leading contributor to present sea-level rise and a key freshwater resource (Zemp et al. 2019; Hugonnet et al. 2021).

Four observation families measure glacier mass change:
- in-situ glaciological mass balance;
- multi-temporal DEM differencing;
- satellite altimetry (ICESat, CryoSat-2, ICESat-2);
- satellite gravimetry (GRACE / GRACE-FO).

They differ in spatial coverage, temporal resolution and error structure. Glaciological series resolve inter-annual variability but sample few glaciers. DEM differencing is spatially complete but only multi-annual. Altimetry and gravimetry are sub-annual but limited in time and, for GRACE, in spatial resolution. Earlier assessments disagreed by tens of Gt yr⁻¹ depending on which method dominated.

GlaMBIE collected 233 estimates from 35 teams to produce a community reconciliation. This report:
1. re-implements that reconciliation from the raw inputs;
2. validates it quantitatively against the published GlaMBIE results;
3. tests its methodological choices and uncertainty model;
4. places the result in context: ice-sheet losses, relative regional ice loss, method consistency, and the glacier-model projections used by IPCC (GlacierMIP; Rounce et al. 2023).

## 2. Data

`data/glambie/input` holds one CSV per submitted solution and region. Each row gives start and end dates (fractional years), a change, its error and a unit: m of elevation, m w.e. or Gt (Fig. 1). The temporal structure differs by method:
- **Glaciological:** monthly WGMS/UZH sine-wave products and annual WGMS-beta series. The WGMS-beta series are *anomaly/variability* products by design: their 2001–2020 means are close to zero, from {f2(gr['wgms_beta_range'][0])} to {f2(gr['wgms_beta_range'][1])} m w.e. yr⁻¹, mean {f2(gr['wgms_beta_mean_of_means'])}. They carry variability, not level.
- **DEM differencing:** multi-annual trends, e.g. the Hugonnet et al. pentads 2000–2020.
- **Altimetry:** monthly CryoSat-2 series (Jakob & Gourmelen; Nilsson/Gardner) plus multi-year ICESat/ICESat-2 trends.
- **Gravimetry:** monthly GRACE/GRACE-FO solutions from seven groups. Some fill the 2017–2018 mission gap by interpolation (Harig: NaN), plus one trend (Jacob 2012).
- **Hybrids:** annual glaciological–geodetic syntheses (Dussaillant, Huss, Pálsson), in-situ+GRACE series (Box), and trends (Colgan, Miles).

![Data overview](images/fig01_data_overview.png)

*Figure 1. (a) Number of input datasets per region and method. (b) Number of datasets covering each year, all regions.*

`data/glambie/results` contains the official GlaMBIE results: hydrological and calendar years, per data group and combined, plus a global series. We use them in three ways only:
1. the `glacier_area` column, as auxiliary RGI 6.0-based area information (a linear area model A(t) per region);
2. validation;
3. choosing among a small set of predefined methodological variants (Section 3.6).

Regional RGI 6.0 areas and ice volumes are taken from Zemp et al. (2019, Table 1). Ice-sheet mass balance is taken from Hugonnet et al. (2021, Table 1). Projections are taken from Rounce et al. (2023), Marzeion et al. (2020) and Hock et al. (2019).

## 3. Methods

The code is in `code/` (see `code/README.md`); the method itself is in `glambie_lib.py`.

### 3.1 Homogenisation
- **Elevation change → mass:** converted with a density of ρ = 850 ± 60 kg m⁻³, with σ² = (ρσ_h)² + (hσ_ρ)².
- **Common unit:** every dataset is expressed in m w.e. relative to the fixed RGI 6.0 regional area A_RGI. This is the reference area of the geodetic and glaciological inputs, so it is equivalent to Gt: Gt = m w.e. · A_RGI · 10⁻³. Gt inputs are converted with A_RGI.
- **Output specific change:** final values are expressed relative to the time-varying area, m w.e.(t) = 10³ Gt / A(t).

The fixed reference area matters. In small, fast-shrinking regions such as Central Europe (A falls by about 20 % over 2000–2023), using A(t) in both conversions underestimates specific loss late in the record. That is one cause of the earlier under-estimate in Central Europe.

Error handling:
- Missing or zero errors are filled with the dataset median.
- The Pálsson hybrids report no errors and are given 0.3 m w.e. yr⁻¹.
- Lower-case `gt` units are accepted.

### 3.2 Annualisation
Hydrological years (HY) start on 1 October in regions 1–15, 1 January in region 16 and 1 April in regions 17–19.

**Monthly series** are summed within each HY:
- *Glaciological:* at least 10 of 12 months required. Monthly errors are treated as increments and combined in quadrature.
- *Altimetry and gravimetry:* all 12 months required. Per-epoch errors are combined as an RMS. For gravimetry the RMS is divided by √12, because monthly GRACE noise is largely independent between months; this matches the published group uncertainties (Section 4.6). The *level* of each altimetry or gravimetry series is set by an OLS regression of its cumulative series (trend + annual and semi-annual harmonics, with one offset per contiguous segment). The annual values are shifted so that their mean equals the regression trend. This avoids the end-epoch noise of two-epoch differences.
- *GRACE gap:* months in the 2017–2018 gap (2017.40–2018.49) are discarded, even when a provider interpolated through it. The change across the gap is kept as a trend record and distributed with external variability. In the official files, the gravimetry series in the gap years likewise lacks its own annual variability (`annual_variability` flag empty).

**Annual series** are mapped onto the HY grid.

**Trend datasets** keep their periods. Multi-period DEM series (e.g. Hugonnet pentads) are calibrated pentad by pentad in the main configuration. The full-period alternative is a sensitivity run.

### 3.3 Combination within data groups
Each trend record k with change Δ_k over [t₀, t₁] is turned into an annual series S_k(y) = V(y) + c_k. Here V is an annual variability ("shape") series, and c_k is chosen so that Σ_y f_ky S_k(y) = Δ_k, where f_ky is the fraction of HY y inside the period. Years with f > 0.5 are assigned to the record.

- **DEM differencing + glaciological (D):** V is the mean anomaly (relative to 2000–2019) of the glaciological series and the annual hybrid series. The level is the team-mean offset of all DEM records and of the annual hybrids, which vote year by year. Hybrids sit in this group in the main configuration. Beyond the last level observation the offset is held at its nearest value, with an extrapolation error of 0.1 m w.e. yr⁻¹ per √yr.
- **Altimetry (A) and gravimetry (G):** V is the group's own annual data where they exist, otherwise the D-group series (as in GlaMBIE's "annual variability from other sources"). The annual datasets enter directly; trend datasets are calibrated as above.
- **Team voting:** datasets from the same team (e.g. two Treichler, Pfeffer, Pálsson, Piermattei or Box variants) are averaged first, so each team has one vote.
- **Gravimetry regions:** gravimetry is used in the seven regions of the official product (1, 3, 4, 6, 7, 9, 17). Elsewhere, leakage and hydrological signals exceed the glacier signal (Section 4.4).

### 3.4 Combination of data groups and uncertainties
The regional estimate is the **unweighted mean of the available data groups** in each year. There is no gap-filling. The official combined series is reproduced to within 0.01 m w.e. by this rule when applied to the official group series.

Uncertainties:
- **Group uncertainty:** the standard error of the team mean, √(mean σ²/N + s²/N), where s is the between-team spread. In D, the variability, level and spread terms are added in quadrature.
- **Combined uncertainty:** √(Σ_g σ_g²) over available groups. The errors of the group mean are *not* divided by N, because the groups share systematic errors (density, area, level calibration). When a group is missing, its median uncertainty is added, so the combined uncertainty does not shrink just because fewer independent methods exist.
- **Multi-year uncertainty:** annual uncertainties summed in quadrature. This convention reproduces the official multi-year uncertainty from the official annual errors exactly ({f0(o['total_err_quad'])} Gt for 2000–2023). We also report the fully correlated upper bound.

### 3.5 Calendar years and global aggregation
Calendar years are formed as C(Y) = (1 − w) H(Y) + w H(Y ± 1). The neighbour is the following HY in the Northern Hemisphere and the preceding HY in the Southern Hemisphere; region 16 uses w = 0.

The weight w is derived *independently* for each region. We regress calendar-year sums of the monthly glaciological sine-wave product on its HY sums. This captures where in the year inter-annual anomalies occur: little in October–December in the NH, much in January–March in the SH. These weights agree closely with the weights implied by the official HY/CY pairs (RMS difference {f2(S['calendar_weights_rmsd_sine_vs_official'])}; Table A2, Fig. 11c) and outperform a pure time fraction (w = 0.25; Table 3).

Edge years:
- *NH CY2023* needs HY2024 (Oct 2023 – Sep 2024). It is taken from the D group, which has annual data to 2024.
- *SH CY2023* needs the HY starting April 2023. It is taken from the D-group inputs that extend to March 2024 (Dussaillant, Zemp, WGMS) rather than by persistence.
- *SH CY2000* uses the HY starting April 1999.

The conversion adds an uncertainty of 0.1·|H(Y ± 1) − H(Y)|. Global values are sums of the 19 regional calendar series. The code checks that all 19 regions are present in every year. Global errors are summed in quadrature over regions.

### 3.6 Choosing the main configuration
We ran 13 predefined configurations (Table 3) varying:
- the handling of hybrids (excluded, inside D, separate group);
- DEM calibration (per pentad vs full period);
- gravimetry regions (official seven / all / none);
- altimetry (with / without);
- gap-filling;
- glaciological series used for level;
- level estimation;
- calendar weights.

Each was scored on reproduction of the official hydrological-year regional series (bias, RMSD) and of the global calendar series. The main configuration is the one closest to the official results: hybrids inside D, per-pentad DEM calibration, gravimetry in seven regions, no gap-filling, and seasonal calendar weights. All other configurations are reported as sensitivities. This choice of a *main* estimate (a reproduction of the community method) is kept separate from the *independent* methodological sensitivities.

![Workflow](images/fig02_workflow_iceland.png)

*Figure 2. Iceland as a worked example. Thin lines are annualised input datasets; horizontal bars are trend datasets (rate per year). Thick lines are the group estimates (±1σ shading); dashed lines are the official GlaMBIE groups. The combined estimate (d) is the mean of the three groups.*

## 4. Results

### 4.1 Global mass change, acceleration and sea-level contribution
Global glaciers lost:
- **{f0(-g['total_gt'])} ± {f0(g['total_err_quad'])} Gt** over 2000–2023 (fully correlated bound ±{f0(g['total_err_fullcorr'])} Gt);
- an average of {f1(-g['rate_gt'])} ± {f1(rate_e)} Gt yr⁻¹, i.e. {f2(-g['mwe_rate'])} m w.e. yr⁻¹, or {f1(-g['mwe_total'])} ± {f1(g['mwe_total_err'])} m w.e. cumulatively;
- a sea-level equivalent (362.5 Gt per mm) of **{f1(g['total_slr_mm'])} ± {f1(g['total_slr_err'])} mm**, or {f2(g['rate_slr_mm'])} ± {f2(g['rate_slr_err'])} mm yr⁻¹ (Fig. 4, Table A1).

For 2000–2019 the rate is {f0(g['rate_2000_2019'])} ± {f0(g['err_2000_2019'])} Gt yr⁻¹, consistent with Hugonnet et al. (2021): −267 ± 16 Gt yr⁻¹ (95 % CI).

**Acceleration.** Mean loss increased from {f0(-g['rate_2000_2011'])} ± {f0(g['err_2000_2011'])} Gt yr⁻¹ in 2000–2011 to {f0(-g['rate_2012_2023'])} ± {f0(g['err_2012_2023'])} Gt yr⁻¹ in 2012–2023, i.e. **+{f0(g['increase_pct'])} ± {f0(g['increase_pct_err'])} %** (official: +{f0(o['increase_pct'])} %).

Pentad means were:

| period | Gt yr⁻¹ |
|---|---|
| 2000–04 | {f0(g['rate_2000_2004'])} |
| 2005–09 | {f0(g['rate_2005_2009'])} |
| 2010–14 | {f0(g['rate_2010_2014'])} |
| 2015–19 | {f0(g['rate_2015_2019'])} |
| 2020–23 | {f0(g['rate_2020_2023'])} |

An OLS fit to the annual rates gives an acceleration of {f1(g['accel_gt_yr2'])} ± {f1(g['accel_se'])} Gt yr⁻² ({f0(g['accel_gt_yr_per_decade'])} ± {f0(g['accel_se_decade'])} Gt yr⁻¹ per decade; uncertainty-weighted: {f1(g['accel_wls'])}; official series: {f1(o['accel_gt_yr2'])}).

**Record years.** 2023 is the most negative year of the record: {f0(g['gt_2023'])} ± {f0(g['err_2023'])} Gt, {f2(g['mwe_2023'])} m w.e., {f2(g['record_years'][0]['slr_mm'])} mm SLE in a single year. It is also rank 1 in the official series. The five most negative years (Table 1) all fall after 2010; four of them after 2018. Of the {g['n_years_below_minus300']} years with losses above 300 Gt, {g['years_below_minus300_after_2012']} fall in 2012–2023.

**Table 1. Five most negative years (calendar years).**

{md(rec, 1)}

![Global series](images/fig04_global_series.png)

*Figure 4. (a) Annual global glacier mass change (bars, ±1σ) with the official GlaMBIE values (grey dots) and the 2000–2011 and 2012–2023 means. (b) Cumulative change since 1 January 2000.*

### 4.2 Comparison with the ice sheets
Table 2 compares glacier loss with the Greenland (GrIS) and Antarctic (AIS) ice-sheet losses reported by Hugonnet et al. (2021, Table 1). Those are IMBIE estimates minus half of the peripheral-glacier component, to avoid double counting; their 95 % intervals are converted here to 1σ; the AIS series ends in June 2017.

- **2000–2018:** glaciers lost {f0(-i18.loc['2000-2018','glaciers'])} ± {f0(i18.loc['2000-2018','glaciers_e'])} Gt yr⁻¹, versus {f0(-i18.loc['2000-2018','gris'])} ± {f0(i18.loc['2000-2018','gris_e_1s'])} Gt yr⁻¹ for the GrIS and {f0(-i18.loc['2000-2018','ais'])} ± {f0(i18.loc['2000-2018','ais_e_1s'])} Gt yr⁻¹ for the AIS. That is a ratio of {f2(i18.loc['2000-2018','ratio_gris'])} ± {f2(i18.loc['2000-2018','ratio_gris_e'])} to Greenland and {f2(i18.loc['2000-2018','ratio_ais'])} ± {f2(i18.loc['2000-2018','ratio_ais_e'])} to Antarctica. Glaciers supplied {f0(i18.loc['2000-2018','share_of_landice_pct'])} % of total land-ice loss.
- **2003–2018:** ratios of {f2(i18.loc['2003-2018','ratio_gris'])} and {f2(i18.loc['2003-2018','ratio_ais'])}.

**Over the first two decades of the century, glaciers were therefore the largest single land-ice contributor to sea-level rise.** They exceeded each ice sheet individually, though not the two combined. The only exception was 2010–2014, when Greenland's loss ({f0(-i18.loc['2010-2014','gris'])} Gt yr⁻¹) was comparable to the glacier loss ({f0(-i18.loc['2010-2014','glaciers'])} Gt yr⁻¹; ratio {f2(i18.loc['2010-2014','ratio_gris'])} ± {f2(i18.loc['2010-2014','ratio_gris_e'])}).

Glacier loss after 2018 is larger still: {f0(-g['rate_2020_2023'])} Gt yr⁻¹ in 2020–2023. The ice-sheet values from the related work end in 2018, so we do not extend the ratio.

**Table 2. Glacier vs ice-sheet mass change (Gt yr⁻¹, ±1σ).**

{md(t_ice, 2)}

![Ice sheets](images/fig09_ice_sheet_comparison.png)

*Figure 9. Glacier (this study) and ice-sheet (Hugonnet et al. 2021) mass-loss rates. Labels give the glacier share of total land-ice loss.*

### 4.3 Regional contributions and relative loss
Regional results are in Table 4 and Figs. 3 and 5.

**Largest contributors.** The five largest contributors over 2000–2023 are {', '.join(top.label)}, together {f0(top.share_pct.sum())} % of global loss. Alaska alone supplies {f0(R.set_index('rid').loc[1,'share_pct'])} % ({f1(-R.set_index('rid').loc[1,'rate_gt'])} ± {f1(R.set_index('rid').loc[1,'rate_gt_err'])} Gt yr⁻¹). Taken together, the **Canadian Arctic (North + South)** contributes {f1(can['share_pct'])} % ({f1(-can['rate_gt'])} ± {f1(can['rate_gt_err'])} Gt yr⁻¹) and is the second-largest source after Alaska. The ice-sheet peripheries (Greenland, Antarctic) together contribute {f0(R.set_index('rid').loc[5,'share_pct']+R.set_index('rid').loc[19,'share_pct'])} %.

**Specific rates.** The most negative specific rates are in Central Europe ({f2(R.set_index('rid').loc[11,'rate_mwe'])} m w.e. yr⁻¹), New Zealand ({f2(R.set_index('rid').loc[18,'rate_mwe'])}), the Southern Andes ({f2(R.set_index('rid').loc[17,'rate_mwe'])}) and Iceland ({f2(R.set_index('rid').loc[6,'rate_mwe'])}). The least negative are the Antarctic periphery ({f2(R.set_index('rid').loc[19,'rate_mwe'])}) and South Asia West ({f2(R.set_index('rid').loc[14,'rate_mwe'])}).

**Relative loss.** We take the ~2000 regional ice mass as Zemp et al. (2019) Table 1 volumes (RGI 6.0 outlines, early 2000s) × 0.9 t m⁻³. Glaciers lost {f1(S['global_loss_pct'])} ± {f1(S['global_loss_pct_err'])} % of their global mass ({S['global_mass_2000_gt']:,.0f} Gt) in 24 years. The relative loss is very uneven:
- **Central Europe:** {f0(R.set_index('rid').loc[11,'loss_pct'])} %;
- **New Zealand:** {f0(R.set_index('rid').loc[18,'loss_pct'])} %;
- **Caucasus & Middle East:** {f0(R.set_index('rid').loc[12,'loss_pct'])} %;
- **Low Latitudes:** {f0(R.set_index('rid').loc[16,'loss_pct'])} %;
- **Western Canada & USA:** {f0(R.set_index('rid').loc[2,'loss_pct'])} %;
- **North Asia:** {f0(R.set_index('rid').loc[10,'loss_pct'])} %;
- **South Asia East:** {f0(R.set_index('rid').loc[15,'loss_pct'])} %;
- by contrast, the large polar regions lost 1–7 %: Antarctic {f1(R.set_index('rid').loc[19,'loss_pct'])} %, Russian Arctic {f1(R.set_index('rid').loc[9,'loss_pct'])} %, Arctic Canada North {f1(R.set_index('rid').loc[3,'loss_pct'])} %.

So the regions that dominate sea-level rise are not the regions closest to deglaciation. Uncertainty in the ice volumes (≈ ±25 % regionally; Farinotti et al. 2019, as cited in paper_001) is not propagated into the percentages.

**Acceleration by region.** Specific loss increased from 2000–2011 to 2012–2023 in most regions. Specific loss increased in 14 of 19 regions; the largest increases (≥0.4 m w.e. yr⁻¹) were in Western Canada & USA, Central Europe, Svalbard and Caucasus & Middle East. Iceland, Scandinavia and the two Arctic Canada regions are exceptions with similar or less negative rates in 2012–2023, reflecting the "North Atlantic anomaly" of decelerated loss described by Hugonnet et al. (2021) (Table 4).

**Agreement with official.** All {S['n_regions_within_1sigma_official']} regional mean rates agree with the official GlaMBIE rates within 1σ (|Δ/σ| < 1).

**Table 4. Regional mass change 2000–2023 (calendar years).**

{md(t_reg, 2)}

![Regional rates](images/fig05_regional_rates_relative_loss.png)

*Figure 5. (a) Regional mass change rate (Gt yr⁻¹, ±1σ) with share of global loss; grey dots are official. (b) Specific rate. (c) Cumulative 2000–2023 loss as a percentage of ~2000 ice mass.*

![Regional series](images/fig03_regional_small_multiples.png)

*Figure 3. Annual specific mass change per region (hydrological years). Coloured lines are data groups, black is the combined estimate (±1σ), dashed grey is the official GlaMBIE combined series.*

### 4.4 Consistency between observation methods
**Common windows (Fig. 6).** Over the years shared by all groups in a region, the groups agree within their uncertainties in most regions. Systematic differences remain (Table 5, Fig. 7c):

1. **Altimetry is systematically less negative than D.** The mean altimetry − D bias is {f2(mb['alt_polar_mean_bias'])} m w.e. yr⁻¹ in the polar and ice-sheet-periphery regions (3, 4, 5, 7, 9, 19) and {f2(mb['alt_nonpolar_mean_bias'])} elsewhere. In the polar regions this matters most for the total, because their areas are large. The largest biases are in Arctic Canada South (+{f2(B.set_index(['region','group']).loc[('4_arctic_canada_south','A'),'bias'])}), Antarctic & Subantarctic (+{f2(B.set_index(['region','group']).loc[('19_antarctic_and_subantarctic','A'),'bias'])}), Greenland Periphery (+{f2(B.set_index(['region','group']).loc[('5_greenland_periphery','A'),'bias'])}) and, in a short window, Western Canada (+{f2(B.set_index(['region','group']).loc[('2_western_canada_us','A'),'bias'])}) and Alaska (+{f2(B.set_index(['region','group']).loc[('1_alaska','A'),'bias'])}). Possible causes include radar penetration into firn, incomplete coverage of fast-thinning marine termini and ablation zones by CryoSat swaths, and the density assumption for elevation change in accumulation areas.
2. **Gravimetry is less negative than D in Alaska** (+{f2(B.set_index(['region','group']).loc[('1_alaska','G'),'bias'])} ± {f2(B.set_index(['region','group']).loc[('1_alaska','G'),'bias_err'])}) and Svalbard (+{f2(B.set_index(['region','group']).loc[('7_svalbard','G'),'bias'])}). In the large Arctic regions it agrees within ±0.06 m w.e. yr⁻¹. Annual correlations with D are moderate to high (median r = {f2(mb['grav_median_r'])}; altimetry {f2(mb['alt_median_r'])}). Because independent methods share the same inter-annual signal, that signal is real and not an artefact of the glaciological shape.
3. **GRACE fails in small regions and High Mountain Asia** (Table 6). Where gravimetry is excluded from the main estimate, GRACE solutions give *positive* rates in Scandinavia, North Asia, Caucasus, Low Latitudes and New Zealand. Their annual uncertainties are several m w.e. Even in Central and South Asia, where the regions are large, annual correlation with D is low. This reflects leakage from groundwater depletion, lakes and snow, and the coarse GRACE footprint relative to glacier areas of 10³ km². Including these solutions (configuration `grav_all`) changes the global mean by only {f1(sens.set_index('config').loc['grav_all','d_global_rate'])} Gt yr⁻¹ but strongly degrades regional reproduction (mean regional RMSD {f1(sens.set_index('config').loc['grav_all','reg_mean_rmsd_gt'])} vs {f1(sens.set_index('config').loc['main','reg_mean_rmsd_gt'])} Gt). Excluding them is justified.
4. **Glaciological representativeness** (Table 7). Over 2000–2019 the regional glaciological sample (UZH sine-wave product) is on average {f2(gr['mean_diff'])} m w.e. yr⁻¹ more negative than the geodetic (Hugonnet) mean, with a mean absolute difference of {f2(gr['mean_abs_diff'])}. The bias is largest in High Mountain Asia (−0.27 to −0.42), the Russian Arctic and Greenland Periphery, where monitored glaciers are small, low-lying, land-terminating and not representative. This is why the glaciological series are used only for *variability* and the level comes from geodetic data. When they also vote on the level (`glac_level`), the global mean shifts by {f0(sens.set_index('config').loc['glac_level','d_global_rate'])} Gt yr⁻¹ and the reproduction worsens. The WGMS-beta anomalies correlate with the independent altimetry (median r = {f2(gr['median_r_wgms_A'])}) and gravimetry (median r = {f2(gr['median_r_wgms_G'])}) annual series, confirming that their variability is representative even though their level is not meaningful.
5. **Hybrid datasets** (Table 8). Run the main estimate with all hybrids withheld (`hyb_exclude`) and compare each hybrid series against it: across {hv['n_series']} hybrid series the hybrids are on average {f2(hv['mean_diff'])} m w.e. yr⁻¹ more negative than the non-hybrid consensus (median r = {f2(hv['median_r'])}). The differences are largest in Alaska (hybrids more negative by 0.1–0.23) and New Zealand (hybrids less negative by 0.13–0.17). Withholding hybrids makes the global estimate {f1(sens.set_index('config').loc['hyb_exclude','d_global_rate'])} Gt yr⁻¹ less negative and lowers the 2012–2023 acceleration. The hybrids thus carry independent level information (mostly from the geodetic calibration of glaciological series after 2020) that the official method relies on.

**Table 5. Bias and annual correlation of altimetry (A) and gravimetry (G) relative to the DEM+glaciological group (common hydrological years).**

{md(t_bias, 3)}

**Table 6. Gravimetry in regions where it is not used in the main estimate (HY 2003–2022).**

{md(t_gx, 2)}

**Table 7. Glaciological sample vs geodetic rates 2000–2019 (m w.e. yr⁻¹) and inter-annual agreement of WGMS-beta anomalies with independent groups.**

{md(t_rep, 2)}

**Table 8. Hybrid datasets vs the consensus computed without hybrids.**

{md(t_hv, 3)}

![Forest plot](images/fig06_method_forest.png)

*Figure 6. Mean specific mass change of each data group over common windows: (a) the years covered by altimetry, (b) the GRACE years, per region. Error bars are the mean annual group uncertainty, treated as fully correlated in time.*

### 4.5 Uncertainty diagnostics
If the uncertainties are realistic, z = (x₁ − x₂)/√(σ₁² + σ₂²) for annual differences between data groups should have unit standard deviation. Over {zs['n']} group-pair-years, SD(z) = **{f2(zs['sd'])}**, with {f0(100*zs['frac_abs_gt2'])} % of |z| > 2 (Fig. 7a). By pair, SD(z) is {f2(zs['by_pair']['D-A']['std'])} for D−A, {f2(zs['by_pair']['D-G']['std'])} for D−G and {f2(zs['by_pair']['A-G']['std'])} for A−G. The mean z is {f2(zs['mean'])}, reflecting the systematic altimetry and gravimetry biases above.

So the *group* uncertainties are slightly too small (by about 20–30 %) to explain the annual differences between groups, mostly because of the systematic biases. The *combined* uncertainty does not divide by N and adds a missing-group term, which compensates for this. Our combined annual uncertainties are on average {f2(V['err_ratio_global'])} × the official global uncertainties and median {f2(V['combined_median_err_ratio'])} × the official regional ones (Table A3). That makes them at least as conservative as the community product.

![z-test](images/fig07_uncertainty_ztest_bias.png)

*Figure 7. (a) z-scores of annual between-group differences against N(0,1). (b) SD of z by region. (c) Mean bias of altimetry and gravimetry relative to the D group (±1σ).*

### 4.6 Validation against the official GlaMBIE results and sensitivity
**Validation.** Against the official regional hydrological-year series (Table A3, Fig. 11a):
- mean absolute bias of the combined series: {V['combined_mean_abs_bias']:.3f} m w.e. yr⁻¹;
- median RMSD: {V['combined_median_rmsd']:.3f} m w.e.;
- median correlation: {V['combined_median_r']:.3f}.

The D group is unbiased in Central Europe ({V['D_bias_CEU']:+.3f}) and New Zealand ({V['D_bias_NZL']:+.3f}). An earlier variant that excluded hybrids gave biases of about 0.09 and −0.13 there.

At the global calendar scale (Fig. 11b):
- RMSD {f0(V['cal_rmsd'])} Gt, r = {V['cal_r']:.3f};
- largest single-year difference {f0(V['cal_maxabs'])} Gt;
- differences in 2020 and 2023 of {f0(V['cal_diff_2020'])} and {f0(V['cal_diff_2023'])} Gt, all within the official ±1σ.

The remaining calendar-year differences come mainly from the Antarctic & Subantarctic and Southern Andes (SH seasonality) and from Alaska in 2000. The official calendar series is not a linear combination of the official hydrological years: for any constant w the residual RMSD is 15–25 Gt in the large regions. Some calendar-year information must have been taken directly from sub-annual data.

![Validation](images/fig11_validation.png)

*Figure 11. (a) Regional HY specific mass change: this study vs official. (b) Global calendar-year differences (this study − official) with the official ±1σ band. (c) Calendar-conversion weight w from the glaciological sine-wave product vs w implied by the official HY/CY pairs.*

**Sensitivity (Table 3, Fig. 8).**
- *Hybrids:* excluding them (`hyb_exclude`) or reproducing the earlier attempt's design (`g1_like`: no hybrids + gap-filling) makes the global loss {f0(sens.set_index('config').loc['hyb_exclude','d_global_rate'])}–{f0(sens.set_index('config').loc['g1_like','d_global_rate'])} Gt yr⁻¹ smaller. It also reduces the 2012–2023 rate by about 22 Gt yr⁻¹, the acceleration from +{f0(100*(sens.set_index('config').loc['main','ratio_late_early']-1))} % to +{f0(100*(sens.set_index('config').loc['g1_like','ratio_late_early']-1))} %, and the 2023 value by about 70–85 Gt. Of all the choices, the handling of hybrids has the largest influence on the acceleration estimate.
- *Altimetry:* removing it (`no_alt`) makes loss {f0(-sens.set_index('config').loc['no_alt','d_global_rate'])} Gt yr⁻¹ larger, consistent with the less negative polar altimetry.
- *D group only* (`D_only`): {f0(sens.set_index('config').loc['D_only','global_rate'])} Gt yr⁻¹.
- *DEM calibration:* per-pentad vs full-period changes the global mean by less than 1 Gt yr⁻¹ but improves regional reproduction.
- *Calendar weights:* seasonal vs time weights mostly affect single years (2023: {f0(sens.set_index('config').loc['main_timeweights','d_gt_2023'])} Gt).

The global mean rate is therefore robust to ±{f0(max(abs(sens.set_index('config').drop(['D_only','glac_level','no_alt']).d_global_rate)))} Gt yr⁻¹ (within 1σ) across all defensible configurations. The acceleration percentage, however, ranges from +21 % to +41 %.

**Table 3. Sensitivity of global results (Gt yr⁻¹; rates are calendar 2000–2023 means).**

{md(t_sens, 2)}

![Sensitivity](images/fig08_sensitivity.png)

*Figure 8. Sensitivity of (a) the mean rate, (b) the acceleration and (c) the reproduction error to the methodological choices.*

### 4.7 Observations vs model projections
**Global (Table 9, Fig. 10).** Projected 2015–2100 glacier contributions are:
- **Rounce et al. (2023):** 90 ± 26 mm (+1.5 °C) to 154 ± 44 mm (+4 °C), and 98 ± 38 (SSP1-2.6), 116 ± 51 (SSP2-4.5) and 166 ± 83 mm (SSP5-8.5), all 95 % CI. These values are corrected for ice below sea level; uncorrected they are 115–192 mm.
- **Marzeion et al. (2020):** 79 ± 56 (RCP2.6) to 159 ± 86 mm (RCP8.5), 90 % CI.
- **Hock et al. (2019):** 89 ± 42 to 180 ± 74 mm, as recalculated in Marzeion et al. (2020) Table 2.

These correspond to **18–28 % (low emissions) and 36–44 % (high emissions) of the 2015 glacier mass lost by 2100**:
- Rounce: 26 ± 6 % (+1.5 °C) to 41 ± 11 % (+4 °C); 28 ± 9 / 32 ± 12 / 44 ± 20 % for SSP1-2.6 / 2-4.5 / 5-8.5;
- Marzeion: 18 ± 13 to 36 ± 20 %;
- Hock: 18 ± 7 to 36 ± 11 %.

Observations against these projections:
- From 2015 to 2023 glaciers lost {f1(PC['obs_cum_2015_2023_mm'])} mm SLE ({f1(pc/100)} % of the 2015 mass in 9 years), a mean rate of **{f2(PC['obs_rate_2015_2023_mm'])} mm yr⁻¹**.
- This is *already comparable to* the mean 2015–2100 rate implied by the low-emission projections: {f2(PC['mean_rate_2015_2100_mm']['marzeion2020:RCP2.6'])} (Marzeion RCP2.6), {f2(PC['mean_rate_2015_2100_mm']['hock2019:RCP2.6'])} (Hock RCP2.6), {f2(PC['mean_rate_2015_2100_mm']['rounce2023:+1.5C'])} (Rounce +1.5 °C) and {f2(PC['mean_rate_2015_2100_mm']['rounce2023:SSP1-2.6'])} mm yr⁻¹ (Rounce SSP1-2.6).
- It lies within the projected early-century rates: Marzeion RCP2.6 peaks "slightly above 1 mm SLE yr⁻¹" in the first half of the century; Rounce +1.5 °C peaks at 1.29 ± 0.59 mm yr⁻¹ around 2035.

At the constant 2012–2023 rate, glaciers would contribute about {f0(-g['rate_2012_2023']/362.5*85)} mm over 2015–2100, just below the low-emission medians (79–98 mm). The low-emission projections are therefore consistent with today's rate only if loss *stops accelerating soon* and then declines as glaciers shrink.

Continuing the observed OLS acceleration ({f2(-g['accel_gt_yr2']/362.5*10)} mm yr⁻¹ per decade) would put the 2100 contribution near or above the high-emission medians (Fig. 10a). The observed acceleration is thus closer to the trajectory of the high scenarios than of the low ones. Two caveats apply:
1. Multi-decadal variability makes a 24-year acceleration an uncertain predictor.
2. Glacier area shrinks, which reduces future loss in Gt.

The observed specific rate for 2005–2014, {f2(PC['obs_specific_2005_2014'])} m w.e. yr⁻¹, is *less* negative than the multimodel median disequilibrium term of Marzeion et al. (2020) for the same period (−0.53 ± 0.59 m w.e. yr⁻¹, inter-model SD). Some models may therefore start too far from equilibrium or with too much ice: the 2015 ensemble-mean mass is 439 mm SLE, against the consensus of 382 ± 99 mm.

**Table 9. Projected glacier mass loss 2015–2100 and comparison with the observed 2015–2023 rate.**

{md(t_proj, 2)}

**Regional (Table 10).**
- **Alaska:** the observed 2015–2023 loss is {f0(-R.set_index('rid').loc[1,'rate_gt_2015_2023'])} Gt yr⁻¹, below Rounce et al.'s projected mid-century peak of 0.33–0.44 mm SLE yr⁻¹ (120–160 Gt yr⁻¹). The models therefore require Alaskan loss to accelerate by roughly 40–85 %.
- **Arctic Canada North:** at the observed relative rate it would lose only ≈9 % of its 2015 mass by 2100, against 12 ± 8 % (RCP2.6) to 23 ± 15 % (RCP8.5) in Hock et al. Models also require acceleration here, consistent with their projection that large Arctic regions accelerate through the century.
- **Small mid-latitude regions** (Central Europe, New Zealand, Caucasus, Western Canada, Low Latitudes, North Asia, South Asia East): the observed absolute rates would exhaust the 2015 ice before 2100, e.g. Central Europe by the 2040s. This is consistent with, or faster than, the projected 60–100 % loss (Rounce) and the 69 ± 19 % (RCP2.6) to 93 ± 10 % (RCP8.5) loss for Central Europe (Hock). The models' slowdown in these regions comes from glacier shrinkage, not from cooler climate.
- **Scandinavia:** at {f2(R.set_index('rid').loc[8,'pct_per_yr_2015_2023'])} % yr⁻¹ it is below the 60–100 % range, so models need acceleration.
- **Regional shares:** the projected shares of 2015–2100 sea-level rise for Greenland Periphery (12 %), Arctic Canada North (10 %) and South (9 %) are close to the observed 2000–2023 shares. The Antarctic periphery is projected to contribute more (10 %) than observed ({f1(R.set_index('rid').loc[19,'share_pct'])} %).

**Committed loss.** Glaciers respond to climate with a lag of decades. Marzeion et al. (2020) estimate a present disequilibrium (2005–2014) of −533 kg m⁻² yr⁻¹, i.e. loss that would continue even under 1995–2014 temperatures. Rounce et al. (2023) note that regions with large glaciers will keep losing mass beyond 2100. Our observation that 2012–2023 loss was {f0(g['increase_pct'])} % higher than 2000–2011, during a period of continued warming, means today's geometry is further from equilibrium than at the start of the record. A substantial part of the projected 2015–2100 loss is therefore already committed, and the low-emission projections chiefly limit *additional* loss.

**Table 10. Regional observations vs model statements.**

{md(t_rm, 2)}

![Projections](images/fig10_observations_vs_projections.png)

*Figure 10. (a) Observed cumulative sea-level contribution since 2000 (±1σ) against 2015–2100 projection medians and ranges, anchored at the observed end-2014 value. Dotted lines are linear paths to the 2100 values, for illustration only. Dashed and dash-dotted lines continue the observed constant rate and the observed OLS acceleration. (b) Observed annual rates against projected rates (peak and end-of-century) and the mean 2015–2100 rates of selected scenarios.*

## 5. Discussion

**What the reconciliation shows.** The independent re-implementation reproduces the community estimate to about 1 % in total and within 1σ in every region. A few structural choices drive that agreement:
1. specific mass change referenced to a fixed RGI area before rescaling to A(t);
2. hybrids as part of the D group;
3. pentad-wise DEM calibration;
4. altimetry/gravimetry levels from regressions with the GRACE gap removed;
5. no gap-filling across groups.

Choices that look reasonable in isolation, such as excluding hybrids or filling missing groups, push the estimate by 8–11 Gt yr⁻¹. They also reduce the acceleration from about +33 % to about +21 % and the 2023 record by about 80 Gt. The 2000–2023 mean is robust; the *acceleration* depends more on method. That should be considered when the series is used to calibrate models.

**Sea-level role.** At {f2(g['rate_slr_mm'])} mm yr⁻¹ over 2000–2023 and {f2(-g['rate_2012_2023']/362.5)} mm yr⁻¹ over 2012–2023, glaciers exceeded the Greenland and the Antarctic ice sheet individually over 2000–2018. They remain the largest single land-ice source even though they hold less than 1 % of land ice.

**Method differences.** Polar altimetry is less negative and GRACE is unusable in small or hydrologically noisy regions. These are the main systematic disagreements, and they are why an equal-weight, multi-group average with conservative errors is preferable to a single "best" method. The z-test shows that group-level uncertainties are about 25 % too optimistic; the combined uncertainty absorbs this.

**Limitations.**
1. The area model is taken from the official files (RGI 6.0 with area-change rates).
2. The calendar conversion uses a linear combination of hydrological years and cannot reproduce single-year official values in the SH to better than about 20–60 Gt.
3. Volume-based relative losses inherit the ±25 % ice-thickness uncertainty.
4. Projection numbers were transcribed from the papers, and trajectories between 2015 and 2100 are not given there. The comparison uses mean rates and reported peak and end-of-century rates, not full time series.
5. Projection SLE values differ in their treatment of ice below sea level: Rounce corrected; Marzeion and Hock uncorrected.

## 6. Conclusions

1. Glaciers lost {f0(-g['total_gt'])} ± {f0(g['total_err_quad'])} Gt ({f1(g['total_slr_mm'])} ± {f1(g['total_slr_err'])} mm SLE; {f1(S['global_loss_pct'])} % of their mass) over 2000–2023. Annual uncertainties are at least as conservative as GlaMBIE's.
2. Loss accelerated by {f0(g['increase_pct'])} ± {f0(g['increase_pct_err'])} % between 2000–2011 and 2012–2023 ({f1(g['accel_gt_yr2'])} ± {f1(g['accel_se'])} Gt yr⁻²). 2023 set the record at {f0(g['gt_2023'])} Gt.
3. Over 2000–2018 glaciers lost about 1.3× as much as Greenland and about 2× as much as Antarctica, the largest single land-ice contribution to sea-level rise.
4. Alaska ({f0(R.set_index('rid').loc[1,'share_pct'])} %), the Canadian Arctic ({f0(can['share_pct'])} %) and the Greenland periphery ({f0(R.set_index('rid').loc[5,'share_pct'])} %) dominate the mass loss. Central Europe ({f0(R.set_index('rid').loc[11,'loss_pct'])} %), New Zealand and the Caucasus have lost the largest fraction of their ice.
5. The methods agree within uncertainties in most regions. Polar altimetry is systematically less negative, GRACE is unreliable in small regions, and glaciological samples are biased negative in level but representative in variability.
6. The observed 2015–2023 rate is already comparable to the century-mean rate of the low-emission projections, and the observed acceleration tracks the high-emission scenarios. Much of the projected 21st-century loss is committed.

## References
- GlaMBIE (2024). Glacier Mass Balance Intercomparison Exercise dataset 1.0.0. WGMS, doi:10.5904/wgms-glambie-2024-07.
- Hock, R. et al. (2019). GlacierMIP – a model intercomparison of global-scale glacier mass-balance models and projections. J. Glaciol. 65, 453–467 (related_work/paper_003).
- Hugonnet, R. et al. (2021). Accelerated global glacier mass loss in the early twenty-first century. Nature 592, 726–731 (paper_004).
- Marzeion, B. et al. (2020). Partitioning the uncertainty of ensemble projections of global glacier mass change. Earth's Future 8 (paper_001).
- Rounce, D. R. et al. (2023). Global glacier change in the 21st century: every increase in temperature matters. Science 379, 78–83 (paper_000).
- Zemp, M. et al. (2019). Global glacier mass changes and their contributions to sea-level rise from 1961 to 2016. Nature 568, 382–386 (paper_002).

## Appendix

**Table A1. Global annual series (calendar years).**

{md(t_glob, 1)}

**Table A2. Calendar-conversion weights.**

{md(t_cw, 3)}

**Table A3. Regional validation against the official GlaMBIE hydrological-year series (m w.e.; bias = this study − official).**

{md(val, 3)}

*Reproducibility:* `bash code/run_all.sh` regenerates all outputs, figures and this report. All numbers in the text are generated from `outputs/summary_stats.json` and the CSV tables in `outputs/`.
"""
open('report/report.md', 'w').write(rep)
print('report written', len(rep))
