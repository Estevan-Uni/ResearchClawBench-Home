"""GlaMBIE-style reconciliation library.

Units convention
----------------
All datasets are first converted to *m w.e. relative to the RGI 6.0 regional area*
("mwe_rgi"), which is proportional to Gt (Gt = mwe_rgi * A_RGI * 1e-3).  Combination is
performed in this unit (equivalent to combining in Gt).  Final specific mass change is
expressed relative to the time-varying glacier area A(t):  mwe(t) = Gt * 1e3 / A(t).

Hydrological-year (HY) grid
---------------------------
HY index i spans [i + off, i + 1 + off] with off = -0.25 (regions 1-15, 1 Oct),
0 (region 16, 1 Jan), +0.25 (regions 17-19, 1 Apr).  For NH regions HY i therefore
corresponds to the official hydrological year ending 30 Sep of year i; for SH regions
HY i is the official hydrological year starting 1 Apr of year i.
"""
import glob, os, re
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, 'data', 'glambie')
RHO, RHO_ERR = 0.850, 0.060          # density of volume change (t m-3)
YEARS = np.arange(1999, 2025)          # internal HY index grid
GT_PER_MM = 362.5                      # Gt per mm sea-level equivalent

REGIONS = ['1_alaska', '2_western_canada_us', '3_arctic_canada_north', '4_arctic_canada_south',
           '5_greenland_periphery', '6_iceland', '7_svalbard', '8_scandinavia', '9_russian_arctic',
           '10_north_asia', '11_central_europe', '12_caucasus_middle_east', '13_central_asia',
           '14_south_asia_west', '15_south_asia_east', '16_low_latitudes', '17_southern_andes',
           '18_new_zealand', '19_antarctic_and_subantarctic']
LABELS = ['Alaska', 'Western Canada & USA', 'Arctic Canada North', 'Arctic Canada South',
          'Greenland Periphery', 'Iceland', 'Svalbard & Jan Mayen', 'Scandinavia', 'Russian Arctic',
          'North Asia', 'Central Europe', 'Caucasus & Middle East', 'Central Asia', 'South Asia West',
          'South Asia East', 'Low Latitudes', 'Southern Andes', 'New Zealand',
          'Antarctic & Subantarctic']
# RGI 6.0 regional areas (km2) and volumes (km3), Zemp et al. 2019 Table 1 (paper_002, p.3)
RGI_AREA = np.array([86725, 14524, 105111, 40888, 89717, 11060, 33959, 2949, 51592, 2410, 2092, 1307,
                     49303, 33568, 14734, 2341, 29429, 1162, 132867], float)
RGI_VOL = np.array([18429, 1048, 29721, 8948, 15780, 3520, 8076, 306, 15449, 146, 116, 63, 3483, 3092,
                    906, 80, 5518, 61, 46801], float)
GRAV_REGIONS_OFFICIAL = {1, 3, 4, 6, 7, 9, 17}
SOURCES = ['glaciological', 'demdiff', 'altimetry', 'gravimetry', 'combined']


def rid_of(reg):
    return int(reg.split('_')[0])


def hy_offset(rid):
    return -0.25 if rid <= 15 else (0.0 if rid == 16 else 0.25)


# ---------------------------------------------------------------- area model
def area_model(reg):
    """Linear area model A(t) fitted to the auxiliary RGI6-based area column of the official
    hydrological-year files (used only as auxiliary area information)."""
    o = pd.read_csv(os.path.join(DATA, 'results', 'hydrological_years', reg + '.csv'))
    tm = 0.5 * (o.start_dates + o.end_dates)
    b, a = np.polyfit(tm, o.glacier_area, 1)
    return lambda t: a + b * np.asarray(t, float)


# ---------------------------------------------------------------- loading
def team_of(author, name):
    return author.strip()


def load_dataset(path, reg):
    rid = rid_of(reg)
    d = pd.read_csv(path)
    name = os.path.basename(path)[:-4]
    src = [s for s in SOURCES if f'_{s}_' in name][0]
    unit = str(d.unit.iloc[0]).lower()
    ch = d.changes.astype(float).values.copy()
    er = d.errors.astype(float).values.copy()
    dur = (d.end_dates - d.start_dates).values
    # fill missing / zero errors with dataset median (Palsson hybrids: zero errors -> 0.3 m w.e./yr)
    bad = ~np.isfinite(er) | (er <= 0)
    bad &= np.isfinite(ch)
    if bad.all():
        er[:] = 0.3 * np.sqrt(np.maximum(dur, 1.0)) if unit in ('mwe', 'm') else np.nan
    elif bad.any():
        er[bad] = np.nanmedian(er[~bad & np.isfinite(ch)])
    if unit == 'm':           # elevation change -> m w.e. with density 850 +/- 60
        mwe = ch * RHO
        err = np.sqrt((RHO * er) ** 2 + (ch * RHO_ERR) ** 2)
    elif unit == 'mwe':
        mwe, err = ch, er
    elif unit == 'gt':        # Gt -> m w.e. relative to RGI area
        mwe = ch * 1e3 / RGI_AREA[rid - 1]
        err = er * 1e3 / RGI_AREA[rid - 1]
    else:
        raise ValueError(unit)
    out = pd.DataFrame(dict(t0=d.start_dates.values, t1=d.end_dates.values, v=mwe, e=err))
    step = np.median(dur)
    if step < 0.2:
        kind = 'monthly'
    elif abs(step - 1) < 0.05 and len(d) >= 3:
        kind = 'annual'
    else:
        kind = 'trend'
    return dict(name=name, src=src, unit=unit, kind=kind, author=d.author.iloc[0],
                team=team_of(d.author.iloc[0], name), data=out, rid=rid, reg=reg)


def load_region(reg):
    fs = sorted(glob.glob(os.path.join(DATA, 'input', reg, '*.csv')))
    return [load_dataset(f, reg) for f in fs]


# ---------------------------------------------------------------- annualisation
def _harmonic_slope(t, c, seg):
    """Least-squares trend of a cumulative series with annual + semi-annual harmonics and one
    offset per contiguous segment (e.g. GRACE / GRACE-FO gap)."""
    cols = [t - t.mean()]
    for k in (1, 2):
        cols += [np.sin(2 * np.pi * k * t), np.cos(2 * np.pi * k * t)]
    for s in np.unique(seg):
        cols.append((seg == s).astype(float))
    X = np.column_stack(cols)
    beta, *_ = np.linalg.lstsq(X, c, rcond=None)
    return beta[0]


GRACE_GAP = (2017.40, 2018.49)


def annualise_monthly(ds, robust_level=True, smooth=0, err_scale=None):
    """Monthly increments -> HY annual changes (mwe_rgi).
    Returns (DataFrame indexed by HY index with columns v, e, f; list of extra trend records
    bridging data gaps).  Glaciological: >= 10 valid months, errors summed in quadrature
    (increments).  Altimetry/gravimetry: 12 valid months required, per-epoch errors -> RMS
    scaled by err_scale; the level of the annual series is set by an OLS regression of the
    cumulative series (trend + annual + semi-annual harmonics, segment offsets)."""
    off = hy_offset(ds['rid'])
    d = ds['data'].sort_values('t0').reset_index(drop=True)
    t0 = np.round(d.t0.values * 12) / 12
    v, e = d.v.values.copy(), d.e.values.copy()
    glac = ds['src'] == 'glaciological'
    raw_ok = np.isfinite(v)
    if ds['src'] == 'gravimetry':
        gap = (t0 >= GRACE_GAP[0]) & (t0 < GRACE_GAP[1])
        v[gap] = np.nan
    ok = np.isfinite(v)
    if err_scale is None:
        err_scale = 1.0
    seg = np.cumsum(~ok)
    cum = np.cumsum(np.where(raw_ok, d.v.values, 0.0))   # cumulative incl. interpolated gap
    tend = t0 + 1 / 12
    rows = {}
    need = 10 if glac else 12
    for i in YEARS:
        a, b = i + off, i + 1 + off
        m = (t0 >= a - 1e-6) & (t0 < b - 1e-6)
        nvalid = (m & ok).sum()
        if nvalid < need:
            continue
        val = np.nansum(v[m & ok]) * 12 / nvalid
        if glac:
            err = np.sqrt(np.nansum(e[m & ok] ** 2))
        else:
            err = np.sqrt(np.nanmean(e[m & ok] ** 2)) * err_scale
        rows[i] = (val, err, 1.0)
    out = pd.DataFrame(rows, index=['v', 'e', 'f']).T
    extra = []
    if len(out) == 0:
        return out, extra
    if robust_level and not glac and len(out) >= 3:
        slope = _harmonic_slope(tend[ok], np.cumsum(np.where(ok, v, 0.0))[ok], seg[ok])
        out['v'] = out['v'] - out['v'].mean() + slope
    # bridge internal gaps (e.g. GRACE/GRACE-FO) with a trend record if the dataset provides
    # (interpolated) values through the gap
    yrs = out.index.values.astype(int)
    for y0, y1 in zip(yrs[:-1], yrs[1:]):
        if y1 - y0 > 1:
            ta, tb = y0 + 1 + off, y1 + off
            ia = np.searchsorted(tend, ta - 1e-6); ib = np.searchsorted(tend, tb - 1e-6)
            if ia < len(tend) and ib < len(tend) and raw_ok[ia:ib + 1].all():
                dv = cum[ib] - cum[ia]
                extra.append((ta, tb, dv, np.sqrt(np.nanmean(e ** 2)) * err_scale * np.sqrt(tb - ta)))
    return out, extra


def annual_from_annual(ds):
    """Annual-resolution datasets: map each record onto the HY grid by overlap."""
    off = hy_offset(ds['rid'])
    d = ds['data']
    rows = {}
    for _, r in d.iterrows():
        if not np.isfinite(r.v):
            continue
        i = int(np.round(r.t0 - off))
        if abs((r.t0 - off) - i) < 0.05 and abs(r.t1 - r.t0 - 1) < 0.05:
            rows[i] = (r.v, r.e, 1.0)
        else:  # mis-aligned: treat as trend later
            return None
    return pd.DataFrame(rows, index=['v', 'e', 'f']).T.loc[lambda x: x.index.isin(YEARS)]


def overlap_fractions(t0, t1, off):
    """fraction of each HY covered by period [t0, t1]"""
    a = YEARS + off
    b = a + 1
    return np.clip(np.minimum(b, t1) - np.maximum(a, t0), 0, 1)


def trend_records(ds, pentad=True):
    """List of (t0, t1, change, error) records of a trend dataset. If pentad=False, contiguous
    sub-periods of the same dataset are merged into one full-period change."""
    d = ds['data'].dropna(subset=['v']).sort_values('t0')
    recs = list(zip(d.t0, d.t1, d.v, d.e))
    if not pentad and len(recs) > 1:
        contiguous = np.allclose(d.t0.values[1:], d.t1.values[:-1], atol=1e-3)
        if contiguous:
            recs = [(d.t0.min(), d.t1.max(), d.v.sum(), np.sqrt((d.e ** 2).sum()))]
    return recs


# ---------------------------------------------------------------- group combination
def calibrate(shape, recs, off):
    """For each trend record find offset c so that sum_y f_y (shape_y + c) = change.
    Returns per-HY offset contributions (array over YEARS, NaN where not covered by >= 0.5)
    and per-year rate errors."""
    cs = np.full(len(YEARS), np.nan)
    es = np.full(len(YEARS), np.nan)
    for t0, t1, dv, de in recs:
        f = overlap_fractions(t0, t1, off)
        sel = f > 0
        if not np.isfinite(shape[sel]).all():
            ok = sel & np.isfinite(shape)
            if f[ok].sum() < 0.8 * f[sel].sum():
                continue
            sel = ok
        c = (dv - np.sum(f[sel] * shape[sel])) / f[sel].sum()
        cov = f > 0.5 + 1e-6
        cs[cov] = c
        es[cov] = de / max(t1 - t0, 1.0)  # error of the mean annual rate (systematic within period)
    return cs, es


def team_mean(rows):
    """rows: list of (team, array). Average within team first, then across teams (NaN-aware)."""
    if not rows:
        return np.full(len(YEARS), np.nan), np.zeros(len(YEARS))
    teams = {}
    for t, a in rows:
        teams.setdefault(t, []).append(a)
    tm = [np.nanmean(np.vstack(v), 0) if len(v) > 1 else v[0] for v in teams.values()]
    with np.errstate(all='ignore'):
        M = np.vstack(tm)
        return np.nanmean(M, 0), np.sum(np.isfinite(M), 0)


def anomaly(a, ref=(2000, 2019)):
    m = (YEARS >= ref[0]) & (YEARS <= ref[1]) & np.isfinite(a)
    return a - np.nanmean(a[m])


# ---------------------------------------------------------------- region processing
DEFAULT_CFG = dict(hybrid='D',            # 'exclude' | 'D' | 'H' (separate group)
                   hyb_vote='year',        # annual hybrids vote on level 'year'-by-year or 'period'
                   dem_pentad=True,        # True: each DEM sub-period (pentad) calibrated separately
                   grav='official',        # 'official' (regions 1,3,4,6,7,9,17) | 'all'
                   gapfill=False,          # fill missing group-years with D + mean offset
                   glac_level=False,       # glaciological series vote on absolute level
                   robust_level=True,      # harmonic regression level for monthly alti/grav
                   ref=(2000, 2019),
                   err_mode='official_like',
                   mon_err_scale=1 / np.sqrt(12))


def annual_series(ds, cfg, with_extra=False):
    extra = []
    if ds['kind'] == 'monthly':
        a, extra = annualise_monthly(ds, robust_level=cfg['robust_level'],
                                     err_scale=cfg['mon_err_scale'] if ds['src'] == 'gravimetry' else 1.0)
    elif ds['kind'] == 'annual':
        a = annual_from_annual(ds)
    else:
        a = None
    return (a, extra) if with_extra else a


def _arr(df, col):
    a = np.full(len(YEARS), np.nan)
    if df is None or len(df) == 0:
        return a
    idx = np.searchsorted(YEARS, df.index.values.astype(int))
    ok = (idx < len(YEARS)) & (YEARS[np.clip(idx, 0, len(YEARS) - 1)] == df.index.values)
    a[idx[ok]] = df[col].values[ok]
    return a


def _ffill_nearest(a):
    a = a.copy()
    ok = np.isfinite(a)
    if not ok.any():
        return a
    idx = np.where(ok)[0]
    for k in np.where(~ok)[0]:
        a[k] = a[idx[np.argmin(np.abs(idx - k))]]
    return a


def build_group(datasets, shape_abs, shape_err, off, cfg, var_only=()):
    """Generic group builder.
    datasets : list of loaded datasets belonging to the group
    shape_abs: absolute annual shape used to distribute trends where the group itself has no
               annual-resolution data (NaN allowed)
    Returns dict(v, e, n, var) arrays over YEARS (v NaN where group not covered)."""
    ann = []      # (team, values, errors) annual datasets that are level-bearing
    trends = []
    extra_recs = []
    for ds in datasets:
        a, extra = annual_series(ds, cfg, with_extra=True)
        if a is not None and len(a) >= 2:
            ann.append((ds['team'], _arr(a, 'v'), _arr(a, 'e'), ds))
            if extra:
                extra_recs.append((ds, extra))
        else:
            trends.append(ds)
    # shape: mean of the group's own annual datasets where available, else external shape
    if ann:
        own, _ = team_mean([(t, v) for t, v, e, d in ann])
        own_e, _ = team_mean([(t, e) for t, v, e, d in ann])
    else:
        own = np.full(len(YEARS), np.nan)
        own_e = np.full(len(YEARS), np.nan)
    var_flag = np.isfinite(own)
    shape = np.where(var_flag, own, shape_abs)
    shp_e = np.where(var_flag, own_e, shape_err)
    votes, verrs = [], []
    for t, v, e, d in ann:
        votes.append((t, v))
        verrs.append((t, e))
    items = [(ds, trend_records(ds, pentad=cfg['dem_pentad'] if ds['src'] == 'demdiff' else True)) for ds in trends]
    for ds, recs in items + extra_recs:
        cs, es = calibrate(shape, recs, off)
        if np.isfinite(cs).any():
            votes.append((ds['team'], shape + cs))
            verrs.append((ds['team'], np.sqrt(es ** 2 + np.nan_to_num(shp_e) ** 2)))
    if not votes:
        return None
    v, n = team_mean(votes)
    # RMS of member errors (members are not independent -> no 1/sqrt(N) reduction) plus spread
    e2, _ = team_mean([(t, e ** 2) for t, e in verrs])
    V = np.vstack([a for _, a in votes])
    with np.errstate(all='ignore'):
        spread = np.where(np.sum(np.isfinite(V), 0) > 1, np.nanstd(V, 0, ddof=1), 0.0)
    nn = np.maximum(np.sum(np.isfinite(V), 0), 1)
    nt = np.maximum(n, 1)
    e = np.sqrt(e2 / nt + spread ** 2 / nn)       # standard error of the team mean
    v = np.where(n > 0, v, np.nan)
    return dict(v=v, e=e, n=n, var=var_flag & np.isfinite(v))


def build_D(datasets, off, cfg):
    """DEM differencing + glaciological group: glaciological (and annual hybrid) series provide
    the annual variability (anomalies), DEM differencing (and hybrids) provide the level."""
    glac = [d for d in datasets if d['src'] == 'glaciological']
    dem = [d for d in datasets if d['src'] == 'demdiff']
    hyb = [d for d in datasets if d['src'] == 'combined'] if cfg['hybrid'] == 'D' else []
    var_rows, var_err, lev_votes, lev_err = [], [], [], []
    hyb_ann = []
    for ds in glac + hyb:
        a = annual_series(ds, cfg)
        if a is None or len(a) < 3:
            if ds['src'] == 'combined':
                dem.append(ds)  # trend-type hybrid -> level only
            continue
        v, e = _arr(a, 'v'), _arr(a, 'e')
        var_rows.append((ds['team'], anomaly(v, cfg['ref'])))
        var_err.append((ds['team'], e))
        if ds['src'] == 'combined' or cfg['glac_level']:
            hyb_ann.append((ds['team'], v, e))
    V, nV = team_mean(var_rows)
    Ve, _ = team_mean(var_err)
    Ve = Ve / np.sqrt(np.maximum(nV, 1)) if False else Ve
    # level votes
    for ds in dem:
        recs = trend_records(ds, pentad=cfg['dem_pentad'])
        cs, es = calibrate(V, recs, off)
        if np.isfinite(cs).any():
            lev_votes.append((ds['team'], cs))
            lev_err.append((ds['team'], es))
    for t, v, e in hyb_ann:
        if cfg['hyb_vote'] == 'year':
            lev_votes.append((t, v - V))
        else:
            ok = np.isfinite(v) & np.isfinite(V)
            c = np.where(np.isfinite(v), np.nanmean((v - V)[ok]), np.nan)
            lev_votes.append((t, c))
        lev_err.append((t, np.where(np.isfinite(v), e, np.nan)))
    C, nC = team_mean(lev_votes)
    Cf = _ffill_nearest(C)
    E2, _ = team_mean([(t, e ** 2) for t, e in lev_err])
    E2 = _ffill_nearest(E2)
    Cmat = np.vstack([c for _, c in lev_votes])
    with np.errstate(all='ignore'):
        spread = np.where(np.sum(np.isfinite(Cmat), 0) > 1, np.nanstd(Cmat, 0, ddof=1), 0.0)
    v = V + Cf
    # extrapolation beyond the last level observation: add growing persistence error
    extrap = ~np.isfinite(C)
    lastobs = np.where(np.isfinite(C))[0]
    ext_e = np.zeros(len(YEARS))
    if len(lastobs):
        for k in np.where(extrap)[0]:
            dist = np.min(np.abs(lastobs - k))
            ext_e[k] = 0.1 * np.sqrt(dist)          # 0.1 m w.e. per sqrt(yr) of extrapolation
    nCf = np.maximum(_ffill_nearest(np.where(nC > 0, nC, np.nan)), 1)
    nVf = np.maximum(nV, 1)
    e = np.sqrt(np.nan_to_num(Ve) ** 2 / nVf + E2 / nCf + spread ** 2 / nCf + ext_e ** 2)
    v = np.where(np.isfinite(V), v, np.nan)
    return dict(v=v, e=e, n=nC, var=np.isfinite(V), shape=V, offset=Cf, shape_e=Ve)


def process_region(reg, cfg=None, datasets=None):
    cfg = {**DEFAULT_CFG, **(cfg or {})}
    rid = rid_of(reg)
    off = hy_offset(rid)
    ds_all = datasets if datasets is not None else load_region(reg)
    D = build_D(ds_all, off, cfg)
    groups = {'D': D}
    alt = [d for d in ds_all if d['src'] == 'altimetry']
    grav = [d for d in ds_all if d['src'] == 'gravimetry']
    if cfg['grav'] == 'official' and rid not in GRAV_REGIONS_OFFICIAL:
        grav = []
    if cfg['grav'] == 'none':
        grav = []
    if cfg.get('no_alt'):
        alt = []
    if alt:
        groups['A'] = build_group(alt, D['v'], D['e'], off, cfg)
    if grav:
        groups['G'] = build_group(grav, D['v'], D['e'], off, cfg)
    if cfg['hybrid'] == 'H':
        hyb = [d for d in ds_all if d['src'] == 'combined']
        if hyb:
            groups['H'] = build_group(hyb, D['v'], D['e'], off, cfg)
    if cfg.get('only_D'):
        groups = {'D': D}
    groups = {k: g for k, g in groups.items() if g is not None}
    return combine_groups(groups, cfg), groups


def combine_groups(groups, cfg):
    keys = list(groups)
    V = np.vstack([groups[k]['v'] for k in keys])
    E = np.vstack([groups[k]['e'] for k in keys])
    if cfg['gapfill']:
        D = groups['D']['v']
        for j, k in enumerate(keys):
            if k == 'D':
                continue
            ok = np.isfinite(V[j]) & np.isfinite(D)
            if ok.sum() >= 3:
                off_ = np.mean(V[j][ok] - D[ok])
                miss = ~np.isfinite(V[j]) & np.isfinite(D)
                V[j][miss] = D[miss] + off_
                E[j][miss] = np.sqrt(groups['D']['e'][miss] ** 2 + np.std(V[j][ok] - D[ok]) ** 2)
    with np.errstate(all='ignore'):
        n = np.sum(np.isfinite(V), 0)
        v = np.nanmean(V, 0)
        me2 = np.nanmean(E ** 2, 0)
        sd = np.where(n > 1, np.nanstd(V, 0, ddof=0), 0.0)
    # typical inter-group spread in this region (used where fewer groups are available)
    nmax = len(keys)
    # where a data group is missing, add that group's median error (official-like convention:
    # the combined error does not shrink because fewer independent groups are available)
    add2 = np.zeros(V.shape[1])
    for j in range(len(keys)):
        med = np.nanmedian(E[j][np.isfinite(V[j])]) if np.isfinite(V[j]).any() else 0.0
        add2 += np.where(~np.isfinite(V[j]) & (n > 0), med ** 2, 0.0)
    if cfg['err_mode'] == 'official_like':
        e = np.sqrt(np.nansum(np.where(np.isfinite(V), E ** 2, 0), 0) / np.maximum(n, 1) * nmax / np.maximum(n, 1) * 0 +
                    np.nansum(np.where(np.isfinite(V), E ** 2, 0), 0) + add2)
    elif cfg['err_mode'] == 'quad_mean':
        e = np.sqrt(np.nansum(E ** 2, 0)) / np.maximum(n, 1)
    else:
        e = np.sqrt(me2 + sd ** 2)
    v = np.where(n > 0, v, np.nan)
    return dict(v=v, e=e, n=n)


# ---------------------------------------------------------------- outputs in physical units
GROUP_COL = {'A': 'altimetry', 'G': 'gravimetry', 'D': 'demdiff_and_glaciological', 'H': 'hybrid'}


def region_table(reg, cfg=None, datasets=None):
    """HY table with mwe (relative to A(t)) and Gt for combined and per-group series."""
    import warnings
    warnings.filterwarnings('ignore')
    rid = rid_of(reg)
    off = hy_offset(rid)
    comb, groups = process_region(reg, cfg, datasets)
    A = area_model(reg)
    t0 = YEARS + off
    area = A(t0 + 0.5)
    k = RGI_AREA[rid - 1] / area          # mwe_rgi -> mwe(t)
    df = pd.DataFrame(dict(hy=YEARS, start=t0, end=t0 + 1, area=area))
    for name, g in [('combined', comb)] + [(GROUP_COL[k_], g_) for k_, g_ in groups.items()]:
        df[name + '_mwe'] = g['v'] * k
        df[name + '_mwe_errors'] = g['e'] * k
        df[name + '_gt'] = g['v'] * RGI_AREA[rid - 1] * 1e-3
        df[name + '_gt_errors'] = g['e'] * RGI_AREA[rid - 1] * 1e-3
        df[name + '_n'] = g['n']
    df['n_groups'] = comb['n']
    return df, groups


def official_hy(reg):
    o = pd.read_csv(os.path.join(DATA, 'results', 'hydrological_years', reg + '.csv'))
    o['hy'] = np.round(o.start_dates - hy_offset(rid_of(reg))).astype(int)
    return o.set_index('hy')


def official_cy(reg):
    o = pd.read_csv(os.path.join(DATA, 'results', 'calendar_years', reg + '.csv'))
    o['year'] = np.round(o.start_dates).astype(int)
    return o.set_index('year')


# ---------------------------------------------------------------- calendar conversion
def calendar_table(hy, rid, w, w_err=0.1):
    """Convert a HY table (from region_table) to calendar years for combined Gt and m w.e.
    NH: C(Y) = (1-w) H(Y) + w H(Y+1);  SH: C(Y) = (1-w) H(Y) + w H(Y-1);  region 16: C = H."""
    h = hy.set_index('hy')
    out = []
    for Y in range(2000, 2024):
        if rid == 16:
            nb = Y
        else:
            nb = Y + 1 if rid <= 15 else Y - 1
        ww = 0.0 if rid == 16 else w
        g0, g1 = h.combined_gt.get(Y, np.nan), h.combined_gt.get(nb, np.nan)
        e0, e1 = h.combined_gt_errors.get(Y, np.nan), h.combined_gt_errors.get(nb, np.nan)
        flag = ''
        if not np.isfinite(g1):
            g1, e1, flag = g0, e0, 'persistence'
        val = (1 - ww) * g0 + ww * g1
        conv = w_err * abs(g1 - g0) if rid != 16 else 0.0
        err = np.sqrt(((1 - ww) * e0) ** 2 + (ww * e1) ** 2 + conv ** 2)
        out.append(dict(year=Y, gt_val=val, gt_err=err, conv_err=conv, flag=flag,
                        n_groups=h.n_groups.get(Y, 0)))
    return pd.DataFrame(out).rename(columns={'gt_val': 'gt'})
