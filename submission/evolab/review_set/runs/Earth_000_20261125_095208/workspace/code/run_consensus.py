"""Run the reconciliation for a set of configurations; write regional HY / calendar tables and
global series; compute agreement metrics with the official GlaMBIE results."""
import sys, os, json, numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, 'code'); import glambie_lib as L

OUT = 'outputs'
CONFIGS = {
    # name: config overrides (DEFAULT_CFG: hybrids in D, per-pentad DEM calibration, official gravimetry regions, no gap fill)
    'main':              {},
    'hyb_exclude':       {'hybrid': 'exclude'},
    'hyb_separate':      {'hybrid': 'H'},
    'hyb_period_vote':   {'hyb_vote': 'period'},
    'dem_fullperiod':    {'dem_pentad': False},
    'g1_like':           {'hybrid': 'exclude', 'gapfill': True},
    'grav_all':          {'grav': 'all'},
    'grav_none':         {'grav': 'none'},
    'no_alt':            {'no_alt': True},
    'gapfill':           {'gapfill': True},
    'glac_level':        {'glac_level': True},
    'no_robust_level':   {'robust_level': False},
    'D_only':            {'only_D': True},
}

def load_weights():
    w = pd.read_csv(os.path.join(OUT, 'calendar_weights.csv')).set_index('region')
    return {r: float(np.clip(w.loc[r, 'w_sinewave'] if np.isfinite(w.loc[r, 'w_sinewave']) else w.loc[r, 'w_time'], 0, 1)) for r in L.REGIONS}

def run(name, cfg, cache, weights, wmode='sinewave', save=True):
    cfg_full = {**L.DEFAULT_CFG, **cfg}
    regs_hy, regs_cy = {}, {}
    for reg in L.REGIONS:
        rid = L.rid_of(reg)
        t, groups = L.region_table(reg, cfg_full, cache[reg])
        regs_hy[reg] = t
        w = weights[reg] if wmode == 'sinewave' else (0.0 if rid == 16 else 0.25)
        c = L.calendar_table(t, rid, w)
        A = L.area_model(reg)
        c['area'] = A(c.year + 0.5)
        c['mwe'] = c['gt'] * 1e3 / c['area']; c['mwe_err'] = c['gt_err'] * 1e3 / c['area']
        regs_cy[reg] = c
    yrs = np.arange(2000, 2024)
    G = pd.DataFrame({'year': yrs})
    gt = np.vstack([regs_cy[r].set_index('year')['gt'].reindex(yrs).values for r in L.REGIONS])
    ge = np.vstack([regs_cy[r].set_index('year').gt_err.reindex(yrs).values for r in L.REGIONS])
    assert np.isfinite(gt).all(), f'{name}: missing regional calendar values'
    G['gt'] = gt.sum(0); G['gt_err'] = np.sqrt((ge ** 2).sum(0))
    G['area'] = sum(regs_cy[r].set_index('year').area.reindex(yrs).values for r in L.REGIONS)
    G['mwe'] = G['gt'] * 1e3 / G['area']; G['mwe_err'] = G['gt_err'] * 1e3 / G['area']
    G['slr_mm'] = -G['gt'] / L.GT_PER_MM
    if save:
        d = os.path.join(OUT, 'configs', name); os.makedirs(d, exist_ok=True)
        for r in L.REGIONS:
            regs_hy[r].to_csv(os.path.join(d, f'{r}_hydro.csv'), index=False)
            regs_cy[r].to_csv(os.path.join(d, f'{r}_calendar.csv'), index=False)
        G.to_csv(os.path.join(d, 'global_calendar.csv'), index=False)
    return regs_hy, regs_cy, G

def metrics(name, regs_hy, regs_cy, G):
    og = pd.read_csv('data/glambie/results/calendar_years/0_global.csv')
    og['year'] = np.round(og.start_dates).astype(int); og = og.set_index('year')
    g = G.set_index('year').rename(columns={'gt': 'GT'})
    m = {}
    m['global_total_gt'] = g.GT.sum(); m['official_total_gt'] = og.combined_gt.sum()
    m['global_rate'] = g.GT.mean(); m['global_total_err_quad'] = np.sqrt((g.gt_err ** 2).sum())
    m['rate_2000_2011'] = g.GT.loc[2000:2011].mean(); m['rate_2012_2023'] = g.GT.loc[2012:2023].mean()
    m['ratio_late_early'] = m['rate_2012_2023'] / m['rate_2000_2011']
    m['off_ratio_late_early'] = og.combined_gt.loc[2012:2023].mean() / og.combined_gt.loc[2000:2011].mean()
    m['gt_2023'] = g.GT.loc[2023]; m['off_2023'] = og.combined_gt.loc[2023]
    dif = g.GT - og.combined_gt
    m['cal_rmsd_gt'] = np.sqrt((dif ** 2).mean()); m['cal_maxabs_gt'] = dif.abs().max(); m['cal_r'] = np.corrcoef(g.GT, og.combined_gt)[0, 1]
    m['err_ratio_global'] = (g.gt_err / og.combined_gt_errors).mean()
    # regional HY agreement
    bias, rmsd, eratio = [], [], []
    for reg in L.REGIONS:
        t = regs_hy[reg].set_index('hy'); o = L.official_hy(reg)
        j = pd.concat([t.combined_gt, o.combined_gt.rename('o'), t.combined_gt_errors.rename('e'), o.combined_gt_errors.rename('oe')], axis=1).dropna()
        bias.append((j.combined_gt - j.o).mean()); rmsd.append(np.sqrt(((j.combined_gt - j.o) ** 2).mean())); eratio.append((j.e / j.oe).median())
    m['reg_sum_abs_bias_gt'] = np.sum(np.abs(bias)); m['reg_mean_rmsd_gt'] = np.mean(rmsd); m['reg_median_err_ratio'] = np.median(eratio)
    m['config'] = name
    return m

if __name__ == '__main__':
    names = sys.argv[1:] or list(CONFIGS)
    cache = {r: L.load_region(r) for r in L.REGIONS}
    weights = load_weights()
    res = []
    for n in names:
        out = run(n, CONFIGS[n], cache, weights)
        res.append(metrics(n, *out))
        if n == 'main':
            out2 = run('main_timeweights', CONFIGS[n], cache, weights, wmode='time')
            res.append(metrics('main_timeweights', *out2))
    df = pd.DataFrame(res).set_index('config')
    df.to_csv(os.path.join(OUT, 'config_metrics.csv'))
    pd.set_option('display.width', 250)
    print(df.round(3).T.to_string())
