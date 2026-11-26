"""Analysis + figures from outputs/results_all.csv. Writes outputs/key_numbers.json and report/images/*.png"""
import sys, os, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from figstyle import *
import params as P, chain as C, model as M

OUT = '/workspace/outputs'
R = pd.read_csv(f'{OUT}/results_all.csv', keep_default_na=False, na_values=[''])
HEX = pd.read_csv(f'{OUT}/hex_prepared.csv', keep_default_na=False, na_values=['']).set_index('hex_id')
KN = {}
MOD = 'Moderate 2.15%'
RATES = list(P.BASE_RATES)
SC = P.FIN_SCEN
ON = HEX.index[HEX.onshore.astype(str) == 'True']

def sel(**kw):
    d = R
    for k, v in kw.items():
        d = d[d[k] == v]
    return d

AF = R[R.group == 'af']
EU = R[R.group == 'eu']
EU = EU[EU.kind == 'eu']

def eu_val(site, scen, rate, mode='const', col='delivered'):
    d = EU[(EU.site == site) & (EU.scen == scen) & (EU.rate == rate) & (EU['mode'] == mode)]
    return float(d[col].iloc[0])

def af_tab(mode, scen, rate, col):
    d = AF[(AF['mode'] == mode) & (AF.scen == scen) & (AF.rate == rate)].set_index('site')
    return d[col]

# =========================================================================== Fig 1: data overview
def fig_data():
    fig = plt.figure(figsize=(11, 6.2))
    gs = fig.add_gridspec(2, 4, width_ratios=[1.3, 1.3, 1, 1])
    for k, (col, lab) in enumerate([('cf_pv', 'PV capacity factor (mapped from theo_pv)'), ('cf_wind', 'Wind capacity factor (mapped from theo_wind)')]):
        ax = fig.add_subplot(gs[:, k]); draw_base(ax)
        on = HEX.onshore.astype(str) == 'True'
        sc = ax.scatter(HEX.lon[on], HEX.lat[on], c=HEX[col][on], cmap=SEQ, s=70, ec='white', lw=0.8, zorder=3)
        ax.scatter(HEX.lon[~on], HEX.lat[~on], c=HEX[col][~on], cmap=SEQ, s=70, ec=INK, lw=0.8, marker='D', zorder=3,
                   vmin=HEX[col].min(), vmax=HEX[col].max())
        for p, v in P.PORTS.items():
            if v['lat'] > -30:
                ax.plot(v['lon'], v['lat'], marker='*', ms=11, color=CAT[1], mec='white', zorder=4)
                ax.text(v['lon'] + 0.3, v['lat'] - 0.15, p, fontsize=7.5, color=INK2, zorder=4)
        cb = fig.colorbar(sc, ax=ax, orientation='horizontal', fraction=0.04, pad=0.02); cb.set_label(lab)
        ax.set_title(['(a) Solar resource', '(b) Wind resource'][k], loc='left')
    names = [('grid_dist_km', 'Grid'), ('road_dist_km', 'Road'), ('ocean_dist_km', 'Ocean'), ('waterbody_dist_km', 'Freshwater body')]
    for k, (col, lab) in enumerate(names):
        ax = fig.add_subplot(gs[k // 2, 2 + k % 2])
        ax.hist(HEX[col], bins=10, color=CAT[0], ec=SURF, lw=2)
        ax.set_title(f'({chr(99 + k)}) Distance to {lab.lower()}', loc='left'); ax.set_xlabel('km'); ax.set_ylabel('hexagons')
    fig.text(0.01, -0.02, 'Circles: onshore hexagons (Natural Earth point-in-polygon); diamonds: points falling in the Atlantic (coastal proxies). Stars: candidate export ports.',
             fontsize=7.5, color=INK2)
    fig.tight_layout(); save(fig, 'fig01_data_overview')
    cnt = HEX.ne_iso.value_counts().to_dict()
    KN['n_hex'] = len(HEX); KN['country_counts'] = cnt
    KN['offshore_km_range'] = [float(HEX.offshore_km[HEX.offshore_km > 0].min()), float(HEX.offshore_km.max())]
    KN['cf_pv_range'] = [float(HEX.cf_pv.min()), float(HEX.cf_pv.max())]; KN['cf_wind_range'] = [float(HEX.cf_wind.min()), float(HEX.cf_wind.max())]
    KN['dist_medians'] = {c: float(HEX[c].median()) for c, _ in names}
    # weather sanity
    pv = M.pv_profile(-22.0, 0.22); wd = M.wind_profile(0.30)
    wk = wd[:8736].reshape(-1, 168).mean(1)
    KN['wind_week_p5_over_mean'] = float(np.percentile(wk, 5) / wd.mean())

# =========================================================================== spatial statistics
def morans_i(vals, lat, lon, perms=999, seed=1):
    x = np.asarray(vals, float); n = len(x)
    from prep import haversine
    D = haversine(lat[:, None], lon[:, None], lat[None, :], lon[None, :])
    W = np.where(D > 0, 1 / np.maximum(D, 1), 0); W = W / W.sum(1, keepdims=True)
    def I(v):
        z = v - v.mean(); return n / W.sum() * (z @ W @ z) / (z @ z)
    i0 = I(x); rng = np.random.default_rng(seed)
    sims = np.array([I(rng.permutation(x)) for _ in range(perms)])
    p = (np.sum(np.abs(sims - sims.mean()) >= abs(i0 - sims.mean())) + 1) / (perms + 1)
    return i0, p, -1 / (n - 1)

def spatial_stats(s):
    s = s.dropna(); h = HEX.loc[s.index]
    i, p, e = morans_i(s.values, h.lat.values, h.lon.values)
    q = s.quantile(0.2)
    return dict(min=float(s.min()), max=float(s.max()), median=float(s.median()), mean=float(s.mean()),
                cv=float(s.std() / s.mean()), range_pct=float((s.max() - s.min()) / s.min()),
                moran_I=float(i), moran_p=float(p), moran_E=float(e), top_quintile=list(s[s <= q].index),
                argmin=s.idxmin(), argmax=s.idxmax())

# =========================================================================== Fig 2: production LCOH maps
def fig_production():
    fl = af_tab('flex', 'Commercial', MOD, 'onsite'); co = af_tab('const', 'Commercial', MOD, 'onsite')
    vmin, vmax = min(fl.min(), co.min()), max(fl.max(), co.max())
    fig = plt.figure(figsize=(12, 5.6)); gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.9])
    stats = {}
    for k, (s, lab) in enumerate([(fl, 'Flexible electrolyser (no storage)'), (co, 'Constant supply for Haber–Bosch (H$_2$ storage)')]):
        ax = fig.add_subplot(gs[0, k]); draw_base(ax)
        h = HEX.loc[s.index]; on = h.onshore.astype(str) == 'True'
        sc = ax.scatter(h.lon[on], h.lat[on], c=s[on], cmap=SEQ, vmin=vmin, vmax=vmax, s=85, ec='white', lw=0.8, zorder=3)
        ax.scatter(h.lon[~on], h.lat[~on], c=s[~on], cmap=SEQ, vmin=vmin, vmax=vmax, s=70, ec=INK, lw=0.8, marker='D', zorder=3)
        st = spatial_stats(s[on.values]); stats['flex' if k == 0 else 'const'] = st
        st_all = spatial_stats(s); stats[('flex' if k == 0 else 'const') + '_all30'] = st_all
        for key, tag in [('argmin', 'min'), ('argmax', 'max')]:
            hh = st[key]; ax.annotate(f'{tag} {s[hh]:.2f}', (HEX.loc[hh].lon, HEX.loc[hh].lat), xytext=(8, 8), textcoords='offset points',
                                      fontsize=8, color=INK, arrowprops=dict(arrowstyle='-', color=INK2, lw=0.7), zorder=5)
        ax.set_title(f'({"ab"[k]}) {lab}', loc='left', fontsize=9.5)
        cb = fig.colorbar(sc, ax=ax, orientation='horizontal', fraction=0.04, pad=0.02); cb.set_label('on-site LCOH (€/kg H$_2$), commercial finance, 2.15% base rate')
    ax = fig.add_subplot(gs[0, 2]); ax.axis('off')
    txt = 'Spatial statistics (23 onshore hexagons)\n\n'
    for k, lab in [('flex', 'Flexible'), ('const', 'Constant supply')]:
        st = stats[k]
        txt += (f'{lab}\n  median {st["median"]:.2f} €/kg, range {st["min"]:.2f}–{st["max"]:.2f}\n'
                f'  max/min − 1 = {100 * st["range_pct"]:.0f}%,  CV = {100 * st["cv"]:.1f}%\n'
                f'  Moran\'s I = {st["moran_I"]:.2f} (E[I] = {st["moran_E"]:.2f}, p = {st["moran_p"]:.2f})\n'
                f'  cheapest quintile: {", ".join(x.replace("hex_", "") for x in st["top_quintile"])}\n\n')
    txt += 'n = 23 is small: Moran\'s I has low power;\nresource indices are spatially unstructured\nin this simulated dataset.'
    ax.text(0, 0.98, txt, va='top', fontsize=8.3, family='DejaVu Sans', color=INK)
    fig.tight_layout(); save(fig, 'fig02_production_lcoh_maps')
    KN['prod_stats'] = stats
    # by rate / scenario
    tab = {}
    for mode in ['flex', 'const']:
        for s in SC + ['Zero']:
            for r in (RATES if s != 'Zero' else ['Zero']):
                v = af_tab(mode, s, r, 'onsite').loc[lambda x: x.index.isin(ON)]
                tab[f'{mode}|{s}|{r}'] = dict(min=float(v.min()), median=float(v.median()), max=float(v.max()))
    KN['onsite_table'] = tab
    # resource drivers of production
    d = AF[(AF['mode'] == 'flex') & (AF.scen == 'Commercial') & (AF.rate == MOD)].set_index('site')
    KN['flex_capacities_median'] = dict(Cpv=float(d.Cpv.median()), Cw=float(d.Cw.median()), Cel=float(d.Cel.median()), el_util=float(d.el_util.median()))
    d2 = AF[(AF['mode'] == 'const') & (AF.scen == 'Commercial') & (AF.rate == MOD)].set_index('site')
    KN['const_capacities_median'] = dict(Cpv=float(d2.Cpv.median()), Cw=float(d2.Cw.median()), Cel=float(d2.Cel.median()),
                                         el_util=float(d2.el_util.median()), h2s_hours=float(d2.h2s_hours.median()),
                                         wind_share_gen=float(np.median(d2.Cw * d2.cf_wind / (d2.Cw * d2.cf_wind + d2.Cpv * d2.cf_pv))))
    KN['corr_flex_cfpv'] = float(np.corrcoef(d.loc[ON].cf_pv, d.loc[ON].onsite)[0, 1])
    KN['corr_flex_cfwind'] = float(np.corrcoef(d.loc[ON].cf_wind, d.loc[ON].onsite)[0, 1])
    KN['corr_const_cfpv'] = float(np.corrcoef(d2.loc[ON].cf_pv, d2.loc[ON].onsite)[0, 1])
    KN['corr_const_cfwind'] = float(np.corrcoef(d2.loc[ON].cf_wind, d2.loc[ON].onsite)[0, 1])
    # constant-supply premium decomposition at median hex
    rep = KN['rep_hex']
    a = d.loc[rep]; b = d2.loc[rep]
    KN['const_premium_rep'] = dict(flex=float(a.onsite), const=float(b.onsite),
                                   pv=float((b.Cpv - a.Cpv) * 0 + 0), )
    comp = {}
    for c_ in ['c_water', 'c_road']:
        comp[c_] = float(b[c_])
    KN['rep_flex_vs_const'] = dict(flex=float(a.onsite), const=float(b.onsite), Cel_flex=float(a.Cel), Cel_const=float(b.Cel),
                                   Cpv_flex=float(a.Cpv), Cpv_const=float(b.Cpv), Cw_flex=float(a.Cw), Cw_const=float(b.Cw),
                                   h2s_hours=float(b.h2s_hours), curt_flex=float(a.curtail), curt_const=float(b.curtail))
    # southern / SA-border sites
    south = HEX[HEX.lat < -26].index
    KN['south_sites'] = {h: dict(iso=HEX.loc[h].ne_iso, lat=float(HEX.loc[h].lat), flex=float(d.loc[h].onsite), const=float(d2.loc[h].onsite),
                                 rank_flex=int(d.loc[ON].onsite.rank().get(h, np.nan)) if h in ON else None) for h in south}

# =========================================================================== Fig 3: delivered maps
def fig_delivered_maps():
    vals = {s: af_tab('const', s, MOD, 'delivered') for s in SC}
    vmin = min(v.min() for v in vals.values()); vmax = max(v.max() for v in vals.values())
    de = eu_val('Germany (north)', 'Commercial', MOD); es = eu_val('Spain (+2,000 km pipe)', 'Commercial', MOD)
    fig, axs = plt.subplots(1, 4, figsize=(14, 4.6))
    for ax, s in zip(axs, SC):
        draw_base(ax, labels=False); v = vals[s]; h = HEX.loc[v.index]; on = h.onshore.astype(str) == 'True'
        sc = ax.scatter(h.lon[on], h.lat[on], c=v[on], cmap=SEQ, vmin=vmin, vmax=vmax, s=70, ec='white', lw=0.8, zorder=3)
        ax.scatter(h.lon[~on], h.lat[~on], c=v[~on], cmap=SEQ, vmin=vmin, vmax=vmax, s=55, ec=INK, lw=0.8, marker='D', zorder=3)
        comp = v[on.values] <= de
        ax.scatter(h.lon[on][comp.values], h.lat[on][comp.values], s=160, facecolors='none', ec=CAT[1], lw=1.6, zorder=4)
        vo = v[on.values]
        ax.set_title(f'{s}\nmedian {vo.median():.2f}, min {vo.min():.2f} €/kg', loc='left', fontsize=9)
        for p in P.PRIMARY_PORTS:
            ax.plot(P.PORTS[p]['lon'], P.PORTS[p]['lat'], marker='*', ms=10, color=CAT[1], mec='white', zorder=4)
    cb = fig.colorbar(sc, ax=axs, orientation='horizontal', fraction=0.05, pad=0.03, aspect=50)
    cb.set_label(f'Delivered H$_2$ cost at Rotterdam via NH$_3$ (€/kg), 2.15% base rate.  Orange rings: ≤ German green H$_2$ ({de:.2f} €/kg, commercial)')
    save(fig, 'fig03_delivered_maps')
    tab = {}
    for s in SC + ['Zero']:
        for r in (RATES if s != 'Zero' else ['Zero']):
            v = af_tab('const', s, r, 'delivered').loc[lambda x: x.index.isin(ON)]
            tab[f'{s}|{r}'] = dict(min=float(v.min()), median=float(v.median()), max=float(v.max()), argmin=v.idxmin())
    KN['delivered_table'] = tab
    eu = {}
    for site in P.EU_SITES:
        for s in ['Commercial', 'Offtake']:
            for r in RATES:
                for mode in ['const', 'flex']:
                    eu[f'{site}|{s}|{r}|{mode}'] = eu_val(site, s, r, mode)
        eu[f'{site}|Zero|Zero|const'] = eu_val(site, 'Zero', 'Zero', 'const')
    KN['eu_table'] = eu
    U = R[(R.group == 'eu_uniform') & (R['mode'] == 'const')]
    KN['eu_uniform_capex'] = {f'{r.site}|{r.scen}|{r.rate}': float(r.delivered) for r in U.itertuples() if r.rate in RATES}
    t = R[(R.group == 'eu_tank') & (R['mode'] == 'const')]
    KN['eu_tank_only'] = {r.site: float(r.delivered) for r in t.itertuples()}
    d = EU[(EU['mode'] == 'const') & (EU.scen == 'Commercial') & (EU.rate == MOD) & (EU.group == 'eu')].set_index('site')
    KN['eu_caps'] = {s: dict(Cpv=float(d.loc[s].Cpv), Cw=float(d.loc[s].Cw), Cel=float(d.loc[s].Cel), h2s_hours=float(d.loc[s].h2s_hours),
                             plant_gate=float(d.loc[s].plant_gate)) for s in d.index}
    # ports & modes
    c = AF[(AF['mode'] == 'const') & (AF.scen == 'Commercial') & (AF.rate == MOD)].set_index('site').loc[ON]
    KN['port_choice'] = c.port.value_counts().to_dict(); KN['inland_mode'] = c.inland_mode.value_counts().to_dict()
    KN['water_src'] = c.water_src.value_counts().to_dict()
    KN['allports_gain'] = dict(median=float((c.delivered - c.delivered_allports).median()), max=float((c.delivered - c.delivered_allports).max()),
                               n_switch=int((c.port_allports != c.port).sum()), switched_to=c.port_allports[c.port_allports != c.port].value_counts().to_dict())
    KN['seaonly_penalty'] = dict(median=float((c.delivered_seaonly - c.delivered).median()), max=float((c.delivered_seaonly - c.delivered).max()))
    KN['loss_frac_median'] = float(c.loss_frac.median())
    full = AF[(AF['mode'] == 'const') & (AF.scen == 'Full de-risking') & (AF.rate == MOD)].set_index('site').loc[ON]
    fx = AF[(AF['mode'] == 'flex') & (AF.rate == MOD)]
    KN['delivered_flexHB'] = {s: dict(median=float(fx[fx.scen == s].set_index('site').loc[ON].delivered_flexHB.median()),
                                      min=float(fx[fx.scen == s].set_index('site').loc[ON].delivered_flexHB.min()),
                                      peak_ratio=float(fx[fx.scen == s].set_index('site').loc[ON].hb_peak_ratio.median())) for s in SC}
    KN['rank_corr_comm_full'] = float(c.delivered.rank().corr(full.delivered.rank()))

# =========================================================================== Fig 4: cost breakdown
COMP = [('c_pv', 'Solar PV'), ('c_wind', 'Wind'), ('c_el', 'Electrolyser'), ('c_h2s', 'H$_2$ storage'), ('c_water', 'Water'),
        ('c_road', 'Road'), ('c_inland', 'Inland transport'), ('c_hb', 'NH$_3$ synthesis'), ('c_terminals', 'Terminals'),
        ('c_ship', 'Shipping'), ('c_crack', 'Cracking'), ('c_losses', 'Conversion & boil-off losses')]
COMP_COL = ['#eda100', '#1baf7a', '#2a78d6', '#86b6ef', '#6da7ec', '#a8a69e', '#898781', '#e87ba4', '#c3c2b7', '#6250d6', '#eb6834', '#e34948']

def fig_breakdown():
    c = AF[(AF['mode'] == 'const') & (AF.rate == MOD)]
    com = c[c.scen == 'Commercial'].set_index('site').loc[ON]
    best = com.delivered.idxmin(); worst = com.delivered.idxmax(); rep = KN['rep_hex']
    bars = []
    for h, tag in [(best, 'best'), (rep, 'median'), (worst, 'worst')]:
        for s in ['Commercial', 'Full de-risking']:
            r = c[(c.site == h) & (c.scen == s)].iloc[0]
            bars.append((f'{h.replace("hex_", "#")} {tag}\n{"Comm." if s == "Commercial" else "Full DR"}', r))
    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(bars) + 2)
    for i, (lab, r) in enumerate(bars):
        bottom = 0
        for (k, name), col in zip(COMP, COMP_COL):
            v = r[k]; ax.bar(i, v, bottom=bottom, color=col, ec=SURF, lw=1.2, width=0.7, label=name if i == 0 else None); bottom += v
        ax.text(i, bottom + 0.08, f'{bottom:.2f}', ha='center', fontsize=8, color=INK)
    # EU
    for j, site in enumerate(P.EU_SITES):
        e = EU[(EU.site == site) & (EU.scen == 'Commercial') & (EU.rate == MOD) & (EU['mode'] == 'const') & (EU.group == 'eu')].iloc[0]
        i = len(bars) + j
        parts = [(e.plant_gate, CAT[6], 'EU plant (PV+wind+el.+cavern)'), (e.c_water, '#6da7ec', None), (e.c_pipe, '#898781', 'EU pipeline 2,000 km' if j == 1 else None)]
        if j == 1: parts[0] = (e.plant_gate, CAT[6], None)
        bottom = 0
        for v, col, lab in parts:
            ax.bar(i, v, bottom=bottom, color=col, ec=SURF, lw=1.2, width=0.7, label=lab); bottom += v
        ax.text(i, bottom + 0.08, f'{bottom:.2f}', ha='center', fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels([b[0] for b in bars] + ['Germany\nComm.', 'Spain+pipe\nComm.'], fontsize=7.8)
    ax.set_ylabel('€/kg H$_2$ delivered (Rotterdam / German demand)'); ax.grid(axis='x', visible=False)
    ax.legend(ncol=1, fontsize=7.5, bbox_to_anchor=(1.01, 1), loc='upper left')
    ax.set_title('Cost build-up of delivered hydrogen, 2.15% base rate (constant-supply production, NH$_3$ chain, cracking)', loc='left')
    fig.tight_layout(); save(fig, 'fig04_cost_breakdown')
    r = com.loc[rep]
    KN['rep_breakdown_comm'] = {k: float(r[k]) for k, _ in COMP} | dict(delivered=float(r.delivered), plant_gate=float(r.plant_gate))
    rf = c[(c.site == rep) & (c.scen == 'Full de-risking')].iloc[0]
    KN['rep_breakdown_full'] = {k: float(rf[k]) for k, _ in COMP} | dict(delivered=float(rf.delivered), plant_gate=float(rf.plant_gate))
    chain_cols = ['c_inland', 'c_hb', 'c_terminals', 'c_ship', 'c_crack', 'c_losses']
    KN['chain_penalty'] = {s: float(c[c.scen == s].set_index('site').loc[ON][chain_cols].sum(1).median()) for s in SC}
    KN['best_hex'] = best; KN['worst_hex'] = worst

# =========================================================================== Fig 5: scenario matrix vs EU
def fig_matrix():
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw=dict(width_ratios=[1.35, 1]))
    ax = axs[0]
    xs = np.arange(len(RATES)); wdt = 0.18
    for j, s in enumerate(SC):
        for i, r in enumerate(RATES):
            v = af_tab('const', s, r, 'delivered').loc[lambda x: x.index.isin(ON)]
            xi = i + (j - 1.5) * wdt
            ax.plot([xi, xi], [v.min(), v.max()], color=SCEN_COL[s], lw=6, alpha=0.35, solid_capstyle='round')
            ax.plot(xi, v.median(), 'o', color=SCEN_COL[s], ms=7, mec='white', label=s if i == 0 else None)
    for site, ls in zip(P.EU_SITES, ['-', '--']):
        for s, lw in [('Commercial', 1.6), ('Offtake', 1.0)]:
            ys = [eu_val(site, s, r) for r in RATES]
            ax.plot(xs, ys, ls, color=EU_COL[site], lw=lw, label=f'{site.split(" (")[0]} green, {s.lower()}')
    ax.set_xticks(xs); ax.set_xticklabels(RATES); ax.set_xlabel('Base-rate environment (Schmidt et al. 2019 scenarios)')
    ax.set_ylabel('€/kg H$_2$ delivered'); ax.legend(fontsize=7.3, ncol=2, loc='upper left')
    ax.set_title('(a) African H$_2$ at Rotterdam vs European production\n(dot = median, bar = range of 23 onshore sites)', loc='left', fontsize=9)
    # heatmap of gap (median Africa - Germany commercial/offtake)
    ax = axs[1]
    G = np.zeros((len(SC), len(RATES))); Gb = np.zeros_like(G)
    for j, s in enumerate(SC):
        for i, r in enumerate(RATES):
            v = af_tab('const', s, r, 'delivered').loc[lambda x: x.index.isin(ON)]
            G[j, i] = v.median() - eu_val('Germany (north)', 'Commercial', r)
            Gb[j, i] = v.min() - eu_val('Germany (north)', 'Commercial', r)
    lim = np.abs(G).max()
    from matplotlib.colors import LinearSegmentedColormap
    div = LinearSegmentedColormap.from_list('div', ['#1c5cab', '#86b6ef', '#f0efec', '#ec835a', '#b23a10'])
    im = ax.imshow(G, cmap=div, vmin=-lim, vmax=lim, aspect='auto')
    for j in range(len(SC)):
        for i in range(len(RATES)):
            ax.text(i, j, f'{G[j, i]:+.2f}\n(best {Gb[j, i]:+.2f})', ha='center', va='center', fontsize=8,
                    color='white' if abs(G[j, i]) > 0.6 * lim else INK)
    ax.set_xticks(range(len(RATES))); ax.set_xticklabels(RATES, fontsize=8); ax.set_yticks(range(len(SC))); ax.set_yticklabels(SC)
    ax.grid(False); fig.colorbar(im, ax=ax, fraction=0.05).set_label('Africa − Germany (€/kg); <0 = Africa cheaper')
    ax.set_title('(b) Median gap to German green H$_2$\n(Germany: commercial finance)', loc='left', fontsize=9)
    fig.tight_layout(); save(fig, 'fig05_scenario_matrix')
    KN['gap_median_vs_DE_comm'] = {f'{s}|{r}': float(G[j, i]) for j, s in enumerate(SC) for i, r in enumerate(RATES)}
    KN['gap_best_vs_DE_comm'] = {f'{s}|{r}': float(Gb[j, i]) for j, s in enumerate(SC) for i, r in enumerate(RATES)}
    # counts competitive
    cnt = {}
    for s in SC:
        for r in RATES:
            v = af_tab('const', s, r, 'delivered').loc[lambda x: x.index.isin(ON)]
            cnt[f'{s}|{r}'] = dict(vs_DE_comm=int((v <= eu_val('Germany (north)', 'Commercial', r)).sum()),
                                   vs_DE_offt=int((v <= eu_val('Germany (north)', 'Offtake', r)).sum()),
                                   vs_ES_comm=int((v <= eu_val('Spain (+2,000 km pipe)', 'Commercial', r)).sum()),
                                   vs_ES_offt=int((v <= eu_val('Spain (+2,000 km pipe)', 'Offtake', r)).sum()))
    KN['n_competitive'] = cnt
    # de-risking value steps at median hex
    steps = {}
    for r in RATES:
        med = {s: float(af_tab('const', s, r, 'delivered').loc[lambda x: x.index.isin(ON)].median()) for s in SC}
        steps[r] = med
    KN['derisk_steps_median'] = steps
    # financing share
    z = af_tab('const', 'Zero', 'Zero', 'delivered'); fs = {}
    for s in SC:
        for r in RATES:
            v = af_tab('const', s, r, 'delivered')
            fs[f'{s}|{r}'] = float((1 - z / v).loc[ON].median())
    for site in P.EU_SITES:
        for r in RATES:
            fs[f'{site}|Commercial|{r}'] = 1 - eu_val(site, 'Zero', 'Zero') / eu_val(site, 'Commercial', r)
            fs[f'{site}|Offtake|{r}'] = 1 - eu_val(site, 'Zero', 'Zero') / eu_val(site, 'Offtake', r)
    KN['financing_share'] = fs
    zf = af_tab('flex', 'Zero', 'Zero', 'onsite')
    KN['financing_share_onsite_flex'] = {f'{s}|{r}': float((1 - zf / af_tab('flex', s, r, 'onsite')).loc[ON].median()) for s in SC for r in RATES}
    # rate sensitivity: change from flat->extreme
    rs = {}
    for s in SC:
        a = af_tab('const', s, RATES[0], 'delivered').loc[ON]; b = af_tab('const', s, RATES[2], 'delivered').loc[ON]
        rs[s] = dict(abs=float((b - a).median()), rel=float(((b - a) / a).median()))
    for site in P.EU_SITES:
        for s in ['Commercial', 'Offtake']:
            a = eu_val(site, s, RATES[0]); b = eu_val(site, s, RATES[2]); rs[f'{site}|{s}'] = dict(abs=b - a, rel=(b - a) / a)
    KN['rate_response'] = rs

# =========================================================================== Fig 6: fossil comparison
def fig_fossil():
    co2 = np.linspace(0, 200, 201)
    fig, axs = plt.subplots(1, 2, figsize=(12, 5.8))
    ax = axs[0]
    for gas, ls in [(25, ':'), (35, '-'), (50, '--')]:
        ax.plot(co2, C.grey_h2(co2, gas), ls, color=INK2, lw=1.6 if gas == 35 else 1.0, label=f'Grey SMR, gas {gas} €/MWh')
    ax.plot(co2, C.blue_h2(co2, 35), '-', color=CAT[6], lw=1.6, label='Blue SMR+CCS (90%), gas 35 €/MWh')
    bands = [('flex', 'Commercial', 'onsite', 'On-site, flexible, commercial', CAT[3]),
             ('flex', 'Full de-risking', 'onsite', 'On-site, flexible, full de-risking', CAT[2]),
             ('const', 'Commercial', 'delivered', 'Delivered Rotterdam, commercial', CAT[1]),
             ('const', 'Full de-risking', 'delivered', 'Delivered Rotterdam, full de-risking', CAT[0])]
    be = {}
    for mode, s, col, lab, cc in bands:
        v = af_tab(mode, s, MOD, col).loc[ON]
        ax.axhspan(v.min(), v.max(), color=cc, alpha=0.18, lw=0)
        ax.axhline(v.median(), color=cc, lw=1.4, label=f'{lab} (median {v.median():.2f})')
        g0 = C.grey_h2(0); slope = P.SMR['co2'] / 1e3
        be[f'{mode}|{s}|{col}'] = dict(median=float((v.median() - g0) / slope), min=float((v.min() - g0) / slope))
    ax.set_xlabel('CO$_2$ price (€/t)'); ax.set_ylabel('€/kg H$_2$'); ax.set_xlim(0, 200)
    ax.legend(fontsize=7, loc='upper center', bbox_to_anchor=(0.5, -0.14), ncol=2); ax.set_title('(a) African green H$_2$ (2.15% base rate) vs fossil H$_2$ in Europe', loc='left', fontsize=9)
    # NH3 panel: delivered NH3 vs Mueller conventional NH3
    ax = axs[1]
    ax.plot(co2, C.grey_nh3(co2) * 1e3, color=INK2, lw=1.6, label='Conventional NH$_3$ 2030 (Müller et al. 2023, Table 5)')
    ax.plot(list(P.MUL_NH3), [v * 1e3 for v in P.MUL_NH3.values()], 'o', color=INK2, ms=5)
    nb = {}
    for s, cc in [('Commercial', CAT[1]), ('Offtake', CAT[3]), ('Blended', CAT[2]), ('Full de-risking', CAT[0])]:
        v = af_tab('const', s, MOD, 'nh3_rotterdam_eur_t').loc[ON]
        ax.axhspan(v.min(), v.max(), color=cc, alpha=0.15, lw=0); ax.axhline(v.median(), color=cc, lw=1.4, label=f'Green NH$_3$ at Rotterdam, {s.lower()} ({v.median():.0f} €/t)')
        nb[s] = dict(median=float((v.median() / 1e3 - 0.40) / 0.00165), min=float((v.min() / 1e3 - 0.40) / 0.00165), nh3_med=float(v.median()))
    for site, ls in zip(P.EU_SITES, ['-', '--']):
        if site.startswith('Germany'):
            vv = eu_val(site, 'Commercial', MOD, col='nh3_eu_eur_t'); vo = eu_val(site, 'Offtake', MOD, col='nh3_eu_eur_t')
            ax.axhline(vv, color=EU_COL[site], lw=1.4, ls='--', label=f'Green NH$_3$ made in Germany, commercial ({vv:.0f} €/t)')
            KN['nh3_DE'] = dict(comm=float(vv), offt=float(vo), comm_flat=eu_val(site, 'Commercial', RATES[0], col='nh3_eu_eur_t'),
                                comm_ext=eu_val(site, 'Commercial', RATES[2], col='nh3_eu_eur_t'))
    nh3c = {}
    for s in SC:
        for r in RATES:
            v = af_tab('const', s, r, 'nh3_rotterdam_eur_t').loc[ON]
            nh3c[f'{s}|{r}'] = dict(median=float(v.median()), min=float(v.min()), n_below_DE=int((v <= eu_val('Germany (north)', 'Commercial', r, col='nh3_eu_eur_t')).sum()))
    KN['nh3_import_table'] = nh3c
    ax.set_xlabel('CO$_2$ price (€/t)'); ax.set_ylabel('€/t NH$_3$'); ax.set_xlim(0, 200)
    ax.legend(fontsize=7, loc='upper center', bbox_to_anchor=(0.5, -0.14), ncol=2); ax.set_title('(b) Ammonia as final product (no cracking)', loc='left', fontsize=9)
    fig.tight_layout(); save(fig, 'fig06_fossil_benchmarks')
    KN['breakeven_co2_vs_grey'] = be; KN['breakeven_co2_nh3'] = nb
    KN['grey_0'] = float(C.grey_h2(0)); KN['grey_100'] = float(C.grey_h2(100)); KN['blue_0'] = float(C.blue_h2(0)); KN['blue_100'] = float(C.blue_h2(100))
    KN['grey_blue_cross'] = float((C.blue_h2(0) - C.grey_h2(0)) / ((P.SMR['co2'] - P.BLUE['co2']) / 1e3))
    # Mueller break-even check: LCOH that makes green NH3 equal conventional, our chain at Lüderitz-port basis
    be_tab = {}
    for s in SC:
        v = af_tab('const', s, MOD, 'nh3_port_eur_t').loc[ON]
        be_tab[s] = float(v.median())
    KN['nh3_port_median'] = be_tab
    # EU green vs grey
    KN['breakeven_co2_EU'] = {f'{site}|{s}': float((eu_val(site, s, MOD) - C.grey_h2(0)) / (P.SMR['co2'] / 1e3)) for site in P.EU_SITES for s in ['Commercial', 'Offtake']}

# =========================================================================== Fig 7: shipping distance sweep
def fig_shipping():
    rep = KN['rep_hex']; best = KN['best_hex']
    import assemble as A
    L = pd.read_csv(f'{OUT}/lp_main.csv', keep_default_na=False, na_values=['']); L['country'] = 'NA'
    nm = np.arange(1500, 7001, 250)
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.4))
    ax = axs[0]; out = {}
    for s in ['Commercial', 'Full de-risking']:
        for h, ls in [(best, '-'), (rep, '--')]:
            row = L[(L.site == h) & (L.scen == s) & (L.rate == MOD) & (L['mode'] == 'const') & (L.group == 'af')].iloc[0].to_dict()
            ys = [A.delivered(row, nmi_override=n)['delivered'] for n in nm]
            ax.plot(nm, ys, ls, color=SCEN_COL[s], label=f'{"best" if h == best else "median"} site {h.replace("hex_", "#")}, {s.lower()}')
            out[f'{h}|{s}'] = dict(slope_per_1000nmi=float((ys[-1] - ys[0]) / ((nm[-1] - nm[0]) / 1000)), at_5450=float(np.interp(5450, nm, ys)))
    for site, ls in zip(P.EU_SITES, ['-', '--']):
        ax.axhline(eu_val(site, 'Commercial', MOD), ls=ls, color=EU_COL[site], lw=1.2, label=f'{site.split(" (")[0]} green, commercial')
    for n, lab in [(1700, 'Morocco–Rotterdam\n(stylised)'), (4980, 'Namibe'), (5450, 'Walvis Bay'), (6080, 'Saldanha')]:
        ax.axvline(n, color=GRID, lw=1); ax.text(n + 40, ax.get_ylim()[1] if False else 0, '', fontsize=7)
    ax.set_xlabel('Shipping distance to Rotterdam (nautical miles)'); ax.set_ylabel('€/kg H$_2$ delivered')
    ax.legend(fontsize=7, loc='upper left'); ax.set_title('(a) Delivered cost vs shipping distance (inland leg & production fixed)', loc='left', fontsize=9)
    yl = ax.get_ylim()
    for n, lab in [(1700, 'N-Africa\n(stylised)'), (4980, 'Namibe'), (5450, 'Walvis'), (6080, 'Saldanha')]:
        ax.text(n + 30, yl[0] + 0.02 * (yl[1] - yl[0]), lab, fontsize=6.8, color=MUTED, rotation=90, va='bottom')
    # (b) required production LCOH (plant-gate) to match Germany vs distance
    ax = axs[1]
    for s in ['Commercial', 'Full de-risking']:
        row = L[(L.site == rep) & (L.scen == s) & (L.rate == MOD) & (L['mode'] == 'const') & (L.group == 'af')].iloc[0].to_dict()
        target = eu_val('Germany (north)', 'Commercial', MOD)
        req = []
        for n in nm:
            d = A.delivered(row, nmi_override=n)
            nonprod = d['delivered'] - d['plant_gate'] * (1 - 0) * (P.H2_DEMAND_KG / (P.H2_DEMAND_KG * (1 - d['loss_frac'])))
            req.append((target - nonprod) * (1 - d['loss_frac']))
        ax.plot(nm, req, color=SCEN_COL[s], label=f'max plant-gate LCOH to match Germany, {s.lower()} chain')
        out[f'req|{s}'] = dict(at_1700=float(np.interp(1700, nm, req)), at_5450=float(np.interp(5450, nm, req)))
    v = af_tab('const', 'Commercial', MOD, 'plant_gate').loc[ON]; v2 = af_tab('const', 'Full de-risking', MOD, 'plant_gate').loc[ON]
    ax.axhspan(v.min(), v.max(), color=SCEN_COL['Commercial'], alpha=0.15, lw=0, label='modelled plant-gate range, commercial')
    ax.axhspan(v2.min(), v2.max(), color=SCEN_COL['Full de-risking'], alpha=0.15, lw=0, label='modelled plant-gate range, full de-risking')
    ax.set_xlabel('Shipping distance (nmi)'); ax.set_ylabel('€/kg H$_2$ at plant gate (constant supply)')
    ax.legend(fontsize=7, loc='lower left'); ax.set_title('(b) Production cost needed to beat German green H$_2$', loc='left', fontsize=9)
    fig.tight_layout(); save(fig, 'fig07_shipping_distance')
    KN['ship_sweep'] = out

# =========================================================================== Fig 8: infrastructure drivers
def std_reg(y, X):
    Xs = (X - X.mean()) / X.std(); ys = (y - y.mean()) / y.std()
    A_ = np.c_[np.ones(len(Xs)), Xs.values]
    b, *_ = np.linalg.lstsq(A_, ys.values, rcond=None)
    pred = A_ @ b; r2 = 1 - ((ys.values - pred) ** 2).sum() / ((ys.values - ys.values.mean()) ** 2).sum()
    # VIF
    vif = {}
    for i, c in enumerate(X.columns):
        o = Xs.drop(columns=c); A2 = np.c_[np.ones(len(o)), o.values]
        bb, *_ = np.linalg.lstsq(A2, Xs[c].values, rcond=None); rr = 1 - ((Xs[c].values - A2 @ bb) ** 2).sum() / ((Xs[c].values - Xs[c].mean()) ** 2).sum()
        vif[c] = 1 / max(1e-9, 1 - rr)
    return dict(zip(X.columns, b[1:])), r2, vif

def fig_infra():
    c = AF[(AF['mode'] == 'const') & (AF.scen == 'Commercial') & (AF.rate == MOD)].set_index('site')
    f = AF[(AF['mode'] == 'flex') & (AF.scen == 'Commercial') & (AF.rate == MOD)].set_index('site')
    G = R[(R.group == 'grid')]
    fig, axs = plt.subplots(2, 3, figsize=(13, 7.6)); axs = axs.ravel()
    on = c.index.isin(ON)
    # (a) grid
    ax = axs[0]; gr = {}
    for s, mk in [('Commercial', 'o'), ('Full de-risking', 's')]:
        g = G[G.scen == s].set_index('site'); base = AF[(AF['mode'] == 'const') & (AF.scen == s) & (AF.rate == MOD)].set_index('site')
        if len(g) == 0: continue
        fixed = (P.GRID_LINE_FIXED * g.grid_dist_km * P.DETOUR + P.GRID_SUBSTATION) * (g.w_infra.apply(lambda w: P.crf(w, P.INFRA_LIFE)) + 0.015) / P.H2_DEMAND_KG
        gtied = g.onsite + fixed
        sav = base.loc[g.index].onsite - gtied
        ax.scatter(g.grid_dist_km, sav, marker=mk, color=SCEN_COL[s], s=40, ec='white', label=s)
        gr[s] = dict(median_saving=float(sav.loc[sav.index.isin(ON)].median()), n_beneficial=int((sav.loc[sav.index.isin(ON)] > 0).sum()),
                     max_saving=float(sav.max()), corr_dist=float(np.corrcoef(g.grid_dist_km, sav)[0, 1]),
                     breakeven_km=float(np.polyval(np.polyfit(sav, g.grid_dist_km, 1), 0)) if sav.std() > 0 else np.nan)
        z = np.polyfit(g.grid_dist_km, sav, 1); xx = np.linspace(0, 250, 10); ax.plot(xx, np.polyval(z, xx), color=SCEN_COL[s], lw=1)
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_xlabel('distance to grid (km)'); ax.set_ylabel('saving of grid tie vs off-grid (€/kg)'); ax.legend(fontsize=7)
    ax.set_title('(a) Grid connection (constant supply)', loc='left', fontsize=9); KN['grid'] = gr
    # (b) road
    ax = axs[1]
    ax.scatter(c.road_dist_km, c.c_road, color=CAT[0], s=35, ec='white'); ax.set_xlabel('distance to road (km)'); ax.set_ylabel('road cost (€/kg)')
    ax.set_title('(b) Road access', loc='left', fontsize=9)
    KN['road'] = dict(max=float(c.c_road.max()), median=float(c.c_road.median()))
    # (c) water
    ax = axs[2]
    for src, cc in [('freshwater', CAT[0]), ('seawater', CAT[2])]:
        m = c.water_src == src
        ax.scatter(np.minimum(c.waterbody_dist_km, c.ocean_dist_km)[m], c.c_water[m], color=cc, s=35, ec='white', label=f'{src} ({m.sum()})')
    ax.set_xlabel('distance to chosen water source (km)'); ax.set_ylabel('water cost (€/kg)'); ax.legend(fontsize=7)
    ax.set_title('(c) Water supply (incl. pipeline & treatment capex)', loc='left', fontsize=9)
    KN['water'] = dict(max=float(c.c_water.max()), median=float(c.c_water.median()), share_median=float((c.c_water / c.delivered).median()))
    # (d) inland transport / port distance
    ax = axs[3]
    ax.scatter(c.inland_km, c.c_inland, color=CAT[6], s=35, ec='white')
    ax.set_xlabel('route length to chosen port (km)'); ax.set_ylabel('inland transport (€/kg)'); ax.set_title('(d) Inland leg to port (H$_2$ pipeline or NH$_3$ truck)', loc='left', fontsize=9)
    KN['inland'] = dict(max=float(c.c_inland.max()), median=float(c.c_inland.median()), km_range=[float(c.inland_km.min()), float(c.inland_km.max())])
    # (e) standardized regression
    ax = axs[4]
    X = pd.DataFrame(dict(cf_pv=c.cf_pv, cf_wind=c.cf_wind, grid_km=c.grid_dist_km, road_km=c.road_dist_km,
                          water_km=np.minimum(c.waterbody_dist_km, c.ocean_dist_km), port_km=c.inland_km))
    regs = {}
    for k, (y, lab) in enumerate([(f.onsite, 'on-site LCOH (flexible)'), (c.delivered, 'delivered cost')]):
        bb, r2, vif = std_reg(y.loc[ON], X.loc[ON]); regs[lab] = dict(beta=bb, r2=r2, vif=vif)
        ax.barh(np.arange(len(bb)) + (k - 0.5) * 0.38, list(bb.values()), height=0.36, color=[CAT[3], CAT[1]][k], label=f'{lab} (R²={r2:.2f})')
    ax.set_yticks(range(len(X.columns))); ax.set_yticklabels(X.columns); ax.axvline(0, color=INK2, lw=0.8)
    ax.set_xlabel('standardised coefficient'); ax.legend(fontsize=7); ax.set_title('(e) Driver attribution (23 onshore, OLS)', loc='left', fontsize=9)
    KN['regression'] = {k: dict(beta={a: float(b) for a, b in v['beta'].items()}, r2=float(v['r2']), vif_max=float(max(v['vif'].values()))) for k, v in regs.items()}
    # (f) cost share of infrastructure terms
    ax = axs[5]
    parts = ['c_water', 'c_road', 'c_inland']
    order = c.loc[ON].sort_values('delivered').index
    bottom = np.zeros(len(order))
    for p_, lab, cc in zip(parts, ['water', 'road', 'inland transport'], [CAT[2], '#a8a69e', CAT[6]]):
        ax.bar(range(len(order)), c.loc[order, p_], bottom=bottom, color=cc, ec=SURF, lw=0.8, label=lab); bottom += c.loc[order, p_].values
    ax.set_xticks(range(len(order))); ax.set_xticklabels([o.replace('hex_', '') for o in order], rotation=90, fontsize=6.5)
    ax.set_ylabel('€/kg'); ax.legend(fontsize=7); ax.grid(axis='x', visible=False)
    ax.set_title('(f) Site-infrastructure costs, sites sorted by delivered cost', loc='left', fontsize=9)
    KN['infra_total'] = dict(median=float(c.loc[ON][parts].sum(1).median()), max=float(c.loc[ON][parts].sum(1).max()),
                             share_median=float((c.loc[ON][parts].sum(1) / c.loc[ON].delivered).median()))
    fig.tight_layout(); save(fig, 'fig08_infrastructure_drivers')

# =========================================================================== Fig 9: tornado (+chain params analytic)
def fig_tornado():
    S = R[R.group == 'sens']; rep = KN['rep_hex']
    if len(S) == 0: return
    base_c = sel(group='af', site=rep, mode='const', scen='Commercial', rate=MOD).iloc[0]
    base_f = sel(group='af', site=rep, mode='flex', scen='Commercial', rate=MOD).iloc[0]
    import assemble as A
    L = pd.read_csv(f'{OUT}/lp_main.csv', keep_default_na=False, na_values=['']); L['country'] = 'NA'
    row = L[(L.site == rep) & (L.scen == 'Commercial') & (L.rate == MOD) & (L['mode'] == 'const') & (L.group == 'af')].iloc[0].to_dict()
    pairs = {}
    for nm_ in sorted(set(S.sens)):
        k = nm_.rsplit(' ', 1)[0] if nm_[-1] == '%' else nm_
        pairs.setdefault(k.replace('Efficiency', 'Electrolyser efficiency'), []).append(nm_)
    rows = []
    for k, names in pairs.items():
        dc = [float(S[(S.sens == n) & (S['mode'] == 'const')].delivered.iloc[0]) - base_c.delivered for n in names if len(S[(S.sens == n) & (S['mode'] == 'const')])]
        rows.append((k, min(dc + [0]), max(dc + [0]), names))
    # analytic chain parameters (re-evaluated with LP design fixed)
    def chain_var(attr, mult):
        old = getattr(P, attr); setattr(P, attr, old * mult)
        try: v = A.delivered(row)['delivered']
        finally: setattr(P, attr, old)
        return v - base_c.delivered
    for attr, lab, m in [('HB_CAPEX', 'NH$_3$ synthesis capex ±30%', 0.3), ('CRACK_CAPEX', 'Cracker capex ±50%', 0.5),
                         ('WATER_PIPE_CAPEX', 'Water pipeline capex ±50%', 0.5), ('H2PIPE_CAPEX', 'H$_2$ pipeline capex ±50%', 0.5),
                         ('TERM_CAPEX', 'Terminal capex ±50%', 0.5)]:
        a, b = chain_var(attr, 1 - m), chain_var(attr, 1 + m); rows.append((lab, min(a, b), max(a, b), []))
    old = P.CRACK_YIELD
    P.CRACK_YIELD = 0.80; a = A.delivered(row)['delivered'] - base_c.delivered
    P.CRACK_YIELD = 0.90; b = A.delivered(row)['delivered'] - base_c.delivered; P.CRACK_YIELD = old
    rows.append(('Cracking yield 0.80/0.90', min(a, b), max(a, b), []))
    # finance rows from main results
    v = {s: float(sel(group='af', site=rep, mode='const', scen=s, rate=MOD).delivered.iloc[0]) for s in SC}
    rows.append(('Finance: full de-risking', v['Full de-risking'] - base_c.delivered, 0, []))
    vr = [float(sel(group='af', site=rep, mode='const', scen='Commercial', rate=r).delivered.iloc[0]) - base_c.delivered for r in RATES]
    rows.append(('Base rate 0.49% / 4.29%', min(vr), max(vr), []))
    rows.sort(key=lambda x: x[2] - x[1])
    fig, ax = plt.subplots(figsize=(9, 6))
    for i, (k, lo, hi, _) in enumerate(rows):
        ax.barh(i, lo, color=CAT[0], height=0.6); ax.barh(i, hi, color=CAT[1], height=0.6)
        ax.text(min(lo, 0) - 0.03, i, f'{lo:+.2f}', ha='right', va='center', fontsize=7.5, color=INK2)
        ax.text(max(hi, 0) + 0.03, i, f'{hi:+.2f}', ha='left', va='center', fontsize=7.5, color=INK2)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows], fontsize=8); ax.axvline(0, color=INK, lw=0.8)
    ax.set_xlabel(f'Δ delivered cost (€/kg) around {base_c.delivered:.2f} €/kg (site {rep.replace("hex_", "#")}, commercial, 2.15%)')
    lim = max(abs(r[1]) for r in rows + [(0, 0, 0, 0)]) + 0.5; ax.set_xlim(-max(abs(min(r[1] for r in rows)), 0.3) - 0.45, max(r[2] for r in rows) + 0.45)
    ax.grid(axis='y', visible=False); ax.set_title('Sensitivity of delivered cost (LP re-optimised for plant parameters)', loc='left')
    fig.tight_layout(); save(fig, 'fig09_tornado')
    KN['tornado'] = {r[0]: [float(r[1]), float(r[2])] for r in rows}
    # battery check
    b = S[S.sens.str.startswith('Battery') & (S['mode'] == 'const')]
    KN['battery_check'] = {r.sens: dict(Eb=float(r.Eb), delta=float(r.delivered - base_c.delivered)) for r in b.itertuples()}
    KN['cf_sens_flex'] = {r.sens: float(r.onsite - base_f.onsite) for r in S[S['mode'] == 'flex'].itertuples()}

# =========================================================================== Fig 10: CRP break-even + rate sweep
def fig_crp_rates():
    Cr = R[R.group == 'crp']
    if len(Cr) == 0: return
    rep = KN['rep_hex']
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.4))
    ax = axs[0]; out = {}
    de_c = eu_val('Germany (north)', 'Commercial', MOD); de_o = eu_val('Germany (north)', 'Offtake', MOD); es_c = eu_val('Spain (+2,000 km pipe)', 'Commercial', MOD)
    for h, ls in [(KN['best_hex'], '-'), (rep, '--')]:
        d = Cr[(Cr.site == h) & (Cr['mode'] == 'const')].sort_values('crp')
        if len(d) == 0: continue
        ax.plot(d.crp * 100, d.delivered, ls, marker='o', color=CAT[1], label=f'{"best" if h == KN["best_hex"] else "median"} site {h.replace("hex_", "#")} (commercial structure)')
        for tgt, nm_ in [(de_c, 'DE_comm'), (de_o, 'DE_offt'), (es_c, 'ES_comm')]:
            x = d.crp.values * 100; y = d.delivered.values
            out[f'{h}|{nm_}'] = float(np.interp(tgt, y, x)) if y.min() <= tgt <= y.max() else (float('nan') if y.min() > tgt else float('inf'))
        out[f'{h}|slope_per_pp'] = float(np.polyfit(d.crp * 100, d.delivered, 1)[0])
    for v, lab, cc, ls in [(de_c, 'Germany, commercial', EU_COL['Germany (north)'], '-'), (de_o, 'Germany, offtake', EU_COL['Germany (north)'], ':'),
                           (es_c, 'Spain + pipeline, commercial', EU_COL['Spain (+2,000 km pipe)'], '--')]:
        ax.axhline(v, color=cc, ls=ls, lw=1.2, label=lab)
    ax.axvline(P.COUNTRY['NA']['crp'] * 100, color=MUTED, lw=1); ax.text(P.COUNTRY['NA']['crp'] * 100 + 0.1, ax.get_ylim()[0] + 0.1, 'Namibia\n(assumed)', fontsize=7, color=MUTED)
    ax.set_xlabel('Country risk premium on equity (%-points; debt spread = 0.7 × CRP)'); ax.set_ylabel('€/kg H$_2$ delivered')
    ax.legend(fontsize=7); ax.set_title('(a) Country-risk break-even (2.15% base rate)', loc='left', fontsize=9)
    KN['crp_breakeven'] = out
    # (b) base-rate sweep
    ax = axs[1]; rs = {}
    W = R[(R.group.isin(['ratesweep', 'af'])) & (R.site == rep) & (R['mode'] == 'const') & (R.scen != 'Zero')]
    for s in ['Commercial', 'Offtake', 'Full de-risking']:
        d = W[W.scen == s].drop_duplicates('ir').sort_values('ir')
        ax.plot(d.ir * 100, d.delivered, marker='o', ms=4, color=SCEN_COL[s], label=f'Africa median site, {s.lower()}')
        rs[f'AF|{s}'] = dict(slope=float(np.polyfit(d.ir * 100, d.delivered, 1)[0]), at0=float(d.delivered.iloc[0]), at6=float(d.delivered.iloc[-1]))
    for site in P.EU_SITES:
        for s, ls in [('Commercial', '-'), ('Offtake', ':')]:
            d = EU[(EU.site == site) & (EU.scen == s) & (EU['mode'] == 'const') & (EU.group == 'eu')].sort_values('ir')
            ax.plot(d.ir * 100, d.delivered, ls, color=EU_COL[site], lw=1.4, label=f'{site.split(" (")[0]}, {s.lower()}')
            rs[f'{site}|{s}'] = dict(slope=float(np.polyfit(d.ir * 100, d.delivered, 1)[0]), at0=float(d.delivered.iloc[0]), at6=float(d.delivered.iloc[-1]))
    for r in P.BASE_RATES.values(): ax.axvline(r * 100, color=GRID, lw=1)
    ax.set_xlabel('Base (risk-free) interest rate (%)'); ax.set_ylabel('€/kg H$_2$ delivered'); ax.legend(fontsize=6.8, ncol=2)
    ax.set_title('(b) Interest-rate environment', loc='left', fontsize=9)
    fig.tight_layout(); save(fig, 'fig10_crp_and_rates')
    KN['rate_sweep'] = rs

# =========================================================================== Fig 11: validation
def fig_validation():
    fig, axs = plt.subplots(2, 3, figsize=(13, 8.4)); axs = axs.ravel()
    # (a) WACC markups vs Steffen
    ax = axs[0]; L17 = -0.0018
    model = {'Germany PV': P.wacc('pv', 'EU', 'Commercial', L17) - L17, 'Germany wind': P.wacc('wind', 'EU', 'Commercial', L17) - L17,
             'Namibia PV': P.wacc('pv', 'NA', 'Commercial', L17) - L17, 'Namibia wind': P.wacc('wind', 'NA', 'Commercial', L17) - L17,
             'South Africa PV': P.wacc('pv', 'ZA', 'Commercial', L17) - L17}
    lit = {'Germany PV': 0.029, 'Germany wind': 0.031, 'Namibia PV': 0.056, 'Namibia wind': 0.102, 'South Africa PV': 0.048}
    litlab = {'Namibia PV': 'non-OECD avg', 'Namibia wind': 'non-OECD avg'}
    x = np.arange(len(model))
    ax.bar(x - 0.2, [v * 100 for v in model.values()], 0.38, color=CAT[0], label='this model'); ax.bar(x + 0.2, [lit[k] * 100 for k in model], 0.38, color=MUTED, label='Steffen (2020) Table 2')
    for i, k in enumerate(model):
        if k in litlab: ax.text(i + 0.2, lit[k] * 100 + 0.2, litlab[k], ha='center', fontsize=6, rotation=90, va='bottom')
    ax.set_xticks(x); ax.set_xticklabels(list(model), rotation=25, fontsize=7.5); ax.set_ylabel('WACC markup over 2017 LIBOR (%-pt)')
    ax.legend(fontsize=7); ax.set_title('(a) Cost-of-capital markups', loc='left', fontsize=9); ax.grid(axis='x', visible=False)
    KN['val_markups'] = {k: float(v) for k, v in model.items()}
    # (b) Schmidt LCOE response (Germany, 25y, flat CF)
    ax = axs[1]; sch = {}
    for tech, capex, fom, cf, lit_inc, cc in [('PV', 450e3, 0.015, 0.11, 0.11, CAT[3]), ('Wind', 1150e3, 0.025, 0.30, 0.25, CAT[2])]:
        a = 'pv' if tech == 'PV' else 'wind'
        irs = np.linspace(0, 0.05, 11)
        l = [capex * (P.crf(P.wacc(a, 'EU', 'Commercial', ir), 25) + fom) / (cf * 8760) for ir in irs]
        ax.plot(irs * 100, l, color=cc, label=tech)
        l0 = capex * (P.crf(P.wacc(a, 'EU', 'Commercial', 0.0049), 25) + fom); l1 = capex * (P.crf(P.wacc(a, 'EU', 'Commercial', 0.0429), 25) + fom)
        sch[tech] = dict(model=float(l1 / l0 - 1), schmidt=lit_inc)
    ax.set_xlabel('base rate (%)'); ax.set_ylabel('LCOE Germany (€/MWh)'); ax.legend(fontsize=7)
    ax.set_title('(b) LCOE response to base rate', loc='left', fontsize=9)
    ax.text(0.35, 0.05, '0.49%→4.29% increase:\n' + '\n'.join(f'{k}: model +{100 * v["model"]:.0f}%  (Schmidt +{100 * v["schmidt"]:.0f}%)' for k, v in sch.items()),
            transform=ax.transAxes, fontsize=7.5, color=INK2)
    KN['val_schmidt'] = sch
    # (c) GeoH2 replication
    ax = axs[2]; g = R[(R.group == 'geoh2') & (R['mode'] == 'const')]
    if len(g):
        gc = g.set_index('site')
        lud = gc.plant_gate + gc.c_water + gc.c_road + gc.c_inland   # cost of H2 arriving at port (GeoH2 metric)
        gf = R[(R.group == 'geoh2') & (R['mode'] == 'flex')].set_index('site').onsite
        ax.hist(lud.loc[lud.index.isin(ON)], bins=10, color=CAT[0], ec=SURF, lw=1.5, alpha=0.9, label='this model, const. supply, GeoH2 inputs, 6%')
        ax.hist(gf.loc[gf.index.isin(ON)], bins=10, color=CAT[3], ec=SURF, lw=1.5, alpha=0.8, label='same, flexible')
        ax.axvspan(5.43, 9.21, color=MUTED, alpha=0.15, label='GeoH2 Namibia range (5.43–9.21)')
        ax.set_xlabel('€/kg H$_2$ at port'); ax.set_ylabel('hexagons'); ax.legend(fontsize=6.8, loc='upper center', bbox_to_anchor=(0.5, -0.17))
        KN['val_geoh2'] = dict(const_min=float(lud.loc[ON].min()), const_med=float(lud.loc[ON].median()), const_max=float(lud.loc[ON].max()),
                               flex_min=float(gf.loc[ON].min()), flex_med=float(gf.loc[ON].median()))
    ax.set_title('(c) Replication with GeoH2 cost inputs', loc='left', fontsize=9)
    # (d) temporal resolution
    ax = axs[3]; T = R[R.group == 'tres']
    if len(T):
        tr = {}
        labs = []; vals1 = []; vals3 = []
        for (site, mode), d in T.groupby(['site', 'mode']):
            if set(d.dt) >= {1, 3}:
                v1 = float(d[d.dt == 1].plant_gate.iloc[0]); v3 = float(d[d.dt == 3].plant_gate.iloc[0])
                tr[f'{site}|{mode}'] = dict(h1=v1, h3=v3, diff=v3 / v1 - 1); labs.append(f'{site.split(" (")[0].replace("hex_", "#")}\n{mode}'); vals1.append(v1); vals3.append(v3)
        x = np.arange(len(labs))
        ax.bar(x - 0.2, vals1, 0.38, color=MUTED, label='hourly'); ax.bar(x + 0.2, vals3, 0.38, color=CAT[0], label='3-hourly (main runs)')
        for i in range(len(labs)): ax.text(i + 0.2, vals3[i] + 0.05, f'{100 * (vals3[i] / vals1[i] - 1):+.1f}%', ha='center', fontsize=7.5)
        ax.set_xticks(x); ax.set_xticklabels(labs, fontsize=7.5); ax.set_ylabel('plant-gate LCOH (€/kg)'); ax.legend(fontsize=7, loc='lower center'); ax.set_ylim(0, 4.6)
        KN['val_tres'] = tr
    ax.set_title('(d) Temporal resolution check', loc='left', fontsize=9); ax.grid(axis='x', visible=False)
    # (e) literature comparison of production LCOH
    ax = axs[4]
    lit = [('IEA 2030 Namibia (low end)', 2.5 * 0.92, 2.5 * 0.92), ('Fraunhofer ISE 2030', 1.13, 1.13), ('NIPDB claim', 1.5 * 0.92, 1.5 * 0.92),
           ('Jülich (min–avg)', 4.17, 5.04), ('GeoH2 Namibia (const., today)', 5.43, 9.21), ('Müller Kenya (today, 8%)', 3.7, 9.9)]
    st = KN['onsite_table']
    ours = [('This: flexible, full DR', st[f'flex|Full de-risking|{MOD}']), ('This: flexible, commercial', st[f'flex|Commercial|{MOD}']),
            ('This: constant, full DR', st[f'const|Full de-risking|{MOD}']), ('This: constant, commercial', st[f'const|Commercial|{MOD}'])]
    yy = 0; yl = []
    for lab, lo, hi in lit:
        ax.plot([lo, hi], [yy, yy], color=MUTED, lw=6, solid_capstyle='round')
        if hi - lo < 1e-6: ax.plot(lo, yy, 'D', color=MUTED, ms=7)
        yl.append(lab); yy += 1
    for lab, s_ in ours:
        ax.plot([s_['min'], s_['max']], [yy, yy], color=CAT[0], lw=6, solid_capstyle='round'); ax.plot(s_['median'], yy, 'o', color='white', ms=4); yl.append(lab); yy += 1
    ax.set_yticks(range(yy)); ax.set_yticklabels(yl, fontsize=7.2); ax.set_xlabel('production LCOH (€/kg)')
    ax.set_title('(e) Production LCOH vs literature', loc='left', fontsize=9); ax.grid(axis='y', visible=False)
    # (f) synthetic weather check
    ax = axs[5]
    pv = M.pv_profile(-22.0, 0.22); wd = M.wind_profile(0.30); pvg = M.pv_profile(53.5, 0.11, 'eu'); wdg = M.wind_profile(0.30, 'eu')
    for prof, lab, cc, ls in [(pv, 'PV Namibia', CAT[3], '-'), (wd, 'wind Namibia', CAT[2], '-'), (pvg, 'PV Germany', CAT[3], ':'), (wdg, 'wind Germany', CAT[2], ':')]:
        m = prof[:8760 - 8760 % 168].reshape(-1, 168).mean(1)
        ax.plot(np.arange(len(m)), m, ls, color=cc, lw=1.4, label=f'{lab} (annual {prof.mean():.2f})')
    ax.set_xlabel('week of year'); ax.set_ylabel('weekly mean CF'); ax.legend(fontsize=6.8, ncol=2, loc='upper center', bbox_to_anchor=(0.5, -0.17)); ax.set_ylim(0, 0.65)
    ax.set_title('(f) Synthetic weather: weekly capacity factors', loc='left', fontsize=9)
    fig.tight_layout(); save(fig, 'fig11_validation')

# =========================================================================== Fig 12: country-assignment sensitivity & offshore
def fig_country():
    Cn = R[R.group == 'country']
    if len(Cn) == 0: return
    c = Cn[Cn['mode'] == 'const'].set_index('site'); base = af_tab('const', 'Commercial', MOD, 'delivered')
    f = Cn[Cn['mode'] == 'flex'].set_index('site'); basef = af_tab('flex', 'Commercial', MOD, 'onsite')
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(c)); lab = [f'{h.replace("hex_", "#")} ({c.loc[h].country})' for h in c.index]
    ax.bar(x - 0.2, base.loc[c.index], 0.38, color=CAT[0], label='Namibian financing (primary framing)')
    ax.bar(x + 0.2, c.delivered, 0.38, color=CAT[1], label='host-country financing (Natural Earth assignment)')
    ax.axhline(eu_val('Germany (north)', 'Commercial', MOD), color=EU_COL['Germany (north)'], lw=1.2, label='Germany green, commercial')
    ax.set_xticks(x); ax.set_xticklabels(lab, fontsize=7.5); ax.set_ylabel('€/kg delivered (commercial, 2.15%)'); ax.legend(fontsize=7)
    ax.grid(axis='x', visible=False); ax.set_title('Sensitivity: financing the non-Namibian hexagons at host-country risk', loc='left')
    fig.tight_layout(); save(fig, 'fig12_country_sensitivity')
    KN['country_sens'] = {h: dict(country=c.loc[h].country, na=float(base.loc[h]), host=float(c.loc[h].delivered),
                                  na_flex=float(basef.loc[h]), host_flex=float(f.loc[h].onsite)) for h in c.index}
    allc = base.copy(); allc.loc[c.index] = c.delivered
    KN['country_sens_best'] = dict(hex=allc.loc[ON].idxmin(), value=float(allc.loc[ON].min()))

def choose_rep():
    c = af_tab('const', 'Commercial', MOD, 'delivered')
    na = [h for h in ON if HEX.loc[h].ne_iso == 'NA']
    s = c.loc[na].sort_values()
    rep = s.index[len(s) // 2]
    KN['rep_hex'] = rep; KN['best_hex'] = c.loc[ON].idxmin()
    return rep

if __name__ == '__main__':
    choose_rep()
    if len(sys.argv) > 1 and sys.argv[1] == 'rep':
        print(KN['rep_hex'], KN['best_hex']); sys.exit()
    fig_data(); fig_production(); fig_delivered_maps(); fig_breakdown(); fig_matrix(); fig_fossil(); fig_shipping()
    fig_infra(); fig_tornado(); fig_crp_rates(); fig_validation(); fig_country()
    def conv(o):
        if isinstance(o, dict): return {str(k): conv(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)): return [conv(v) for v in o]
        if isinstance(o, (np.floating, np.integer)): return o.item()
        return o
    json.dump(conv(KN), open(f'{OUT}/key_numbers.json', 'w'), indent=1)
    print('ok')
