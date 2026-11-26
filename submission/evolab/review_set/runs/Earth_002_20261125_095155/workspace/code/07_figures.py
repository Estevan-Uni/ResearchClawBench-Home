"""All report figures (PNG)."""
import numpy as np, json, csv, shapefile
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
import model as M
W = M.W; IMG = W + 'report/images/'
plt.rcParams.update({'font.size': 9, 'axes.edgecolor': '#888', 'axes.titlesize': 10, 'figure.dpi': 110})
R = np.load(W + 'outputs/site_results_central.npz')
lon, lat, wa = R['lon'], R['lat'], R['w_area']
SSPL = {'ssp245': 'SSP2-4.5', 'ssp370': 'SSP3-7.0', 'ssp585': 'SSP5-8.5'}
VL = {'A_MIT': 'Variant A (MIT-consistent, primary)', 'B_Knutson': 'Variant B (Knutson-2020 consensus)'}
SCOL = ['#d9d7d0', '#fde0a6', '#f9a65a', '#e8603c', '#b42a2f', '#4f0820']
TCCOL = ['#d9d7d0', '#9ec5f4', '#3987e5', '#104281']
SLCOL = ['#d9d7d0', '#f9c784', '#c2410c']
# ---- land polygons
sh = shapefile.Reader(W + 'outputs/ne/ne_50m_land.shp')
polys = []
for s in sh.shapes():
    pts = np.array(s.points); parts = list(s.parts) + [len(pts)]
    for a, b in zip(parts[:-1], parts[1:]):
        polys.append(pts[a:b])
def basemap(ax, ext=(-180, 180, -42, 40), step=None):
    ax.add_collection(PolyCollection(polys, facecolor='#f3f2ee', edgecolor='#b8b6ae', linewidth=0.3, zorder=0))
    ax.set_facecolor('#ffffff')
    if step is None: step = 60 if ext[1] - ext[0] > 100 else (10 if ext[1] - ext[0] > 25 else 5)
    ax.set_xticks(np.arange(np.ceil(ext[0] / step) * step, ext[1] + 1e-6, step))
    ys = 20 if step == 60 else step
    ax.set_yticks(np.arange(np.ceil(ext[2] / ys) * ys, ext[3] + 1e-6, ys))
    ax.tick_params(labelsize=7, length=2, colors='#555')
    ax.grid(color='#e5e4df', lw=0.4, zorder=0)
    ax.set_aspect('equal')
    ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])     # limits LAST
def scat(ax, val, cols, s=1.2, order=True, mask=None):
    m = np.ones(len(val), bool) if mask is None else mask
    idx = np.where(m)[0]
    if order: idx = idx[np.argsort(val[idx], kind='stable')]
    ax.scatter(lon[idx], lat[idx], c=np.array(cols)[val[idx]], s=s, lw=0, zorder=3, rasterized=True)
def legend(ax, cols, names, loc='lower left', title=None, ncol=1):
    ax.legend([Patch(color=c) for c in cols], names, loc=loc, fontsize=7, frameon=True, title=title, title_fontsize=7, ncol=ncol, framealpha=0.9)
def save(fig, name):
    fig.savefig(IMG + name, dpi=150, bbox_inches='tight'); plt.close(fig); print('saved', name)

# ===== Fig 1 data overview
tlon, tlat, wind = M.load_tc()
fig, axs = plt.subplots(3, 1, figsize=(10, 10.5))
ax = axs[0]; basemap(ax)
sc = ax.scatter(lon, lat, c=np.log10(wa), s=1, cmap='viridis', lw=0, rasterized=True)
fig.colorbar(sc, ax=ax, shrink=0.7, pad=0.01, label='log10 area weight per point (ha)')
ax.set_title(f'(a) GMW v4 reference sample points used (ref_cls = 1, n = {len(lon):,}); colour = CWON-2020 area weight')
ax = axs[1]; basemap(ax, (-180, 180, -50, 50))
rng = np.random.default_rng(0); sub = rng.choice(len(tlon), 60000, replace=False)
cls = np.digitize(wind[sub], [49, 58])
for c, col, n in zip(range(3), ['#9ec5f4', '#eb6834', '#4f0820'], ['L 33–49 m/s', 'M 49–58 m/s', 'H ≥58 m/s']):
    mm = cls == c; ax.scatter(tlon[sub][mm], tlat[sub][mm], s=0.4, c=col, lw=0, label=n, rasterized=True, zorder=2 + c)
ax.legend(markerscale=12, fontsize=7, loc='lower left')
ax.set_title('(b) MIT/MPI-ESM1-2-HR historical TC track points (≥33 m/s, random 60k of 200k shown)')
ax = axs[2]; basemap(ax)
sc = ax.scatter(lon, lat, c=R['rate_ssp245'], s=1.2, cmap='YlOrRd', vmin=3, vmax=11, lw=0, rasterized=True)
fig.colorbar(sc, ax=ax, shrink=0.7, pad=0.01, label='mm yr$^{-1}$')
ax.set_title('(c) AR6 median relative SLR rate, mean 2020–2100, SSP2-4.5 (nearest AR6 location)')
save(fig, 'fig01_data_overview.png')

# ===== Fig 2 baseline return periods and Kropf states
f0 = R['f0']; st = R['state']
fig, axs = plt.subplots(2, 1, figsize=(10, 7.2))
ax = axs[0]; basemap(ax)
rp = 1 / np.maximum(f0.sum(1), 1e-9)
bins = [0, 5, 10, 20, 50, 100, 250, 1e12]; rcol = ['#4f0820', '#b42a2f', '#e8603c', '#f9a65a', '#fde0a6', '#cde2fb', '#d9d7d0']
k = np.digitize(rp, bins[1:-1]); scat(ax, (6 - k), rcol[::-1])
legend(ax, rcol, ['<5', '5–10', '10–20', '20–50', '50–100', '100–250', '>250 (unaffected)'], title='Return period (yr), any class ≥33 m/s', ncol=2)
ax.set_title('(a) Baseline return period of hurricane-strength storms within 100 km (basin bias-corrected)')
ax = axs[1]; basemap(ax)
smax = st.max(1); STC = ['#d9d7d0', '#eda100', '#1baf7a', '#6250d6']
scat(ax, smax, STC)
legend(ax, STC, ['unaffected', 'vulnerable', 'resilient', 'dependent'], title='Kropf state (max over L/M/H), 5° units')
ax.set_title('(b) Baseline adaptation state of mangrove units (Kropf et al. Table 2 rules)')
save(fig, 'fig02_baseline_tc_states.png')

# ===== Fig 3 TC level maps
fig, axs = plt.subplots(3, 2, figsize=(15, 8.2))
for j, v in enumerate(M.VARIANTS):
    for i, ssp in enumerate(M.SSPS):
        ax = axs[i, j]; basemap(ax); scat(ax, R[f'tcl_{v}_{ssp}'], TCCOL, s=1.0)
        sh_ = [wa[R[f'tcl_{v}_{ssp}'] == c].sum() / wa.sum() for c in range(4)]
        ax.set_title(f'{VL[v].split(" (")[0]} · {SSPL[ssp]} — shifted area {100*(1-sh_[0]):.0f}%')
legend(axs[0, 0], TCCOL, ['0 no shift', '1 shift in L', '2 shift in M', '3 shift in H'], title='TC regime-shift level')
save(fig, 'fig03_tc_shift_levels.png')

# ===== Fig 4 SLR classes
fig = plt.figure(figsize=(14, 7)); gs = fig.add_gridspec(2, 3, height_ratios=[1, 0.8])
for i, ssp in enumerate(M.SSPS):
    ax = fig.add_subplot(gs[0, i]); basemap(ax, (-120, 180, -40, 35)); scat(ax, R[f'sl_{ssp}'], SLCOL, s=0.8)
    sh_ = [wa[R[f'sl_{ssp}'] == c].sum() / wa.sum() for c in range(3)]
    ax.set_title(f'{SSPL[ssp]}: ≥7 mm/yr = {100*sh_[2]:.0f}% of area')
legend(fig.axes[0], SLCOL, ['0: <4', '1: 4–7', '2: ≥7 mm/yr'], title='SLR level')
ax = fig.add_subplot(gs[1, :])
SLm = np.load(W + 'outputs/slr_sites.npz'); m1 = np.load(W + 'outputs/sites.npz')['cls'] == 1
hc = ['#2a78d6', '#eb6834', '#4f0820']
for c, ssp in zip(hc, M.SSPS):
    ax.hist(R[f'rate_{ssp}'], bins=np.arange(0, 16, 0.25), weights=wa / 1e6, histtype='step', lw=2, color=c, label=f'{SSPL[ssp]} median (2020–2100 mean)')
    ax.hist(SLm[f'{ssp}_q50_late'][m1], bins=np.arange(0, 16, 0.25), weights=wa / 1e6, histtype='step', lw=1, ls='--', color=c, label=f'{SSPL[ssp]} median (2080–2100 mean)')
for x in [4, 7]: ax.axvline(x, color='#333', lw=0.8, ls=':')
ax.set_xlabel('Relative SLR rate (mm yr$^{-1}$)'); ax.set_ylabel('Mangrove area (Mha per 0.25 mm/yr bin)'); ax.legend(fontsize=7, ncol=2)
ax.set_title('Area-weighted distribution of RSLR rates; dotted lines = Saintilan et al. thresholds (4, 7 mm/yr)')
save(fig, 'fig04_slr_levels.png')

# ===== Fig 5 global score maps
for v in M.VARIANTS:
    fig, axs = plt.subplots(3, 1, figsize=(11, 8.6))
    for i, ssp in enumerate(M.SSPS):
        ax = axs[i]; basemap(ax); scs = R[f'score_{v}_{ssp}']; scat(ax, scs, SCOL, s=1.6)
        ge3 = wa[scs >= 3].sum() / wa.sum(); s5 = wa[scs == 5].sum() / wa.sum()
        ax.set_title(f'({"abc"[i]}) {SSPL[ssp]} — composite score ≥3 (High–Severe): {100*ge3:.0f}% of area; Severe: {100*s5:.0f}%')
    legend(axs[0], SCOL, [f'{k} {n}' for k, n in enumerate(M.SCORE_NAMES)], title='Composite score (TC 0–3 + SLR 0–2)', ncol=2)
    fig.suptitle(f'Composite TC-regime-shift + SLR risk to mangroves by 2100 — {VL[v]}', fontsize=11)
    fig.tight_layout()
    save(fig, f'fig05_score_global_{v}.png')

# ===== Fig 6 regional zooms
REGZ = [('Philippines / Vietnam / S. China Sea', (100, 128, 5, 24)), ('Indonesia / Malaysia', (94, 142, -11, 8)),
        ('Mozambique / Madagascar', (30, 52, -27, -9)), ('Central America / Caribbean', (-100, -58, 7, 30)),
        ('SW Pacific (PNG, Solomons, Vanuatu, Fiji, NE Australia)', (140, 182, -24, 0))]
for v in M.VARIANTS:
    fig, axs = plt.subplots(3, 5, figsize=(20, 10.5))
    for i, ssp in enumerate(M.SSPS):
        for j, (nm, ext) in enumerate(REGZ):
            ax = axs[i, j]; scs = R[f'score_{v}_{ssp}']
            mm = (lon >= ext[0]) & (lon <= ext[1]) & (lat >= ext[2]) & (lat <= ext[3])
            basemap(ax, ext); scat(ax, scs, SCOL, s=5, mask=mm)
            g = wa[mm & (scs >= 3)].sum() / max(wa[mm].sum(), 1)
            ax.set_title(f'{nm if i == 0 else ""}\n{SSPL[ssp]}: score≥3 = {100*g:.0f}%' if i == 0 else f'{SSPL[ssp]}: score≥3 = {100*g:.0f}%', fontsize=9)
    legend(axs[0, 0], SCOL, M.SCORE_NAMES, loc='lower left', ncol=2)
    fig.suptitle(f'Regional hotspots of the composite score — {VL[v]}', fontsize=12)
    fig.tight_layout(); save(fig, f'fig06_regional_zooms_{v}.png')

# ===== Fig 7 stacked class shares
S = json.load(open(W + 'outputs/summary_central.json'))
fig, axs = plt.subplots(1, 3, figsize=(14, 4.6), sharey=True)
for a, (wk, lab) in enumerate([('area', 'Mangrove area (CWON 2020)'), ('pop', 'People protected (CWON)'), ('stock', 'Built capital protected (CWON)')]):
    ax = axs[a]; xl = []
    for x, (v, ssp) in enumerate([(v, s) for v in M.VARIANTS for s in M.SSPS]):
        sc_ = np.array(S[f'{v}|{ssp}'][wk]['score']) * 100; b = 0
        xp = x + (0.5 if v == 'B_Knutson' else 0)
        for k in range(6):
            ax.bar(xp, sc_[k], bottom=b, color=SCOL[k], width=0.8, edgecolor='white', lw=1); b += sc_[k]
        ax.text(xp, 101, f'{sc_[3:].sum():.0f}', ha='center', fontsize=7, color='#333')
        xl.append((xp, f'{SSPL[ssp][3:]}\n{"A" if v == "A_MIT" else "B"}'))
    ax.set_xticks([p for p, _ in xl]); ax.set_xticklabels([l for _, l in xl], fontsize=7)
    ax.set_title(lab); ax.set_ylim(0, 108); ax.spines[['top', 'right']].set_visible(False)
axs[0].set_ylabel('Share (%) — number above bar = % with score ≥3')
axs[2].legend([Patch(color=c) for c in SCOL[::-1]], [f'{k} {n}' for k, n in list(enumerate(M.SCORE_NAMES))[::-1]], fontsize=7, loc='center left', bbox_to_anchor=(1.01, 0.5))
save(fig, 'fig07_class_shares.png')

# ===== Fig 8 country ranking
rows = list(csv.DictReader(open(W + 'outputs/country_table.csv')))
rows = sorted(rows, key=lambda d: -float(d['area_ge3_A_MIT_ssp370']))[:25][::-1]
fig, axs = plt.subplots(1, 3, figsize=(15, 7.5), sharey=True)
y = np.arange(len(rows)); names = [d['country'][:22] for d in rows]
for a, (pref, lab, sc_) in enumerate([('area', 'Mangrove area at score ≥3 (kha)', 1e3), ('pop', 'People protected by mangroves at score ≥3 (thousand)', 1e3), ('stock', 'Built capital protected at score ≥3 (US$ M)', 1e6)]):
    ax = axs[a]
    tot = np.array([float(d[pref if pref != 'area' else 'area_ha']) for d in rows]) / sc_
    ax.barh(y, tot, color='#ecebe6', height=0.75, label='total in country')
    for off, ssp, col in [(-0.25, 'ssp245', '#f9a65a'), (0, 'ssp370', '#e8603c'), (0.25, 'ssp585', '#4f0820')]:
        ax.barh(y + off, [float(d[f'{pref}_ge3_A_MIT_{ssp}']) / sc_ for d in rows], height=0.25, color=col, label=f'{SSPL[ssp]} (A)')
    ax.scatter([float(d[f'{pref}_ge3_B_Knutson_ssp370']) / sc_ for d in rows], y, marker='|', s=60, color='#2a78d6', zorder=5, label='SSP3-7.0 (B)')
    ax.set_title(lab, fontsize=9); ax.spines[['top', 'right']].set_visible(False)
    if pref == 'stock': ax.set_xscale('symlog', linthresh=10)
axs[0].set_yticks(y); axs[0].set_yticklabels(names, fontsize=8); axs[0].legend(fontsize=7, loc='lower right')
fig.suptitle('Top 25 countries by mangrove area at composite score ≥3 under SSP3-7.0 (Variant A)', fontsize=11)
fig.tight_layout(); save(fig, 'fig08_country_ranking.png')

# ===== Fig 9 uncertainty
MC = json.load(open(W + 'outputs/montecarlo.json'))['res']
SE = json.load(open(W + 'outputs/sensitivity.json'))
fig, axs = plt.subplots(1, 3, figsize=(16, 5.2), gridspec_kw=dict(width_ratios=[1, 1.2, 1.6]))
ax = axs[0]; pos = []
for j, v in enumerate(M.VARIANTS):
    for i, ssp in enumerate(M.SSPS):
        d = np.array(MC[f'{v}|{ssp}']['ge3']) * 100; p = i + 4 * j
        bp = ax.boxplot(d, positions=[p], widths=0.6, whis=(5, 95), patch_artist=True, showfliers=False)
        bp['boxes'][0].set_facecolor(['#f9a65a', '#e8603c', '#b42a2f'][i]); bp['medians'][0].set_color('k')
        ax.scatter(p, S[f'{v}|{ssp}']['area']['ge3'] * 100, marker='D', color='#2a78d6', s=25, zorder=5)
        pos.append((p, f'{SSPL[ssp][3:]}\n{"A" if j == 0 else "B"}'))
ax.set_xticks([p for p, _ in pos]); ax.set_xticklabels([l for _, l in pos], fontsize=7)
ax.set_ylabel('Mangrove area with score ≥3 (%)'); ax.set_title('(a) Monte Carlo (n=60; box 25–75, whiskers 5–95%)\n◆ = central run')
ax = axs[1]
regs = ['SE Asia', 'SW Pacific', 'Central America & Caribbean', 'SE Africa']
for i, rg in enumerate(regs):
    for j, v in enumerate(M.VARIANTS):
        d = np.array(MC[f'{v}|ssp370']['reg'][rg]) * 100
        bp = ax.boxplot(d, positions=[i * 3 + j], widths=0.7, whis=(5, 95), patch_artist=True, showfliers=False)
        bp['boxes'][0].set_facecolor(['#e8603c', '#9ec5f4'][j]); bp['medians'][0].set_color('k')
ax.set_xticks([i * 3 + 0.5 for i in range(4)]); ax.set_xticklabels(['SE Asia', 'SW Pacific', 'C. America\n& Caribbean', 'SE Africa'], fontsize=8)
ax.legend([Patch(color='#e8603c'), Patch(color='#9ec5f4')], ['Variant A', 'Variant B'], fontsize=7)
ax.set_title('(b) Regional share with score ≥3, SSP3-7.0 (MC)'); ax.set_ylabel('%')
ax = axs[2]
base = SE['central (5deg unit, R100, q50, ref_cls=1)']['A_MIT|ssp370']['ge3'] * 100
items = [(k, v['A_MIT|ssp370']['ge3'] * 100, v.get('B_Knutson|ssp370', {}).get('ge3', np.nan) * 100) for k, v in SE.items() if not k.startswith('central')]
items.sort(key=lambda t: abs(t[1] - base))
yy = np.arange(len(items))
ax.barh(yy, [t[1] - base for t in items], left=base, color='#e8603c', height=0.6, label='Variant A')
ax.scatter([t[2] for t in items], yy, marker='|', color='#2a78d6', s=80, label='Variant B value')
ax.axvline(base, color='k', lw=0.8); ax.set_yticks(yy); ax.set_yticklabels([t[0] for t in items], fontsize=7)
ax.set_xlabel('Area share with score ≥3, SSP3-7.0 (%)'); ax.set_title(f'(c) One-at-a-time sensitivity (central A = {base:.0f}%)'); ax.legend(fontsize=7, loc='lower right')
fig.tight_layout(); save(fig, 'fig09_uncertainty.png')

# ===== Fig 10 validation
V = json.load(open(W + 'outputs/validation.json'))
fig, axs = plt.subplots(1, 3, figsize=(15, 4.6))
ax = axs[0]
for n, d in V['ref'].items():
    o, mdl, mu = d['obs_rp'], d['rp_R100'], d['rp_R100_nocorr']
    mdl = min(mdl, 1000); mu = min(mu, 1000)
    ax.plot([o, o], [mu, mdl], color='#bbb', lw=0.8)
    ax.scatter(o, mu, marker='o', facecolor='none', edgecolor='#888', s=20)
    ax.scatter(o, mdl, color='#2a78d6', s=22, zorder=4); ax.annotate(n, (o, mdl), fontsize=6.5, xytext=(3, 2), textcoords='offset points')
xx = np.array([1, 1000]); ax.plot(xx, xx, 'k-', lw=0.7); ax.fill_between(xx, xx / 2, xx * 2, color='#eee', zorder=0)
ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlim(1, 100); ax.set_ylim(1, 1000)
ax.set_xlabel('Approx. observed return period (yr)'); ax.set_ylabel('Modelled return period, R=100 km (yr)')
ax.set_title('(a) TC return periods: ● basin-corrected, ○ uncorrected\n(shaded = factor 2)')
ax = axs[1]
SLm = np.load(W + 'outputs/slr_sites.npz'); d = SLm['nn_dist_km'][m1]
ax.hist(d, bins=50, color='#2a78d6', edgecolor='white'); ax.set_xlabel('Distance to nearest AR6 location (km)'); ax.set_ylabel('Number of sites')
ax.set_title(f'(b) SLR nearest-neighbour distance (median {np.median(d):.0f} km, max {d.max():.0f} km)')
ax = axs[2]
B = json.load(open(W + 'outputs/basin_direction.json')); x = np.arange(len(B))
ax.bar(x - 0.2, [b['A_MIT'] * 100 for b in B], 0.4, color='#e8603c', label='Variant A (SSP2-4.5)')
ax.bar(x + 0.2, [b['B_Knutson'] * 100 for b in B], 0.4, color='#9ec5f4', label='Variant B (SSP2-4.5)')
for i, b in enumerate(B): ax.text(i, -28, 'Mo: ' + ('↑' if b['mo2023'] == '+' else '↓'), ha='center', fontsize=8)
ax.axhline(0, color='k', lw=0.6); ax.set_xticks(x); ax.set_xticklabels([b['basin'] for b in B]); ax.set_ylim(-32, 50)
ax.set_ylabel('Change in damage-weighted TC exposure (%)'); ax.legend(fontsize=7)
ax.set_title('(c) Basin direction vs. Mo et al. (2023) risk-change signs')
fig.tight_layout(); save(fig, 'fig10_validation.png')

# ===== Fig 11 probability map
P = np.load(W + 'outputs/mc_probmap.npz')
fig, axs = plt.subplots(2, 1, figsize=(10, 7.4))
for i, v in enumerate(M.VARIANTS):
    ax = axs[i]; basemap(ax); p = P[f'{v}_ssp370']; o = np.argsort(p)
    sc = ax.scatter(lon[o], lat[o], c=p[o], cmap='magma_r', vmin=0, vmax=1, s=1.4, lw=0, rasterized=True)
    fig.colorbar(sc, ax=ax, shrink=0.75, pad=0.01, label='P(score ≥ 3)')
    ax.set_title(f'({"ab"[i]}) Monte Carlo probability of High-to-Severe risk, SSP3-7.0 — {VL[v]}')
save(fig, 'fig11_probability_map.png')
