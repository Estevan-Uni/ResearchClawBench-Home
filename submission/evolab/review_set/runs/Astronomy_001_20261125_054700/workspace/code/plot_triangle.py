"""Triangle plots of the mock chains (GetDist) + parameter-shift table/figure."""
import os, numpy as np, csv
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from getdist import loadMCSamples, plots
from common import load_data, SHARED, MODELS, NICE, COLORS, OUT, IMG, LABELS

os.makedirs(IMG, exist_ok=True)
d = load_data()

def load(suffix=''):
    return {m: loadMCSamples(os.path.join(OUT, 'chains', m + suffix), settings={'ignore_rows': 0,
            'smooth_scale_2D': 0.3, 'smooth_scale_1D': 0.3}) for m in MODELS}

for suffix, fname in [('', 'figure3_triangle.png'), ('_indep', 'figure3_triangle_indep.png')]:
    S = load(suffix)
    for m in S: S[m].label = NICE[m]
    g = plots.get_subplot_plotter(width_inch=11)
    g.settings.legend_fontsize = 15; g.settings.axes_labelsize = 13; g.settings.axes_fontsize = 9
    g.triangle_plot([S['lcdm'], S['ede'], S['w0wa']], SHARED, filled=True,
                    contour_colors=[COLORS['lcdm'], COLORS['ede'], COLORS['w0wa']],
                    legend_labels=[NICE['lcdm'] + ' (CMB+DESI)', NICE['ede'] + ' (CMB+DESI)',
                                   NICE['w0wa'] + ' (CMB+DESI)'], legend_loc='upper right')
    tag = 'assumed physical correlations' if suffix == '' else 'independent Gaussians'
    pass  # title in report caption
    g.export(os.path.join(IMG, fname), dpi=130)
    plt.close('all')

# EDE-specific triangle
S = load('')
S['ede'].label = 'EDE (CMB+DESI)'
g = plots.get_subplot_plotter(width_inch=8)
g.settings.axes_labelsize = 13; g.settings.legend_fontsize = 12
g.triangle_plot([S['ede']], ['f_EDE', 'log10_ac', 'H0', 'omega_m', 'sigma8', 'ns'], filled=True,
                contour_colors=[COLORS['ede']], legend_labels=['EDE (CMB+DESI), mock chain'])
# mark LCDM means on shared params
lc = d['lcdm_params']
for i, p in enumerate(['f_EDE', 'log10_ac', 'H0', 'omega_m', 'sigma8', 'ns']):
    for j, q in enumerate(['f_EDE', 'log10_ac', 'H0', 'omega_m', 'sigma8', 'ns']):
        if j > i: continue
        ax = g.subplots[i, j]
        if ax is None: continue
        if q in lc: ax.axvline(lc[q][0], color=COLORS['lcdm'], ls='--', lw=1)
        if i != j and p in lc: ax.axhline(lc[p][0], color=COLORS['lcdm'], ls='--', lw=1)
        if q == 'f_EDE': ax.axvline(0, color='k', ls=':', lw=1)
g.fig.text(0.6, 0.75, r'dashed: $\Lambda$CDM mean (CMB+DESI)' + '\n' + r'dotted: $f_\mathrm{EDE}=0$',
           fontsize=11, color=COLORS['lcdm'])
g.export(os.path.join(IMG, 'figure3b_ede_params.png'), dpi=130)
plt.close('all')

# 1D marginal statistics from chains (for the report) + shifts
rows = []
for m, key in MODELS.items():
    ms = S[m].getMargeStats()
    for p in d[key]:
        par = ms.parWithName(p)
        lim = par.limits[0]
        rows.append(dict(model=m, param=p, mean=par.mean, err=par.err, lo68=lim.lower, hi68=lim.upper))
with open(os.path.join(OUT, 'chain_marginals.csv'), 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

shift = []
for m in ['ede', 'w0wa']:
    for p in SHARED:
        mu, s = d[MODELS[m]][p]; mu0, s0 = lc[p]
        shift.append(dict(model=m, param=p, delta=mu - mu0, shift_sigma_ext=(mu - mu0) / s,
                          shift_sigma_lcdm=(mu - mu0) / s0, shift_sigma_quad=(mu - mu0) / np.hypot(s, s0),
                          err_ratio=s / s0))
with open(os.path.join(OUT, 'parameter_shifts.csv'), 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(shift[0].keys())); w.writeheader(); w.writerows(shift)
for r in shift: print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()})

# shift bar chart
fig, ax = plt.subplots(figsize=(8, 4.2))
x = np.arange(len(SHARED)); wd = 0.38
for k, m in enumerate(['ede', 'w0wa']):
    v = [r['shift_sigma_lcdm'] for r in shift if r['model'] == m]
    ax.bar(x + (k - 0.5) * wd, v, wd, color=COLORS[m], label=NICE[m])
    for xi, vi in zip(x + (k - 0.5) * wd, v):
        ax.text(xi, vi + (0.3 if vi >= 0 else -0.3), f'{vi:+.1f}', ha='center',
                va='bottom' if vi >= 0 else 'top', fontsize=8)
ax.axhline(0, color='k', lw=0.8)
ax.set_xticks(x); ax.set_xticklabels(['$' + LABELS[p] + '$' for p in SHARED], fontsize=11)
ax.set_ylabel(r'$(\mu_\mathrm{ext}-\mu_{\Lambda\mathrm{CDM}})/\sigma_{\Lambda\mathrm{CDM}}$')
ax.set_title(r'Shift of CMB+DESI parameter means relative to $\Lambda$CDM')
ax.legend(frameon=False); ax.set_ylim(-19.5, 16)
fig.tight_layout(); fig.savefig(os.path.join(IMG, 'figure_param_shifts.png'), dpi=140)
