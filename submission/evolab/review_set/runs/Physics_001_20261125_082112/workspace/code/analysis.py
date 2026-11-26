"""Full analysis of the simulated MATBG superfluid-stiffness dataset.

Run:  python code/analysis.py     (from /workspace)
Produces outputs/results.json and report/images/*.png.  Fixed RNG seed.
"""
import os, json, warnings
import numpy as np
from scipy.optimize import curve_fit, nnls, minimize_scalar
import matplotlib
import matplotlib.ticker
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from parse_data import parse

warnings.filterwarnings('ignore', category=RuntimeWarning)
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
IMG = os.path.join(ROOT, 'report', 'images'); OUT = os.path.join(ROOT, 'outputs')
os.makedirs(IMG, exist_ok=True); os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(20241125)
plt.rcParams.update({'font.size': 9, 'axes.titlesize': 10, 'axes.labelsize': 9,
                     'legend.fontsize': 7.5, 'figure.dpi': 140, 'savefig.bbox': 'tight'})
C = dict(blue='#1f5fa8', orange='#d9711c', green='#2e8b57', red='#c0392b', purple='#7d4fa0',
         grey='#666666', teal='#16a3a3', brown='#8c564b')

D, P = parse()
R = {}  # results


def r2(y, yhat):
    return float(1 - np.sum((y - yhat) ** 2) / np.sum((y - np.mean(y)) ** 2))


def rms(y, yhat):
    return float(np.sqrt(np.mean((y - yhat) ** 2)))


def bic(y, yhat, p):
    n = len(y); rss = np.sum((y - yhat) ** 2)
    return float(n * np.log(rss / n) + p * np.log(n))


# ---------------------------------------------------------------- 0. inventory
coord = {'D_s_conv': 'n_eff', 'D_s_geom': 'n_eff', 'D_s_exp_hole': 'n_eff', 'D_s_exp_electron': 'n_eff',
         'D_s_bcs': 'T', 'D_s_nodal': 'T', 'D_s_power_n2': 'T', 'D_s_power_n2_5': 'T', 'D_s_power_n3': 'T',
         'D_s_experimental': 'T', 'D_s_gl': 'I_dc', 'D_s_linear': 'I_dc', 'D_s_dc_exp': 'I_dc',
         'I_mw_amplitude': 'P_mw', 'D_s_mw_exp': 'P_mw'}
R['inventory'] = {k: {'len': len(D[k]), 'coord': c, 'coord_len': len(D[c]),
                      'matched': len(D[k]) == len(D[c])} for k, c in coord.items()}
R['parameters'] = P
assert len(D['T']) == 100 and len(D['D_s_experimental']) == 110 and len(D['D_s_dc_exp']) == 85
for k in ['D_s_experimental', 'D_s_dc_exp', 'D_s_mw_exp']:
    assert abs(D[k][0] - 100) < 1e-9
dT = float(D['T'][1] - D['T'][0]); dI = float(D['I_dc'][1] - D['I_dc'][0])
R['grid_steps'] = {'dT_K': dT, 'dI_nA': dI}

# ======================================================================= 1. DENSITY
n = D['n_eff']; dc, dg = D['D_s_conv'], D['D_s_geom']; dh, de = D['D_s_exp_hole'], D['D_s_exp_electron']
n15 = n / 1e15
dens = {}
eh = de / dh; dens['e_over_h'] = {'min': float(eh.min()), 'max': float(eh.max()), 'mean': float(eh.mean())}
assert eh.std() < 1e-3
for lab, y in [('hole', dh), ('electron', de)]:
    s = {}
    for mlab, m in [('conv', dc), ('geom', dg)]:
        rat = y / m
        s[f'exp_over_{mlab}'] = {'min': float(rat.min()), 'median': float(np.median(rat)), 'max': float(rat.max()),
                                 'log10_min': float(np.log10(rat.min())), 'log10_median': float(np.log10(np.median(rat))),
                                 'log10_max': float(np.log10(rat.max()))}
    # NNLS decomposition
    A = np.vstack([dc, dg]).T; coef, res = nnls(A, y); yh = A @ coef
    s['nnls'] = {'a_conv': float(coef[0]), 'b_geom': float(coef[1]), 'R2': r2(y, yh),
                 'rel_rms_resid': float(rms(y, yh) / y.mean())}
    # one-parameter scaled models
    for mlab, m in [('conv', dc), ('geom', dg)]:
        b = float(m @ y / (m @ m)); s[f'scaled_{mlab}'] = {'b': b, 'R2': r2(y, b * m), 'rel_rms': rms(y, b * m) / y.mean()}
    # linear in n
    sl, ic = np.polyfit(n15, y, 1); s['linear'] = {'slope_per_1e15': float(sl), 'n0_1e15': float(-ic / sl),
                                                   'R2': r2(y, sl * n15 + ic)}
    # power law
    al, la = np.polyfit(np.log(n), np.log(y), 1); s['loglog_exponent'] = float(al)
    s['loglog_R2'] = r2(np.log(y), al * np.log(n) + la)
    s['vF_required_conv_only_m_per_s'] = {'median': float(700 * np.median(y / dc)), 'min': float(700 * (y / dc).min()),
                                          'max': float(700 * (y / dc).max())}
    s['vF_required_geom_scaling_m_per_s'] = {'median': float(3000 * np.median(y / dg))}
    s['first_last_increment'] = float(y[-1] / y[0])
    dens[lab] = s
for mlab, m in [('conv', dc), ('geom', dg)]:
    al, _ = np.polyfit(np.log(n), np.log(m), 1); dens[f'loglog_exponent_{mlab}'] = float(al)
    dens[f'argmax_{mlab}_1e15'] = float(n15[np.argmax(m)])
    dens[f'increase_{mlab}'] = float(m.max() / m[0])
gc = dg / dc
dens['geom_over_conv'] = {'min': float(gc.min()), 'max': float(gc.max()), 'vF_ratio': 3000 / 700}
# reverse engineering: D_model = C * v_F * sqrt(n) * f(n)
Cc = dc / (700 * np.sqrt(n)); Cg = dg / (3000 * np.sqrt(n))
dens['model_structure'] = {'prefactor_conv_low_n': float(Cc[0]), 'prefactor_geom_low_n': float(Cg[0]),
                           'f_conv_at_max_n': float(Cc[-1] / Cc[0]), 'f_geom_at_max_n': float(Cg[-1] / Cg[0]),
                           'loglog_slope_first5_conv': float(np.polyfit(np.log(n[:5]), np.log(dc[:5]), 1)[0]),
                           'loglog_slope_first5_geom': float(np.polyfit(np.log(n[:5]), np.log(dg[:5]), 1)[0])}
# experimental relative scatter about smooth fit (quadratic in n)
pq = np.polyfit(n15, dh, 2); dens['hole_rel_scatter_about_quadratic'] = float(np.std(dh / np.polyval(pq, n15) - 1))
R['density'] = dens

fig, ax = plt.subplots(1, 3, figsize=(13, 3.9))
a = ax[0]
a.scatter(n15, dh / 1e10, s=16, color=C['red'], label='Exp. hole-doped', zorder=3)
a.scatter(n15, de / 1e10, s=16, marker='s', facecolor='none', edgecolor=C['blue'], label='Exp. electron-doped', zorder=3)
a.plot(n15, dc / 1e10, color=C['grey'], lw=1.8, label=r'Conventional FL ($v_F$=700 m/s)')
a.plot(n15, dg / 1e10, color=C['green'], lw=1.8, label=r'Quantum-geometric ($v_F$=3000 m/s)')
b = dens['hole']['scaled_geom']['b']
a.plot(n15, b * dg / 1e10, '--', color=C['green'], lw=1.2, label=fr'{b:.1f}$\times$geom (shape test, $R^2$={dens["hole"]["scaled_geom"]["R2"]:.2f})')
L = dens['hole']['linear']
a.plot(n15, (L['slope_per_1e15'] * (n15 - L['n0_1e15'])) / 1e10, ':', color=C['red'], lw=1.2,
       label=fr'linear fit hole ($R^2$={L["R2"]:.3f})')
a.set_xlabel(r'$n_{\rm eff}$ ($10^{15}$ m$^{-2}$)'); a.set_ylabel(r'$D_s$ ($10^{10}$, dataset units)')
a.set_title('(a) Stiffness vs carrier density (linear axes)'); a.legend(loc='upper left', frameon=False)
a.set_ylim(0, 26)
a = ax[1]
for lab, y, mk in [('hole', dh, 'o'), ('electron', de, 's')]:
    a.plot(n15, y / dc, mk, ms=3.5, color=C['grey'] if lab == 'hole' else C['brown'], mfc='none' if lab == 'electron' else None,
           label=f'{lab} / conventional')
    a.plot(n15, y / dg, mk, ms=3.5, color=C['green'] if lab == 'hole' else C['teal'], mfc='none' if lab == 'electron' else None,
           label=f'{lab} / geometric')
a.plot(n15, gc, '-', color=C['purple'], label='geometric / conventional')
a.axhline(10, color='k', lw=0.8, ls='--'); a.text(0.6, 11, 'order of magnitude', fontsize=7)
a.set_yscale('log'); a.set_xlabel(r'$n_{\rm eff}$ ($10^{15}$ m$^{-2}$)'); a.set_ylabel('ratio')
a.set_title('(b) Enhancement ratios'); a.set_ylim(2, 300); a.legend(frameon=False, ncol=2, loc='upper left', fontsize=6.5)
a = ax[2]
for y, lab, col, st in [(dh, 'exp hole', C['red'], 'o'), (dc, 'conv', C['grey'], '-'), (dg, 'geom', C['green'], '-')]:
    al = np.polyfit(np.log(n), np.log(y), 1)[0]
    if st == 'o': a.loglog(n15, y, 'o', ms=3, color=col, label=fr'{lab}: slope {al:.2f}')
    else: a.loglog(n15, y, st, color=col, label=fr'{lab}: slope {al:.2f}')
a.loglog(n15, Cc[0] * 700 * np.sqrt(n), ':', color=C['grey'], lw=1, label=r'$\propto v_F\sqrt{n}$ envelope')
a.loglog(n15, Cg[0] * 3000 * np.sqrt(n), ':', color=C['green'], lw=1)
a.set_xlabel(r'$n_{\rm eff}$ ($10^{15}$ m$^{-2}$)'); a.set_ylabel(r'$D_s$'); a.set_title('(c) Log-log scaling')
a.set_xticks([0.5, 1, 2, 5]); a.set_xticklabels(['0.5', '1', '2', '5']); a.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
a.set_ylim(5e8, 1e12); a.legend(frameon=False, loc='upper left', fontsize=7)
fig.tight_layout(); fig.savefig(os.path.join(IMG, 'fig2_density.png')); plt.close(fig)

# ======================================================================= 2. TEMPERATURE
T = D['T']; Tc = P['temperature_dependence.py']['T_c']
temp = {}
# ---- 2a template audit
aud = {}
aud['bcs_equals_power_n2'] = bool(np.allclose(D['D_s_bcs'], D['D_s_power_n2']))
for key, nom in [('D_s_nodal', 1.0), ('D_s_power_n2', 2.0), ('D_s_power_n2_5', 2.5), ('D_s_power_n3', 3.0)]:
    y = D[key]; k = np.arange(len(y)); Tk = T[:len(y)]
    pred = 100 * np.clip(1 - (Tk / Tc) ** nom, 0, None)
    e = {'len': len(y), 'max_abs_dev_from_nominal_on_T_grid': float(np.max(np.abs(y - pred))),
         'n_nonzero': int(np.sum(y > 0))}
    m = (y > 0) & (y < 100) & (Tk < Tc)
    try:
        f = lambda t, nn, tc: 100 * (1 - (t / tc) ** nn)
        pp, cv = curve_fit(f, Tk[m], y[m], p0=[nom, 1.0])
        e['free_fit_full_range'] = {'n': float(pp[0]), 'n_err': float(np.sqrt(cv[0, 0])), 'Tc': float(pp[1]),
                                    'R2': r2(y[m], f(Tk[m], *pp))}
        mm = m & (k <= 40)
        uu = 1 - y[mm] / 100
        e['lowT_loglog_exponent_k_le_40'] = float(np.polyfit(np.log(Tk[mm]), np.log(uu), 1)[0])
        # where does it hit zero (first zero index)
        e['first_zero_index'] = int(np.argmax(y <= 0)) if np.any(y <= 0) else None
        tinv = (1 - y[y > 0] / 100) ** (1 / nom); st = np.diff(tinv)
        e['implied_T_step_first'] = float(st[0]); e['implied_T_step_last'] = float(st[-1])
        e['implied_T_step_ratio_max_min'] = float(st.max() / st.min())
    except Exception as ex:
        e['fit_error'] = str(ex)
    aud[key] = e
temp['template_audit'] = aud

# ---- 2b clean-limit theory curves
xi = np.linspace(0, 1, 1)  # placeholder


def gap_T(t):
    return np.where(t < 1, np.tanh(1.74 * np.sqrt(np.clip(1 / np.maximum(t, 1e-9) - 1, 0, None))), 0.0)


def stiffness_curve(t_arr, kind, nth=361, nxi=3000):
    th = np.linspace(0, 2 * np.pi, nth, endpoint=False)
    if kind == 's':
        d0 = 1.764 * np.ones_like(th)
    elif kind == 'd':
        d0 = 2.14 * np.abs(np.cos(2 * th))
    elif kind == 'aniso':  # nodeless anisotropic, Delta_min/Delta_max = 0.2
        d0 = 1.9 * (1 + (2 / 3) * np.cos(6 * th))
    out = []
    for t in t_arr:
        if t <= 0: out.append(1.0); continue
        if t >= 1: out.append(0.0); continue
        dl = d0 * gap_T(t)
        xm = 40 * t
        xg = np.linspace(0, xm, nxi)
        E = np.sqrt(xg[None, :] ** 2 + dl[:, None] ** 2)
        z = np.exp(-E / t)
        kern = z / (1 + z) ** 2 / t  # = -df/dE = (1/4t) sech^2(E/2t)
        integ = 2 * np.trapezoid(kern, xg, axis=1)  # over xi in (-inf, inf)
        out.append(1 - np.mean(integ))
    return np.array(out)


t_th = np.concatenate([np.linspace(0, 0.2, 41), np.linspace(0.205, 1.0, 160)])
theory = {k: stiffness_curve(t_th, k) for k in ['s', 'aniso', 'd']}
th_res = {}
for k, v in theory.items():
    u = 1 - v
    for lo, hi in [(0.05, 0.3), (0.1, 0.5), (0.2, 0.6)]:
        m = (t_th >= lo) & (t_th <= hi) & (u > 1e-12)
        th_res[f'{k}_exponent_{lo}-{hi}'] = float(np.polyfit(np.log(t_th[m]), np.log(u[m]), 1)[0]) if m.sum() > 3 else None
    th_res[f'{k}_rho_at_t0.2'] = float(np.interp(0.2, t_th, v))
temp['theory_curves'] = th_res
rho_interp = {k: (lambda tt, v=v: np.interp(tt, t_th, v, right=0.0)) for k, v in theory.items()}

# ---- 2c experimental series (index-based, mapping-invariant fits)
y = D['D_s_experimental']; N = len(y); k = np.arange(N, dtype=float); u = 1 - y / 100
exp = {}
d2 = np.diff(y, 2); kinks = np.where(d2 > 0.015)[0] + 1
exp['kink_indices'] = kinks.tolist(); exp['kink_spacing_after_41'] = int(np.median(np.diff(kinks[kinks >= 41])))
exp['final_value'] = float(y[-1]); exp['monotonic'] = bool(np.all(np.diff(y) < 0))
exp['concave_fraction_d2_lt_0_smoothed'] = None
# smoothed concavity: fit cubic spline-like polynomial
ps = np.polyfit(k, y, 5); d2s = np.polyval(np.polyder(ps, 2), k)
exp['smoothed_d2_positive_fraction'] = float(np.mean(d2s > 0))  # >0 means convex in D -> concave in u

kk = k[1:]; uu = u[1:]; yy = y[1:]
models = {}


def fitmodel(name, f, p0, npar, x=kk, yv=yy, bounds=(-np.inf, np.inf)):
    try:
        pp, cv = curve_fit(f, x, yv, p0=p0, maxfev=40000, bounds=bounds)
        yh = f(x, *pp)
        models[name] = {'params': [float(q) for q in pp], 'perr': [float(q) for q in np.sqrt(np.diag(cv))],
                        'R2': r2(yv, yh), 'RMS': rms(yv, yh), 'BIC': bic(yv, yh, npar), 'npar': npar}
        return pp
    except Exception as ex:
        models[name] = {'error': str(ex)}
        return None


p_pow = fitmodel('free_power', lambda x, A, nn: 100 * (1 - A * x ** nn), [0.01, 0.7], 2)
for nfix in [1.0, 2.0, 2.5, 3.0]:
    fitmodel(f'fixed_n{nfix}', lambda x, A, nf=nfix: 100 * (1 - A * x ** nf), [1e-3], 1)
fitmodel('power_with_offset', lambda x, A, nn, x0: 100 * (1 - A * ((x + x0) ** nn - x0 ** nn)), [0.01, 0.7, 1.0], 3,
         bounds=([0, 0.05, 0], [10, 5, 500]))
p_act = fitmodel('activated_swave', lambda x, A, B: 100 * (1 - A * np.exp(-B / x)), [0.5, 30], 2,
                 bounds=([0, 0], [100, 1e5]))
fitmodel('activated_sqrt_prefactor', lambda x, A, B: 100 * (1 - A * np.sqrt(B / x) * np.exp(-B / x)), [0.5, 30], 2,
         bounds=([0, 1e-6], [100, 1e5]))
fitmodel('exp_saturation', lambda x, A, tau: 100 * (1 - A * (1 - np.exp(-x / tau))), [0.4, 80], 2)


def fit_theory(kind):
    best = None
    for s0 in np.geomspace(1e-4, 0.05, 30):
        f = lambda x, s, D0: D0 * rho_interp[kind](x * s)
        try:
            pp, _ = curve_fit(f, kk, yy, p0=[s0, 100], bounds=([1e-6, 50], [1, 150]), maxfev=20000)
            rr = np.sum((yy - f(kk, *pp)) ** 2)
            if best is None or rr < best[1]: best = (pp, rr)
        except Exception:
            pass
    pp = best[0]; yh = pp[1] * rho_interp[kind](kk * pp[0])
    models[f'theory_{kind}'] = {'params': [float(pp[0]), float(pp[1])], 'R2': r2(yy, yh), 'RMS': rms(yy, yh),
                                'BIC': bic(yy, yh, 2), 'npar': 2, 'k_at_Tc': float(1 / pp[0])}
    return pp


p_th = {kd: fit_theory(kd) for kd in ['s', 'aniso', 'd']}
bmin = min(m['BIC'] for m in models.values() if 'BIC' in m)
for m in models.values():
    if 'BIC' in m: m['dBIC'] = m['BIC'] - bmin
exp['models'] = models
nfree = p_pow[1]; Afree = p_pow[0]
# block bootstrap of free exponent (residual moving blocks, L=10)
res0 = yy - 100 * (1 - Afree * kk ** nfree); Lb = 10; nb = 1000; boots = []
for _ in range(nb):
    idx = np.concatenate([np.arange(s, s + Lb) for s in rng.integers(0, len(res0) - Lb, size=len(res0) // Lb + 1)])[:len(res0)]
    ys = 100 * (1 - Afree * kk ** nfree) + res0[idx]
    try:
        pp, _ = curve_fit(lambda x, A, nn: 100 * (1 - A * x ** nn), kk, ys, p0=[Afree, nfree]); boots.append(pp[1])
    except Exception:
        pass
boots = np.array(boots)
exp['free_exponent'] = {'n': float(nfree), 'A': float(Afree), 'boot_CI95': [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
                        'boot_std': float(boots.std())}
# windowed fits
win = {}
for kmax in [20, 40, 60, 80, 109]:
    m = kk <= kmax
    pp, cv = curve_fit(lambda x, A, nn: 100 * (1 - A * x ** nn), kk[m], yy[m], p0=[0.01, 0.7])
    win[str(kmax)] = {'n': float(pp[1]), 'n_err': float(np.sqrt(cv[1, 1]))}
exp['windowed_exponent'] = win
# running exponent (local log-log slope, window 15 points)
run_k, run_n = [], []
for c0 in range(8, N - 7):
    sl = slice(c0 - 7, c0 + 8)
    run_k.append(c0); run_n.append(np.polyfit(np.log(k[sl]), np.log(u[sl]), 1)[0])
run_k, run_n = np.array(run_k), np.array(run_n)
exp['running_exponent'] = {'min': float(run_n.min()), 'max': float(run_n.max()), 'median': float(np.median(run_n))}
# required non-affine grid: T_k ∝ k^p
exp['required_grid_power_p'] = {str(nt): float(nfree / nt) for nt in [1.0, 2.0, 2.5, 3.0]}
# non-affine "natural" grids
nat = {}
for name, tgrid in [('sqrt', np.sqrt(k / k[-1])), ('square', (k / k[-1]) ** 2),
                    ('geomspace_0.01-1', np.concatenate([[0], np.geomspace(0.01, 1, N - 1)])),
                    ('log1p', np.log1p(k) / np.log1p(k[-1]))]:
    tt = tgrid[1:]
    try:
        pp, _ = curve_fit(lambda x, A, nn: 100 * (1 - A * x ** nn), tt, yy, p0=[0.3, 1.0], maxfev=20000)
        nat[name] = {'n': float(pp[1]), 'R2': r2(yy, 100 * (1 - pp[0] * tt ** pp[1]))}
    except Exception as ex:
        nat[name] = {'error': str(ex)}
exp['nonaffine_grids'] = nat
# mappings to T/Tc
k_c_free = Afree ** (-1 / nfree)  # index at which free power law reaches zero
maps = {'a_supplied_step': k * dT / Tc, 'b_powerlaw_zero_at_Tc': k / k_c_free, 'c_spread_0_to_Tc': k / k[-1]}
exp['mappings'] = {'a_supplied_step_tmax': float(maps['a_supplied_step'][-1]),
                   'b_powerlaw_zero_at_Tc_tmax': float(maps['b_powerlaw_zero_at_Tc'][-1]), 'b_k_c': float(k_c_free),
                   'c_spread_0_to_Tc_tmax': 1.0,
                   'note': 'For x0=0 affine maps the exponent and all model BIC values are invariant; only the T scale changes.'}
exp['D_at_index_where_supplied_T_equals_Tc'] = float(np.interp(Tc / dT, k, y))
# gap estimate from activated fit under each mapping (Delta/kB in units of Tc)
exp['activated_gap_over_Tc_under_mapping'] = {mk: float(p_act[1] * (tv[1] - tv[0])) for mk, tv in maps.items()}
temp['experimental'] = exp

# ---- 2d injection-recovery (synthetic, method validation)
sig = rms(yy, 100 * (1 - Afree * kk ** nfree))
r_ = res0 - res0.mean(); phi = float(np.corrcoef(r_[:-1], r_[1:])[0, 1])
tg = T[T < Tc]; tg = tg[1:]
inj = {'sigma': sig, 'phi_AR1': phi, 'n_true': 2.5, 'n_rep': 200}
rec_n, dB_act, dB_n1, rec_n_white = [], [], [], []
for rep in range(200):
    e = np.zeros(len(tg)); e[0] = rng.normal() * sig
    for i in range(1, len(tg)): e[i] = phi * e[i - 1] + np.sqrt(1 - phi ** 2) * sig * rng.normal()
    ys = 100 * (1 - (tg / Tc) ** 2.5) + e
    pp, _ = curve_fit(lambda x, tc, nn: 100 * (1 - (x / tc) ** nn), tg, ys, p0=[1.0, 2.0], maxfev=20000)
    rec_n.append(pp[1]); b_pow = bic(ys, 100 * (1 - (tg / pp[0]) ** pp[1]), 2)
    pa, _ = curve_fit(lambda x, A, B: 100 * (1 - A * np.exp(-B / x)), tg, ys, p0=[3, 1.5], bounds=([0, 0], [1e4, 100]), maxfev=20000)
    dB_act.append(bic(ys, 100 * (1 - pa[0] * np.exp(-pa[1] / tg)), 2) - b_pow)
    p1, _ = curve_fit(lambda x, tc: 100 * (1 - (x / tc)), tg, ys, p0=[1.0])
    dB_n1.append(bic(ys, 100 * (1 - tg / p1[0]), 1) - b_pow)
    ysw = 100 * (1 - (tg / Tc) ** 2.5) + rng.normal(size=len(tg)) * sig
    pw, _ = curve_fit(lambda x, tc, nn: 100 * (1 - (x / tc) ** nn), tg, ysw, p0=[1.0, 2.0], maxfev=20000)
    rec_n_white.append(pw[1])
rec_n = np.array(rec_n); dB_act = np.array(dB_act); dB_n1 = np.array(dB_n1); rec_n_white = np.array(rec_n_white)
inj.update({'recovered_n_median': float(np.median(rec_n)), 'recovered_n_CI95': [float(np.percentile(rec_n, 2.5)), float(np.percentile(rec_n, 97.5))],
            'recovered_n_white_median': float(np.median(rec_n_white)),
            'recovered_n_white_CI95': [float(np.percentile(rec_n_white, 2.5)), float(np.percentile(rec_n_white, 97.5))],
            'dBIC_activated_minus_power_median': float(np.median(dB_act)), 'frac_activated_rejected_dBIC>10': float(np.mean(dB_act > 10)),
            'dBIC_n1_minus_power_median': float(np.median(dB_n1)), 'frac_n1_rejected_dBIC>10': float(np.mean(dB_n1 > 10))})
# low-T-only window version (t<=0.5)
m5 = tg <= 0.5; rn5 = []
for rep in range(200):
    ys = 100 * (1 - (tg[m5] / Tc) ** 2.5) + rng.normal(size=m5.sum()) * sig
    pp, _ = curve_fit(lambda x, A, nn: 100 * (1 - A * x ** nn), tg[m5], ys, p0=[1, 2], maxfev=20000); rn5.append(pp[1])
rn5 = np.array(rn5)
inj['lowT_window_t_le_0.5'] = {'median': float(np.median(rn5)), 'CI95': [float(np.percentile(rn5, 2.5)), float(np.percentile(rn5, 97.5))]}
temp['injection_recovery'] = inj
R['temperature'] = temp

# ---- figures: templates + theory
fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
a = ax[0]
for key, col, nom in [('D_s_nodal', C['blue'], 1), ('D_s_bcs', C['red'], 2), ('D_s_power_n2_5', C['orange'], 2.5), ('D_s_power_n3', C['purple'], 3)]:
    yv = D[key]; a.plot(T[:len(yv)], yv, 'o', ms=2.5, color=col, label=f'{key} (len {len(yv)})')
    a.plot(T, 100 * np.clip(1 - T ** nom, 0, None), '-', lw=0.8, color=col)
a.set_xlabel(r'supplied $T$ grid (K), H1: index $\to T_k$'); a.set_ylabel(r'$D_s$ (% of $D_{s0}$)')
a.set_title('(a) Template audit: points vs nominal $1-(T/T_c)^n$ (lines)'); a.legend(frameon=False)
a = ax[1]
for kd, col, lab in [('s', C['red'], 's-wave (isotropic)'), ('aniso', C['orange'], r'anisotropic nodeless ($\Delta_{min}/\Delta_{max}$=0.2)'), ('d', C['blue'], 'd-wave (nodal)')]:
    a.plot(t_th, theory[kd], color=col, label=lab)
a.set_xlabel(r'$T/T_c$'); a.set_ylabel(r'$D_s(T)/D_s(0)$'); a.set_title('(b) Clean-limit BCS stiffness'); a.legend(frameon=False)
a = ax[2]
for kd, col in [('s', C['red']), ('aniso', C['orange']), ('d', C['blue'])]:
    uu_ = 1 - theory[kd]; m = uu_ > 1e-9
    a.loglog(t_th[m], uu_[m], color=col, label=f'{kd}: n_eff(0.1-0.5)={th_res[f"{kd}_exponent_0.1-0.5"]:.1f}')
tt_ = np.linspace(0.05, 1, 50)
for nn, ls in [(1, ':'), (2, '--'), (3, '-.')]:
    a.loglog(tt_, 0.5 * tt_ ** nn, ls, color='k', lw=0.7, label=fr'$\propto t^{nn}$')
a.set_ylim(1e-6, 1.5); a.set_xlim(0.04, 1); a.set_xlabel(r'$T/T_c$'); a.set_ylabel(r'$1-D_s/D_{s0}$')
a.set_title('(c) Low-T suppression of theory curves'); a.legend(frameon=False, loc='lower right')
fig.tight_layout(); fig.savefig(os.path.join(IMG, 'fig3_templates_theory.png')); plt.close(fig)

# ---- figure: experimental temperature dependence
fig = plt.figure(figsize=(13, 8.2))
gs = fig.add_gridspec(2, 3)
a = fig.add_subplot(gs[0, 0:2])
mapcols = {'a_supplied_step': C['grey'], 'b_powerlaw_zero_at_Tc': C['blue'], 'c_spread_0_to_Tc': C['purple']}
maplabs = {'a_supplied_step': r'(a) $T_k=k\,\Delta T$ (supplied step)', 'b_powerlaw_zero_at_Tc': r'(b) free power law $\to0$ at $T_c$',
           'c_spread_0_to_Tc': r'(c) 110 pts spread over $0$–$T_c$'}
for mk, tv in maps.items():
    a.plot(tv, y, 'o', ms=2.4, color=mapcols[mk], label=maplabs[mk])
tv = maps['c_spread_0_to_Tc']; kf = np.linspace(1e-3, k[-1], 400); tf = kf / k[-1]
a.plot(tf, 100 * (1 - Afree * kf ** nfree), '-', color='k', lw=1.2, label=fr'free power law, $n$={nfree:.2f}')
a.plot(tf, 100 * (1 - models['activated_swave']['params'][0] * np.exp(-models['activated_swave']['params'][1] / kf)), '--', color=C['red'],
       lw=1.2, label=fr'activated $A e^{{-\Delta/T}}$ ($R^2$={models["activated_swave"]["R2"]:.3f})')
ps_ = models['theory_s']['params']
a.plot(tf, ps_[1] * rho_interp['s'](kf * ps_[0]), '-.', color=C['red'], lw=1.2, label=fr'BCS s-wave curve, free $T$ scale ($R^2$={models["theory_s"]["R2"]:.3f})')
pa_ = models['fixed_n2.5']['params']
a.plot(tf, 100 * (1 - pa_[0] * kf ** 2.5), ':', color=C['orange'], lw=1.5, label=fr'forced $n$=2.5 ($R^2$={models["fixed_n2.5"]["R2"]:.3f})')
pd_ = models['theory_d']['params']
a.plot(tf, pd_[1] * rho_interp['d'](kf * pd_[0]), '-', color=C['teal'], lw=1, label=fr'clean d-wave curve, free $T$ scale ($R^2$={models["theory_d"]["R2"]:.3f})')
a.set_xlabel(r'$T/T_c$ (mapping-dependent; fit lines drawn for mapping (c))'); a.set_ylabel(r'$D_s$ (% of $D_{s0}$)')
a.set_title('Experimental $D_s(T)$ under three index$\\to$T mappings'); a.legend(frameon=False, loc='lower left', ncol=2)
a.set_xlim(0, 1.36); a.set_ylim(55, 101.5)
ins = a.inset_axes([0.70, 0.46, 0.28, 0.42])
tc_ = maps['c_spread_0_to_Tc'][1:]
ins.loglog(tc_, uu, 'o', ms=2, color=C['purple'])
for nn, ls in [(1, ':'), (2, '--'), (2.5, '-'), (3, '-.')]:
    ins.loglog(tc_, uu[9] * (tc_ / tc_[9]) ** nn, ls, color='k', lw=0.7)
ins.loglog(tc_, models['activated_swave']['params'][0] * np.exp(-models['activated_swave']['params'][1] / kk), '--', color=C['red'], lw=0.8)
ins.set_ylim(3e-3, 2); ins.set_xlim(7e-3, 1.6); ins.tick_params(labelsize=6)
ins.set_xlabel(r'$T/T_c$ (c)', fontsize=6, labelpad=0); ins.set_ylabel(r'$1-D_s/D_{s0}$', fontsize=6, labelpad=0)
ins.text(0.03, 0.97, 'log-log (mapping c)\nslopes 1 (dot), 2 (dash),\n2.5 (solid), 3 (dash-dot)\nred dashed: activated fit', transform=ins.transAxes, fontsize=5.3, va='top')
a = fig.add_subplot(gs[0, 2])
names = ['free_power', 'power_with_offset', 'exp_saturation', 'fixed_n1.0', 'theory_d', 'fixed_n2.0', 'fixed_n2.5', 'fixed_n3.0',
         'theory_aniso', 'activated_swave', 'activated_sqrt_prefactor', 'theory_s']
vals = [max(models[nm]['dBIC'], 0.1) for nm in names]
cols = [C['green'] if v < 10 else C['red'] for v in vals]
a.barh(range(len(names)), vals, color=cols); a.set_xscale('log'); a.set_yticks(range(len(names)))
a.set_yticklabels([nm.replace('_', ' ') + f'  (R²={models[nm]["R2"]:.3f})' for nm in names], fontsize=7)
a.axvline(10, color='k', ls='--', lw=0.8); a.invert_yaxis(); a.set_xlabel(r'$\Delta$BIC vs best (log)')
a.set_title('Model comparison (mapping-invariant)')
a = fig.add_subplot(gs[1, 0])
a.plot(run_k, run_n, '-', color=C['blue'], label='running local exponent (15-pt window)')
for kmx, v in win.items():
    a.errorbar(float(kmx), v['n'], yerr=v['n_err'], fmt='s', color=C['orange'], ms=4)
a.errorbar([], [], fmt='s', color=C['orange'], label=r'cumulative fit $k\leq k_{max}$')
a.axhspan(*exp['free_exponent']['boot_CI95'], color=C['grey'], alpha=0.3, label='block-bootstrap 95% CI (all data)')
for nn in [2, 2.5, 3]: a.axhline(nn, ls=':', color=C['red'], lw=0.8)
a.text(5, 2.55, 'n = 2, 2.5, 3 (target range)', fontsize=7, color=C['red'])
a.set_ylim(0, 3.2); a.set_xlabel('index k (∝ T under affine mapping)'); a.set_ylabel('power-law exponent n')
a.set_title('Exponent under affine mappings'); a.legend(frameon=False, loc='center right')
a = fig.add_subplot(gs[1, 1])
pr = np.linspace(0.2, 1.2, 200)
a.plot(pr, nfree / pr, color=C['blue'])
for nm_, v in nat.items():
    if 'n' in v: pass
for nt in [2, 2.5, 3]:
    a.axhline(nt, ls=':', color=C['red'], lw=0.8); a.plot(nfree / nt, nt, 'o', color=C['red'], ms=4)
a.axvline(1, color='k', lw=0.8, ls='--'); a.text(1.01, 0.3, 'affine', fontsize=7)
a.axvline(0.5, color=C['grey'], lw=0.8, ls='--'); a.text(0.51, 0.3, r'$\sqrt{k}$ grid', fontsize=7)
a.set_xlabel(r'grid power $p$ in $T_k\propto k^p$'); a.set_ylabel('implied exponent n'); a.set_ylim(0, 4)
a.set_title('Grid required for a given n');
a = fig.add_subplot(gs[1, 2])
res_best = yy - 100 * (1 - Afree * kk ** nfree)
a.plot(kk, res_best, '-', color=C['blue'], lw=0.8, label='free power law')
a.plot(kk, yy - models['theory_s']['params'][1] * rho_interp['s'](kk * models['theory_s']['params'][0]), '-', color=C['red'], lw=0.8, label='BCS s-wave curve')
a.plot(kk, yy - 100 * (1 - models['fixed_n2.5']['params'][0] * kk ** 2.5), '-', color=C['orange'], lw=0.8, label='forced n=2.5')
a.axhline(0, color='k', lw=0.5); a.set_xlabel('index k'); a.set_ylabel(r'residual (% $D_{s0}$)'); a.set_title('Fit residuals')
a.legend(frameon=False)
fig.tight_layout(); fig.savefig(os.path.join(IMG, 'fig4_temperature.png')); plt.close(fig)

# ---- figure: injection-recovery
fig, ax = plt.subplots(1, 3, figsize=(12, 3.5))
a = ax[0]
e = np.zeros(len(tg)); e[0] = rng.normal() * sig
for i in range(1, len(tg)): e[i] = phi * e[i - 1] + np.sqrt(1 - phi ** 2) * sig * rng.normal()
ys = 100 * (1 - tg ** 2.5) + e
pp, _ = curve_fit(lambda x, tc, nn: 100 * (1 - (x / tc) ** nn), tg, ys, p0=[1, 2])
a.plot(tg, ys, 'o', ms=2.5, color=C['grey'], label='synthetic replica')
a.plot(tg, 100 * (1 - (tg / pp[0]) ** pp[1]), color=C['blue'], label=f'free fit n={pp[1]:.2f}')
a.set_xlabel(r'$T/T_c$ (supplied grid)'); a.set_ylabel(r'$D_s$'); a.set_title('(a) SYNTHETIC: n=2.5 + AR(1) noise'); a.legend(frameon=False)
a = ax[1]
a.hist(rec_n, bins=30, color=C['blue'], alpha=0.7, label='AR(1) noise, full grid')
a.hist(rec_n_white, bins=30, color=C['orange'], alpha=0.6, label='white noise, full grid')
a.hist(rn5, bins=30, color=C['green'], alpha=0.5, label=r'white, $t\leq0.5$ only')
a.axvline(2.5, color='k', ls='--'); a.axvline(nfree, color=C['red'], ls='-'); a.text(nfree + 0.05, 5, 'data n', color=C['red'], fontsize=7)
a.set_xlabel('recovered n'); a.set_ylabel('count'); a.set_title('(b) SYNTHETIC: exponent recovery (200 reps)'); a.legend(frameon=False, fontsize=6.5)
a = ax[2]
a.hist(np.log10(np.clip(dB_act, 1, None)), bins=30, color=C['red'], alpha=0.7, label='activated − power')
a.hist(np.log10(np.clip(dB_n1, 1, None)), bins=30, color=C['purple'], alpha=0.6, label='n=1 − power')
a.axvline(1, color='k', ls='--'); a.set_xlabel(r'$\log_{10}\Delta$BIC'); a.set_title('(c) SYNTHETIC: rejection power'); a.legend(frameon=False)
fig.tight_layout(); fig.savefig(os.path.join(IMG, 'fig5_injection_recovery.png')); plt.close(fig)

# ======================================================================= 3. CURRENT
Ic = P['current_dependence.py']['I_c']; cur = {}
# templates
gl = D['D_s_gl']; lin = D['D_s_linear']; kg = np.arange(len(gl))
Iinv = Ic * np.sqrt(np.clip(1 - gl / 100, 0, None)); dev = np.abs(Iinv - kg * dI)
cur['template_audit'] = {'linear_max_dev_from_1-I/Ic_on_grid': float(np.max(np.abs(lin - 100 * np.clip(1 - D['I_dc'] / Ic, 0, None)))),
                         'gl_points_matching_grid_within_0.05nA': int(np.argmax(dev > 0.05)), 'gl_first_zero_index': int(np.argmax(gl <= 0)),
                         'gl_implied_step_late_nA': float(np.median(np.diff(Iinv[20:50])))}
yd = D['D_s_dc_exp']; kd = np.arange(len(yd), dtype=float); kmin = int(np.argmin(yd))
cur['dc_min'] = {'index': kmin, 'value': float(yd[kmin]), 'final': float(yd[-1])}
mono = kd < kmin; ud = 1 - yd / 100
dcr = {}
for kmx in [10, 15, 20, 30]:
    m = (kd >= 1) & (kd <= kmx)
    pp, cv = curve_fit(lambda x, a_, m_: 100 * (1 - a_ * x ** m_), kd[m], yd[m], p0=[1e-3, 2])
    dcr[f'free_exponent_k_le_{kmx}'] = {'m': float(pp[1]), 'err': float(np.sqrt(cv[1, 1]))}
assert 1.9 <= dcr['free_exponent_k_le_10']['m'] <= 2.1
for lab, sel in [('k_le_20', kd <= 20), ('monotonic_branch', mono)]:
    x = kd[sel]; yv = yd[sel]
    pq, _ = curve_fit(lambda x, D0, a2: D0 - a2 * x ** 2, x, yv, p0=[100, 0.04])
    pl, _ = curve_fit(lambda x, D0, a1: D0 - a1 * x, x, yv, p0=[100, 1])
    pq4, _ = curve_fit(lambda x, D0, a2, a4: D0 - a2 * x ** 2 - a4 * x ** 4, x, yv, p0=[100, 0.04, 0])
    dcr[f'quadratic_{lab}'] = {'D0': float(pq[0]), 'a2_per_k2': float(pq[1]), 'R2': r2(yv, pq[0] - pq[1] * x ** 2),
                               'RMS': rms(yv, pq[0] - pq[1] * x ** 2), 'BIC': bic(yv, pq[0] - pq[1] * x ** 2, 2)}
    dcr[f'linear_{lab}'] = {'D0': float(pl[0]), 'a1_per_k': float(pl[1]), 'R2': r2(yv, pl[0] - pl[1] * x),
                            'RMS': rms(yv, pl[0] - pl[1] * x), 'BIC': bic(yv, pl[0] - pl[1] * x, 2)}
    dcr[f'quartic_{lab}'] = {'D0': float(pq4[0]), 'a2': float(pq4[1]), 'a4': float(pq4[2]),
                             'R2': r2(yv, pq4[0] - pq4[1] * x ** 2 - pq4[2] * x ** 4), 'RMS': rms(yv, pq4[0] - pq4[1] * x ** 2 - pq4[2] * x ** 4)}
# leading coefficient (fixed D0=100) k<=10
m10 = (kd >= 1) & (kd <= 10)
a2_low = float(np.sum((100 - yd[m10]) * kd[m10] ** 2) / np.sum(kd[m10] ** 4))
dcr['a2_lowk_fixedD0'] = a2_low
steps = {'H1_supplied_step': dI, 'H2_85pts_over_0-60nA': 60 / 84}
dcr['Istar_from_lowk_quadratic'] = {h: float(np.sqrt(100 / a2_low) * s) for h, s in steps.items()}
assert abs(dcr['Istar_from_lowk_quadratic']['H1_supplied_step'] - Ic) < 3
# GL fits with free Ic under H1 (current in nA)
Ik = kd * dI
glf = {}
for lab, sel in [('k_le_20', kd <= 20), ('k_le_30', kd <= 30), ('monotonic_branch', mono)]:
    f = lambda I, D0, Icf: D0 * (1 - (I / Icf) ** 2)
    pp, cv = curve_fit(f, Ik[sel], yd[sel], p0=[100, 50])
    # bootstrap CI on Ic (residual blocks)
    res = yd[sel] - f(Ik[sel], *pp); bs = []
    Lb2 = 5
    for _ in range(500):
        idx = np.concatenate([np.arange(s, s + Lb2) for s in rng.integers(0, max(len(res) - Lb2, 1), size=len(res) // Lb2 + 1)])[:len(res)]
        try:
            q, _ = curve_fit(f, Ik[sel], f(Ik[sel], *pp) + res[idx], p0=pp); bs.append(q[1])
        except Exception:
            pass
    glf[lab] = {'D0': float(pp[0]), 'Ic_nA': float(pp[1]), 'Ic_err': float(np.sqrt(cv[1, 1])), 'Ic_boot_CI95': [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                'R2': r2(yd[sel], f(Ik[sel], *pp)), 'RMS': rms(yd[sel], f(Ik[sel], *pp))}
    fl = lambda I, D0, Icf: D0 * (1 - I / Icf)
    pl, _ = curve_fit(fl, Ik[sel], yd[sel], p0=[100, 50])
    glf[lab]['linear_law'] = {'Ic_nA': float(pl[1]), 'R2': r2(yd[sel], fl(Ik[sel], *pl)), 'RMS': rms(yd[sel], fl(Ik[sel], *pl))}
    fq = lambda I, D0, Icf, b4: D0 * (1 - (I / Icf) ** 2 - b4 * (I / Icf) ** 4)
    try:
        p4, _ = curve_fit(fq, Ik[sel], yd[sel], p0=[100, 50, 0]); glf[lab]['with_quartic'] = {'Ic_nA': float(p4[1]), 'b4': float(p4[2]), 'R2': r2(yd[sel], fq(Ik[sel], *p4))}
    except Exception:
        pass
    glf[lab]['fixed_Ic50_R2'] = r2(yd[sel], 100 * (1 - (Ik[sel] / Ic) ** 2))
dcr['GL_fits_H1'] = glf
# self-consistent GL depairing: D/D0 = 1-q^2, I/I_dep = (3*sqrt(3)/2) q (1-q^2)
dcr['selfconsistent_GL_note'] = {'low_current_coeff': 4 / 27, 'I_dep_if_selfconsistent_H1_nA': float(np.sqrt(4 / 27) * np.sqrt(100 / a2_low) * dI),
                                 'D_at_depairing_over_D0': 2 / 3}
cur['dc'] = dcr
# ---- microwave
Pm = D['P_mw']; Ia = D['I_mw_amplitude']; ym = D['D_s_mw_exp']
mw = {'Iamp_over_sqrtP': [float((Ia[1:] / np.sqrt(Pm[1:])).min()), float((Ia[1:] / np.sqrt(Pm[1:])).max())]}
fits = {}
specs = {'I2': (lambda x, c: 100 - c * x ** 2, [0.03], 'I'), 'I2_I4': (lambda x, c, c4: 100 - c * x ** 2 - c4 * x ** 4, [0.03, 0], 'I'),
         'absI': (lambda x, c: 100 - c * x, [0.5], 'I'), 'free_power_I': (lambda x, c, m_: 100 - c * x ** m_, [0.03, 2], 'I'),
         'P': (lambda x, c: 100 - c * x, [13], 'P'), 'P_P2': (lambda x, c, c2: 100 - c * x - c2 * x ** 2, [10, 1], 'P'),
         'free_power_P': (lambda x, c, m_: 100 - c * x ** m_, [13, 1], 'P')}
mwfit_par = {}
for nm, (f, p0, var) in specs.items():
    x = Ia if var == 'I' else Pm
    pp, cv = curve_fit(f, x, ym, p0=p0, maxfev=20000); yh = f(x, *pp); mwfit_par[nm] = pp
    fits[nm] = {'params': [float(q) for q in pp], 'perr': [float(q) for q in np.sqrt(np.diag(cv))], 'R2': r2(ym, yh), 'RMS': rms(ym, yh), 'BIC': bic(ym, yh, len(pp))}
mw['fits'] = fits
# low-amplitude leading coefficient (first 15 points, I2+I4)
m15 = np.arange(len(Ia)) < 15
pl15, _ = curve_fit(lambda x, c, c4: 100 - c * x ** 2 - c4 * x ** 4, Ia[m15], ym[m15], p0=[0.03, 0])
c_amp = float(mwfit_par['I2_I4'][0]); c_amp_low = float(pl15[0])
dc_coeff = a2_low / dI ** 2  # per nA^2 (DC, H1)
mw['leading_coeff_amp_basis_per_nA2'] = {'I2_I4_all': c_amp, 'I2_I4_first15': c_amp_low, 'I2_only': float(mwfit_par['I2'][0])}
mw['leading_coeff_rms_basis_per_nA2'] = {'I2_I4_all': 2 * c_amp, 'I2_I4_first15': 2 * c_amp_low}
mw['dc_coeff_per_nA2_H1'] = float(dc_coeff)
mw['rms_over_dc_ratio'] = {'I2_I4_all': float(2 * c_amp / dc_coeff), 'I2_I4_first15': float(2 * c_amp_low / dc_coeff)}
mw['Istar_amp_nA'] = float(np.sqrt(100 / c_amp)); mw['Istar_rms_nA'] = float(np.sqrt(100 / (2 * c_amp)))
mw['Istar_rms_nA_first15'] = float(np.sqrt(100 / (2 * c_amp_low)))
mw['max_suppression_percent'] = float(100 - ym[-1])
mw['Iamp_max_over_Ic'] = float(Ia[-1] / Ic)
# resonance frequency conversion
fr = {}
for g in [0.0, 1.0, 4.0]:  # L_g / L_k0
    fr[str(g)] = {'mw_df_over_f0_at_Pmax': float(np.sqrt((g + 1) / (g + 100 / ym[-1])) - 1),
                  'dc_df_over_f0_at_min': float(np.sqrt((g + 1) / (g + 100 / yd[kmin])) - 1)}
mw['resonance_shift'] = fr
cur['microwave'] = mw
R['current'] = cur


def fr_ratio(Dv, g):
    return np.sqrt((g + 1) / (g + 100 / Dv))


# ---- DC figure
fig, ax = plt.subplots(1, 3, figsize=(13, 3.9))
a = ax[0]
a.plot(Ik[mono], yd[mono], 'o', ms=3, color=C['blue'], label='DC data (monotonic branch, H1)')
a.plot(Ik[~mono], yd[~mono], 'x', ms=3.5, color=C['grey'], label=f'post-minimum upturn (k≥{kmin}), excluded')
a.axvspan(Ik[kmin], Ik[-1], color=C['grey'], alpha=0.12)
If = np.linspace(0, Ik[-1], 300)
g20 = glf['k_le_20']; gm = glf['monotonic_branch']
a.plot(If, np.clip(g20['D0'] * (1 - (If / g20['Ic_nA']) ** 2), 0, None), '-', color=C['red'], label=fr'GL $1-(I/I_c)^2$, fit $k\leq20$: $I_c$={g20["Ic_nA"]:.1f} nA ($R^2$={g20["R2"]:.4f})')
a.plot(If, np.clip(gm['D0'] * (1 - (If / gm['Ic_nA']) ** 2), 0, None), '--', color=C['orange'], label=fr'GL fit monotonic branch: $I_c$={gm["Ic_nA"]:.1f} nA ($R^2$={gm["R2"]:.3f})')
a.plot(D['I_dc'], lin, ':', color=C['purple'], label=r'linear Meissner template $1-I/I_c$')
a.plot(D['I_dc'][:len(gl)] if len(gl) <= 50 else kg * dI, gl, '-', color=C['grey'], lw=0.7, label='GL template (H1 index)')
a.set_xlabel(r'$I_{dc}$ (nA), mapping H1: $I_k=k\,\Delta I$'); a.set_ylabel(r'$D_s$ (% of $D_{s0}$)'); a.set_ylim(-3, 108)
a.set_title('(a) DC current suppression'); a.legend(frameon=False, fontsize=6.5, loc='upper center', bbox_to_anchor=(0.5, -0.16), ncol=2)
a = ax[1]
m = (kd >= 1) & mono
a.loglog(Ik[m], ud[m], 'o', ms=3, color=C['blue'], label='DC data')
a.loglog(Ik[m], (Ik[m] / Ic) ** 2, '-', color=C['red'], label=r'$(I/50\,{\rm nA})^2$')
a.loglog(Ik[m], Ik[m] / Ic, ':', color=C['purple'], label=r'$I/50\,{\rm nA}$ (linear)')
a.set_xticks([2, 5, 10, 20, 50]); a.set_xticklabels(['2', '5', '10', '20', '50']); a.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
a.set_xlabel(r'$I_{dc}$ (nA, H1)'); a.set_ylabel(r'$1-D_s/D_{s0}$'); a.set_title(fr'(b) log-log: slope {dcr["free_exponent_k_le_10"]["m"]:.2f} for $k\leq10$')
a.legend(frameon=False)
a = ax[2]
x = kd[kd <= 20]; yv = yd[kd <= 20]
q = dcr['quadratic_k_le_20']; l = dcr['linear_k_le_20']
a.plot(Ik[kd <= 20], yv - (q['D0'] - q['a2_per_k2'] * x ** 2), 'o-', ms=3, color=C['red'], label=fr'quadratic: $R^2$={q["R2"]:.5f}, RMS={q["RMS"]:.3f}')
a.plot(Ik[kd <= 20], yv - (l['D0'] - l['a1_per_k'] * x), 's-', ms=3, color=C['purple'], label=fr'linear: $R^2$={l["R2"]:.4f}, RMS={l["RMS"]:.2f}')
a.axhline(0, color='k', lw=0.5); a.set_xlabel(r'$I_{dc}$ (nA, H1)'); a.set_ylabel('residual (% $D_{s0}$)')
a.set_title(r'(c) residuals, $k\leq20$ (free $D_0$)'); a.legend(frameon=False)
fig.tight_layout(); fig.savefig(os.path.join(IMG, 'fig6_dc_current.png')); plt.close(fig)

# ---- microwave figure
fig, ax = plt.subplots(2, 2, figsize=(11.5, 7.5))
a = ax[0, 0]
a.plot(Pm, ym, 'o', ms=3, color=C['blue'], label='MW data')
for nm, col, ls in [('P', C['purple'], ':'), ('I2', C['red'], '--'), ('I2_I4', C['orange'], '-'), ('absI', C['grey'], '-.')]:
    f, _, var = specs[nm]; x = Ia if var == 'I' else Pm
    a.plot(Pm, f(x, *mwfit_par[nm]), ls, color=col, label=f'{nm.replace("_", "+")}: $R^2$={fits[nm]["R2"]:.4f}')
a.set_xlabel(r'$P_{mw}$ (normalized)'); a.set_ylabel(r'$D_s$ (% of $D_{s0}$)'); a.set_title('(a) MW probe-power dependence'); a.legend(frameon=False)
a = ax[0, 1]
for g, col in [(0.0, C['blue']), (1.0, C['orange']), (4.0, C['green'])]:
    a.plot(Pm, 100 * (fr_ratio(ym, g) - 1), 'o-', ms=2, color=col, label=fr'$L_g/L_{{k0}}$={g:g}' + (' (kinetic limit, $f\\propto\\sqrt{D_s}$)' if g == 0 else ''))
a.set_xlabel(r'$P_{mw}$ (normalized)'); a.set_ylabel(r'$\Delta f_r/f_{r0}$ (%)'); a.set_title('(b) Implied resonance-frequency shift vs MW power'); a.legend(frameon=False)
a = ax[1, 0]
a.plot((Ik[mono]) ** 2, 100 - yd[mono], 'o', ms=3, color=C['blue'], label=r'DC (H1): $I_{dc}^2$')
a.plot(Ia ** 2 / 2, 100 - ym, 's', ms=3, color=C['orange'], label=r'MW: $\langle I^2\rangle=I_{amp}^2/2$')
xx = np.linspace(0, 400, 50); a.plot(xx, dc_coeff * xx, '-', color=C['red'], lw=1, label=fr'DC low-$I$ slope ({dc_coeff:.4f}/nA$^2$)')
ia = np.linspace(0, Ia[-1], 100); a.plot(ia ** 2 / 2, mwfit_par['I2_I4'][0] * ia ** 2 + mwfit_par['I2_I4'][1] * ia ** 4, '--', color=C['orange'], lw=1, label=r'MW $I^2+I^4$ fit')
a.set_xlim(0, 400); a.set_ylim(0, 16)
a.set_xlabel(r'$\langle I^2\rangle$ (nA$^2$)'); a.set_ylabel(r'$100-D_s$'); a.set_title(r'(c) DC vs MW on common $\langle I^2\rangle$ axis'); a.legend(frameon=False)
a = ax[1, 1]
for g, col in [(0.0, C['blue']), (1.0, C['orange']), (4.0, C['green'])]:
    a.plot(Ik[mono], 100 * (fr_ratio(yd[mono], g) - 1), 'o-', ms=2, color=col, label=fr'$L_g/L_{{k0}}$={g:g}')
a.set_xlabel(r'$I_{dc}$ (nA, H1)'); a.set_ylabel(r'$\Delta f_r/f_{r0}$ (%)'); a.set_title('(d) Implied resonance shift vs DC bias'); a.legend(frameon=False)
fig.tight_layout(); fig.savefig(os.path.join(IMG, 'fig7_microwave.png')); plt.close(fig)

# ---- figure 1: data overview
fig, ax = plt.subplots(2, 3, figsize=(13, 6.5))
a = ax[0, 0]
for key, col in [('D_s_conv', C['grey']), ('D_s_geom', C['green']), ('D_s_exp_hole', C['red']), ('D_s_exp_electron', C['blue'])]:
    a.plot(n15, D[key], 'o-', ms=2, color=col, label=f'{key} ({len(D[key])})')
a.set_yscale('log'); a.set_xlabel(r'$n_{\rm eff}$ ($10^{15}$ m$^{-2}$) [50]'); a.set_title('File 1: density arrays'); a.legend(frameon=False)
a = ax[0, 1]
for key, col in [('D_s_bcs', C['red']), ('D_s_nodal', C['blue']), ('D_s_power_n2_5', C['orange']), ('D_s_power_n3', C['purple']), ('D_s_experimental', 'k')]:
    a.plot(np.arange(len(D[key])), D[key], '-', color=col, label=f'{key} ({len(D[key])})')
a.axvline(len(T) - 1, color=C['grey'], ls=':'); a.text(len(T) - 1, 50, ' end of T grid (100)', fontsize=7)
a.set_xlabel('array index'); a.set_title('File 2: temperature arrays vs index'); a.legend(frameon=False, fontsize=6.5)
a = ax[0, 2]
for key, col in [('D_s_gl', C['red']), ('D_s_linear', C['purple']), ('D_s_dc_exp', C['blue'])]:
    a.plot(np.arange(len(D[key])), D[key], '-', color=col, label=f'{key} ({len(D[key])})')
a.axvline(len(D['I_dc']) - 1, color=C['grey'], ls=':'); a.text(len(D['I_dc']) - 1, 60, ' end of I_dc grid (50)', fontsize=7)
a.set_xlabel('array index'); a.set_title('File 3: DC arrays vs index'); a.legend(frameon=False)
a = ax[1, 0]
a.plot(Pm, Ia, 'o', ms=2, color=C['blue'], label='I_mw_amplitude (50)'); a.plot(Pm, 20 * np.sqrt(Pm), '-', color=C['red'], lw=0.8, label=r'$20\sqrt{P}$')
a.set_xlabel('P_mw (50)'); a.set_ylabel('nA'); a.set_title('MW calibration'); a.legend(frameon=False)
a = ax[1, 1]
a.plot(Pm, ym, 'o', ms=2, color=C['orange'], label='D_s_mw_exp (50)'); a.set_xlabel('P_mw'); a.set_title('MW stiffness'); a.legend(frameon=False)
a = ax[1, 2]; a.axis('off')
rows = [[k_, str(v['len']), v['coord'], str(v['coord_len']), 'yes' if v['matched'] else 'NO'] for k_, v in R['inventory'].items()]
tb = a.table(cellText=rows, colLabels=['array', 'len', 'coord', 'coord len', 'match'], loc='center', cellLoc='left', colWidths=[0.42, 0.12, 0.2, 0.16, 0.12])
tb.auto_set_font_size(False); tb.set_fontsize(6.5); tb.scale(1, 1.05)
for (i, j), cell in tb.get_celld().items():
    if i > 0 and rows[i - 1][4] == 'NO': cell.set_facecolor('#fde2e2')
a.set_title('Array-length inventory', pad=30)
fig.tight_layout(); fig.savefig(os.path.join(IMG, 'fig1_data_overview.png')); plt.close(fig)


def clean(o):
    if isinstance(o, dict): return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [clean(v) for v in o]
    if isinstance(o, (np.floating,)): return float(o)
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, np.bool_): return bool(o)
    return o


json.dump(clean(R), open(os.path.join(OUT, 'results.json'), 'w'), indent=1)
print(json.dumps(clean({'density_hole': R['density']['hole'], 'temp_models': {k: (v.get('R2'), v.get('dBIC'), v.get('params')) for k, v in models.items()},
                        'free_exp': exp['free_exponent'], 'win': win, 'nat': nat, 'inj': inj, 'templ': aud, 'theory': th_res,
                        'dc': {k: v for k, v in dcr.items()}, 'mw': mw, 'maps': exp['mappings'], 'gap': exp['activated_gap_over_Tc_under_mapping'],
                        'cur_audit': cur['template_audit']}), indent=1))
