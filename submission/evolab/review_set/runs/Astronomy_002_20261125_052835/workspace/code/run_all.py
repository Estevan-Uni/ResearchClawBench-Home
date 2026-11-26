"""Regenerate every table (outputs/) and figure (report/images/) of the report.
Usage:  python code/run_all.py          (fixed seeds; ~1-2 min)
"""
import os, json, sys
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from h0dn_core import load_dataset, fit, clip_fit, explicit_nuisance, System, gls, par, fz, C_KMS, LN10

ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, 'outputs'); IMG = os.path.join(ROOT, 'report', 'images')
os.makedirs(OUT, exist_ok=True); os.makedirs(IMG, exist_ok=True)
D = load_dataset(os.path.join(ROOT, 'data', 'H0DN_MinimalDataset.txt'))

# palette (validated categorical default, fixed order)
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, GREY, INK = '#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#8a8a85', '#2b2b2a'
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.grid': True, 'grid.color': '#e6e6e3', 'grid.linewidth': 0.6,
                     'axes.edgecolor': '#8a8a85', 'savefig.dpi': 150, 'savefig.bbox': 'tight'})
MSTYLE = {('Cepheid', 'N4258'): (BLUE, 'o', 'Cepheid / N4258'),
          ('Cepheid', 'LMC'): (ORANGE, 's', 'Cepheid / LMC'),
          ('TRGB', 'N4258'): (AQUA, 'D', 'TRGB / N4258')}

# literature values (quoted, not computed from the file)
LIT = dict(R22_aB=(0.714158, 0.0018),          # R22 Table 5 fit 1; sigma ~ R16 value (assumed)
           R22_MB=(-19.253, 0.027), R22_H0=(73.04, 1.04),
           H0DN_H0=(73.50, 0.81), Planck_H0=(67.4, 0.5), B24_H0=(73.17, 0.86))
log = {}
def save(name, obj):
    with open(os.path.join(OUT, name), 'w') as f:
        json.dump(obj, f, indent=1, default=float)

def summ(R, label):
    s = R['sys']
    out = dict(label=label, H0=R['H0'], sH0=R['sH0'], pct=100 * R['sH0'] / R['H0'],
               chi2=R['chi2'], dof=R['dof'], n=s.n, p=s.p, pval=R['pval'])
    for k in ['5logH0', 'M_SN', 'M_SBF', 'mu_Fornax']:
        if k in R['P']:
            out[k], out['s' + k] = par(R, k)
    out['chi2nu'] = R['chi2'] / R['dof'] if R['dof'] > 0 else np.nan
    return out

# ============================================================ 1. sanity arithmetic
print('== 1. sanity arithmetic')
san = {'hosts': [], 'sn': [], 'hf': []}
prim = [r for r in D['host_measurements'] if r[2] in ('N4258', 'LMC')]
wmu = {}
for h in dict.fromkeys(r[0] for r in prim):
    rr = [r for r in prim if r[0] == h]
    w = np.array([1 / r[4] ** 2 for r in rr]); m = np.array([r[3] for r in rr])
    wmu[h] = (float((w * m).sum() / w.sum()), float(1 / np.sqrt(w.sum())))
    san['hosts'].append(dict(host=h, n=len(rr), mu=wmu[h][0], err=wmu[h][1], methods='+'.join(f'{r[1]}/{r[2]}' for r in rr)))
for h, mb, e in D['sneia_calibrators']:
    san['sn'].append(dict(host=h, mB=mb, MB=mb - wmu[h][0], err=float(np.hypot(e, wmu[h][1]))))
muF_trgb = [r[3] for r in prim if r[0] == 'NGC1316'][0]
MSBF_quick = np.mean([m for h, m, e in D['sbf_calibrators'] if D['host_group'][h] == 'Fornax']) - muF_trgb
for lab, rows, M in [('SN', D['hubble_flow_sneia'], -19.25), ('SBF', D['hubble_flow_sbf'], MSBF_quick)]:
    for z, m, e, v in rows:
        mu = m - M; dL = 10 ** ((mu - 25) / 5)
        H0 = C_KMS * z * fz(z) / dL
        san['hf'].append(dict(sample=lab, z=z, m=m, M_assumed=M, H0_implied=H0,
                              dm_for_73=5 * np.log10(H0 / 73.0)))
save('sanity.json', san)
for r in san['sn']: print('  SN cal %-8s M_B=%.3f +- %.3f' % (r['host'], r['MB'], r['err']))
for r in san['hf']: print('  HF %-3s z=%.3f M=%.2f -> H0=%.1f  (needs %+.2f mag)' % (r['sample'], r['z'], r['M_assumed'], r['H0_implied'], r['dm_for_73']))

# ============================================================ 2. compact system / accounting
print('== 2. system')
BASE = dict()                                   # defaults = baseline configuration
R_all = fit(D, **BASE)                          # no outlier rejection
R_base = clip_fit(D, **BASE)                    # R22-style global 3.3 sigma clip -> baseline (file-internal)
OPT_B = R_base['clip_opt']
HYB = dict(hf_mode='none', ext_aB=LIT['R22_aB'])
R_hyb_all = fit(D, **HYB)
R_hyb = clip_fit(D, **HYB)
OPT_H = R_hyb['clip_opt']
print('  clipped (baseline):', R_base['clipped'])
E_base = explicit_nuisance(D, **OPT_B)
E_hyb = explicit_nuisance(D, **OPT_H)
acc = []
for lab, R, E in [('file-internal baseline', R_base, E_base), ('hybrid (file ladder + R22 a_B)', R_hyb, E_hyb)]:
    s = R['sys']; t = s.types()
    rows = {k: int((t == k).sum()) for k in dict.fromkeys(t)}
    pars = {'host mu': sum(p.startswith('mu_') and p != 'mu_Fornax' and p != 'mu_Virgo' for p in s.params),
            'group mu': sum(p in ('mu_Fornax', 'mu_Virgo') for p in s.params),
            'M_SN': int('M_SN' in s.params), 'M_SBF': int('M_SBF' in s.params), '5logH0': int('5logH0' in s.params)}
    dq = np.max(np.abs(E['q'][:s.p] - R['q']))
    acc.append(dict(label=lab, rows=rows, N=s.n, params=pars, P=s.p, dof=R['dof'], rank=int(R['rank']),
                    N_ext=E['n'], P_ext=E['p'], dof_ext=E['dof'],
                    H0=R['H0'], H0_ext=E['H0'], chi2=R['chi2'], chi2_ext=E['chi2'], max_dq=dq,
                    dsig=abs(np.sqrt(E['cov'][s.p - 1, s.p - 1]) - np.sqrt(R['cov'][s.p - 1, s.p - 1]))))
    print('  %s: N=%d P=%d dof=%d rank=%d | explicit N=%d P=%d dof=%d | dH0=%.1e dchi2=%.1e max|dq|=%.1e'
          % (lab, s.n, s.p, R['dof'], R['rank'], E['n'], E['p'], E['dof'], R['H0'] - E['H0'], R['chi2'] - E['chi2'], dq))
save('system_accounting.json', acc)
# full design listing for baseline
with open(os.path.join(OUT, 'baseline_design.txt'), 'w') as f:
    s = R_base['sys']
    f.write('params: ' + ', '.join(s.params) + '\n')
    for k, i in enumerate(s.info):
        coefs = ' '.join('%+g*%s' % (c, s.params[j]) for j, c in enumerate(s.L[k]) if c != 0)
        f.write('%2d %-8s %-8s %-8s %-6s y=%9.4f sig=%.4f  model= %s\n' % (k, i['type'], i.get('host', ''), i.get('method', ''), i.get('anchor', ''), s.y[k], s.sig[k], coefs))
    f.write('clipped: %s\n' % R_base['clipped'])

heads = [summ(R_all, 'file-internal, no clipping'), summ(R_base, 'file-internal baseline (3.3σ clip)'),
         summ(R_hyb_all, 'hybrid: file ladder + R22 a_B, no clipping'), summ(R_hyb, 'hybrid: file ladder + R22 a_B (3.3σ clip)')]
# hybrid with SN intrinsic scatter (sigma_int chosen so calibrator chi2/dof = 1)
def chi2nu_sigint(si):
    R = fit(D, **dict(OPT_H, sn_sigint=si)); return R['chi2'] / R['dof'] - 1
from scipy.optimize import brentq
SIGINT = brentq(chi2nu_sigint, 0.0, 0.5)
R_hyb_si = fit(D, **dict(OPT_H, sn_sigint=SIGINT))
heads.append(summ(R_hyb_si, 'hybrid + SN σ_int=%.3f (χ²/ν=1)' % SIGINT))
R_base_si = fit(D, **dict(OPT_B, sn_sigint=SIGINT))
heads.append(summ(R_base_si, 'file-internal baseline + SN σ_int=%.3f' % SIGINT))
save('headline.json', heads)
for h in heads: print('  %-48s H0=%.2f±%.2f (%.2f%%) M_SN=%.3f M_SBF=%s chi2=%.1f/%d' % (h['label'], h['H0'], h['sH0'], h['pct'], h['M_SN'], '%.3f' % h['M_SBF'] if 'M_SBF' in h else '-', h['chi2'], h['dof']))

# ============================================================ 3. rung-1/2 forward host fit + residual figure
print('== 3. host residuals')
S0 = System(D, hf_mode='none', use_sbf=False)
ip = [k for k, i in enumerate(S0.info) if i['type'] == 'primary']
L1 = S0.L[np.ix_(ip, list(range(len(S0.hosts))))]; y1 = S0.y[ip]; C1 = S0.C[np.ix_(ip, ip)]
F = gls(L1, y1, C1)
info1 = [S0.info[k] for k in ip]
mu_fwd = {h: (F['q'][j], np.sqrt(F['cov'][j, j])) for j, h in enumerate(S0.hosts)}
res_tab = []
def joint_mu(R, h):
    return R['q'][R['P']['mu_' + h]]
for k, i in enumerate(info1):
    err = np.sqrt(C1[k, k])
    res_tab.append(dict(host=i['host'], method=i['method'], anchor=i['anchor'], mu=y1[k], err_tot=err,
                        res_fwd=y1[k] - F['q'][S0.hosts.index(i['host'])],
                        res_joint_all=y1[k] - joint_mu(R_all, i['host']),
                        res_joint_base=y1[k] - joint_mu(R_base, i['host'])))
save('host_residuals.json', dict(forward_chi2=F['chi2'], forward_dof=len(y1) - len(S0.hosts), rows=res_tab, mu_fwd=mu_fwd))
inband = np.mean([abs(r['res_fwd']) <= 0.1 for r in res_tab])
print('  forward chi2=%.2f/%d, frac |res|<=0.1 = %.2f' % (F['chi2'], len(y1) - len(S0.hosts), inband))
xc = {}
for h in ['NGC1365', 'M101']:
    rr = {(r['method'], r['anchor']): r for r in res_tab if r['host'] == h}
    xc[h] = dict(cep_minus_trgb=rr[('Cepheid', 'N4258')]['mu'] - rr[('TRGB', 'N4258')]['mu'],
                 err=np.hypot(rr[('Cepheid', 'N4258')]['err_tot'], rr[('TRGB', 'N4258')]['err_tot']),
                 res={'%s/%s' % k: v['res_fwd'] for k, v in rr.items()})
    print('  %s Cep-TRGB = %+.3f +- %.3f, residuals %s' % (h, xc[h]['cep_minus_trgb'], xc[h]['err'], xc[h]['res']))
save('cep_trgb_crosscheck.json', xc)

hosts_order = S0.hosts
fig, axes = plt.subplots(1, 3, figsize=(14, 4.6), sharey=True)
for ax, key, title in zip(axes, ['res_fwd', 'res_joint_base', 'res_joint_all'],
                          ['(a) Rung-1/2 forward fit (primaries + anchors)', '(b) Joint fit, baseline (NGC1309 SN clipped)', '(c) Joint fit, no clipping']):
    ax.axhspan(-0.1, 0.1, color='#eef3fb', zorder=0); ax.axhline(0, color=GREY, lw=1)
    for lab in ['Cepheid / N4258', 'Cepheid / LMC', 'TRGB / N4258']:
        pass
    seen = set()
    for r in res_tab:
        col, mk, lab = MSTYLE[(r['method'], r['anchor'])]
        off = {'Cepheid / N4258': -0.18, 'Cepheid / LMC': 0.0, 'TRGB / N4258': 0.18}[lab]
        x = hosts_order.index(r['host']) + off
        ax.errorbar(x, r[key], yerr=r['err_tot'], fmt=mk, color=col, ms=7, mec='white', mew=1.2, elinewidth=1.6, capsize=0,
                    label=lab if lab not in seen else None)
        seen.add(lab)
    ax.set_xticks(range(len(hosts_order))); ax.set_xticklabels(hosts_order, rotation=40, ha='right')
    ax.set_title(title, fontsize=10, loc='left')
    ax.set_ylim(-0.75, 0.75)
axes[0].set_ylabel(r'$\mu_{\rm meas}-\mu_{\rm fit}$ [mag]')
axes[0].legend(loc='lower left', frameon=False, fontsize=9)
axes[0].text(0.02, 0.95, '±0.1 mag band; %d/%d inside' % (sum(abs(r['res_fwd']) <= 0.1 for r in res_tab), len(res_tab)),
             transform=axes[0].transAxes, fontsize=8.5, color=INK, va='top')
fig.suptitle('Host-distance residuals per primary measurement (error bars include anchor and method–anchor terms)', fontsize=11)
fig.savefig(os.path.join(IMG, 'fig3_host_residuals.png')); plt.close(fig)

# ============================================================ 4. data overview: topology + calibrator absolute mags
print('== 4. overview figure')
fig = plt.figure(figsize=(14, 5.2))
ax = fig.add_axes([0.0, 0.0, 0.52, 1.0]); ax.axis('off')
col_x = {'anchor': 0.06, 'host': 0.36, 'sec': 0.66, 'hf': 0.93}
anchors_pos = {'N4258': 0.70, 'LMC': 0.35, 'MW': 0.08}
hosts_pos = {h: 0.92 - 0.11 * j for j, h in enumerate(hosts_order)}
fornax_pos = 0.18
for a, yv in anchors_pos.items():
    ax.text(col_x['anchor'], yv, a + ('\n(no data, σ=0)' if a == 'MW' else '\nμ=%.3f±%.3f' % (D['anchors'][a]['mu'], D['anchors'][a]['err'])),
            ha='center', va='center', fontsize=8.5, color=GREY if a == 'MW' else INK,
            bbox=dict(boxstyle='round,pad=0.35', fc='white', ec=GREY if a == 'MW' else INK, lw=1))
for h, yv in hosts_pos.items():
    ax.text(col_x['host'], yv, h, ha='center', va='center', fontsize=8.5,
            bbox=dict(boxstyle='round,pad=0.3', fc='#f4f4f2', ec=INK, lw=0.8))
for h, m, a, mu, e in D['host_measurements']:
    col, mk, lab = MSTYLE[(m, a)]
    ax.annotate('', xy=(col_x['host'] - 0.07, hosts_pos[h]), xytext=(col_x['anchor'] + 0.095, anchors_pos[a]),
                arrowprops=dict(arrowstyle='-', color=col, lw=1.5, alpha=0.85))
ax.text(col_x['sec'], 0.62, 'SN Ia calibrators\n(7 hosts)  →  M_SN', ha='center', va='center', fontsize=8.5, bbox=dict(boxstyle='round', fc='white', ec=BLUE))
for h in [r[0] for r in D['sneia_calibrators']]:
    ax.annotate('', xy=(col_x['sec'] - 0.11, 0.62), xytext=(col_x['host'] + 0.07, hosts_pos[h]), arrowprops=dict(arrowstyle='-', color='#b9b9b4', lw=0.8))
ax.text(col_x['sec'], fornax_pos, 'Fornax group μ_F\nSBF: NGC1399, NGC1404\n→  M_SBF', ha='center', va='center', fontsize=8.5, bbox=dict(boxstyle='round', fc='white', ec=MAGENTA))
ax.annotate('', xy=(col_x['sec'] - 0.11, fornax_pos), xytext=(col_x['host'] + 0.07, hosts_pos['NGC1316']),
            arrowprops=dict(arrowstyle='-|>', color=MAGENTA, lw=1.6))
ax.text(0.40, 0.10, 'TRGB route: NGC1316 (Fornax A) → group, ±0.10 depth', fontsize=7.5, color=MAGENTA, ha='left')
ax.text(col_x['sec'], 0.03, 'NGC4472 (Virgo): no anchored path → excluded', fontsize=7.5, color=GREY, ha='center')
ax.text(col_x['hf'], 0.62, 'HF SNe Ia\n5, z=0.034–0.082', ha='center', va='center', fontsize=8.5, bbox=dict(boxstyle='round', fc='#eef3fb', ec=BLUE))
ax.text(col_x['hf'], fornax_pos, 'HF SBF\n3, z=0.023–0.045', ha='center', va='center', fontsize=8.5, bbox=dict(boxstyle='round', fc='#fcf0f5', ec=MAGENTA))
ax.annotate('', xy=(col_x['hf'] - 0.08, 0.62), xytext=(col_x['sec'] + 0.11, 0.62), arrowprops=dict(arrowstyle='-|>', color=BLUE))
ax.annotate('', xy=(col_x['hf'] - 0.08, fornax_pos), xytext=(col_x['sec'] + 0.11, fornax_pos), arrowprops=dict(arrowstyle='-|>', color=MAGENTA))
for k, t in zip(['anchor', 'host', 'sec', 'hf'], ['Rung 1: anchors', 'Rung 2: primary hosts', 'Rung 2→3: secondaries', 'Rung 3: Hubble flow']):
    ax.text(col_x[k], 1.0, t, ha='center', fontsize=9.5, weight='bold', color=INK)
for (m, a), (col, mk, lab) in MSTYLE.items():
    pass
ax.legend(handles=[plt.Line2D([], [], color=c, lw=2, label=l) for c, mk, l in MSTYLE.values()], loc='lower left', frameon=False, fontsize=8, bbox_to_anchor=(0.0, 0.12))
ax2 = fig.add_axes([0.60, 0.13, 0.38, 0.78])
sn = san['sn']
for j, r in enumerate(sn):
    mu, smu = mu_fwd[r['host']]
    MB = [x[1] for x in D['sneia_calibrators'] if x[0] == r['host']][0] - mu
    e = np.hypot([x[2] for x in D['sneia_calibrators'] if x[0] == r['host']][0], smu)
    ax2.errorbar(MB, j, xerr=e, fmt='o', color=ORANGE if r['host'] == 'NGC1309' else BLUE, ms=8, mec='white', mew=1.2, elinewidth=1.8)
    ax2.text(MB + (0.06 if r['host'] != 'NGC1309' else 0.06), j + 0.22, '%.2f' % MB, fontsize=8, color=INK)
ax2.axvline(LIT['R22_MB'][0], color=GREY, ls='--', lw=1.2, label='R22 M_B = −19.253 (quoted)')
Mh = par(R_hyb, 'M_SN')
ax2.axvspan(Mh[0] - Mh[1], Mh[0] + Mh[1], color=BLUE, alpha=0.15, label='this work, NGC1309 clipped: %.3f±%.3f' % Mh)
ax2.legend(loc='upper right', frameon=True, fontsize=8)
ax2.set_yticks(range(len(sn))); ax2.set_yticklabels([r['host'] for r in sn]); ax2.set_ylim(-0.6, len(sn) + 0.6)
ax2.set_xlabel(r'$M_B = m_B - \mu_{\rm host}$ (rung-1/2 forward distance) [mag]'); ax2.invert_xaxis()
ax2.set_title('SN Ia calibrator absolute magnitudes', fontsize=10, loc='left')
fig.savefig(os.path.join(IMG, 'fig1_data_overview.png')); plt.close(fig)

# ============================================================ 5. Hubble-flow convention audit
print('== 5. HF audit')
def V(label, R, group):
    s = summ(R, label); s['group'] = group; return s
var = []
var.append(V('Baseline (rows, f(z) q0=−0.55, σv=250, NGC1309 SN clipped)', R_base, 'baseline'))
var.append(V('f(z) off (pure Hubble law)', fit(D, **dict(OPT_B, use_fz=False)), 'HF convention'))
var.append(V('q0 = −0.51', fit(D, **dict(OPT_B, q0=-0.51)), 'HF convention'))
var.append(V('Aggregated a_B intercept per sample (R22 Eq. 4–5)', fit(D, **dict(OPT_B, hf_mode='aggregate')), 'HF convention'))
var.append(V('HF SN only', fit(D, **dict(OPT_B, hf_samples=('SN',))), 'HF sample'))
var.append(V('HF SBF only', fit(D, **dict(OPT_B, hf_samples=('SBF',))), 'HF sample'))
var.append(V('HF z < 0.05', fit(D, **dict(OPT_B, zmax=0.05)), 'HF sample'))
var.append(V('HF z ≥ 0.05 (SN only)', fit(D, **dict(OPT_B, zmin=0.05)), 'HF sample'))
R_slope = fit(D, **dict(OPT_B, free_slope=True))
var.append(V('Free HF slope (β fitted)', R_slope, 'HF convention'))
for sv in [150, 350]:
    var.append(V('σv = %d km/s' % sv, fit(D, **dict(OPT_B, sigv=sv)), 'HF convention'))
var.append(V('No outlier rejection (NGC1309 kept)', R_all, 'calibration'))
var.append(V('NGC1365 also linked to Fornax', fit(D, **dict(OPT_B, fornax_links=('NGC1316', 'NGC1365'))), 'calibration'))
var.append(V('NGC4472 kept with free μ_Virgo', fit(D, **dict(OPT_B, keep_virgo=True)), 'calibration'))
var.append(V('No SBF at all (Cepheid/TRGB→SN only)', fit(D, **dict(OPT_B, use_sbf=False)), 'calibration'))
var.append(V('Cepheids only (TRGB rows dropped)', fit(D, **dict(OPT_B, drop_primary=tuple(OPT_B.get('drop_primary', ())) + tuple(tuple(r[:3]) for r in D['host_measurements'] if r[1] == 'TRGB'), drop_hosts=tuple(OPT_B.get('drop_hosts', ())) + ('NGC5643',), fornax_links=(), use_sbf=False)), 'calibration'))
var.append(V('N4258 anchor only', fit(D, **dict(OPT_B, drop_primary=tuple(tuple(r[:3]) for r in D['host_measurements'] if r[2] == 'LMC'))), 'calibration'))
var.append(V('SN σ_int = %.3f' % SIGINT, R_base_si, 'calibration'))
Rb = fit(D, **dict(OPT_B, extra_err=0.0)); birge = np.sqrt(Rb['chi2'] / Rb['dof'])
vb = V('Birge rescaling (σ×%.2f)' % birge, Rb, 'calibration'); vb['sH0'] *= birge; vb['pct'] *= birge; vb['sM_SN'] *= birge
var.append(vb)
# offsets for H0 = 73
H0t = 73.0
delta_all = 5 * np.log10(R_base['H0'] / H0t)
Rsn = fit(D, **dict(OPT_B, hf_samples=('SN',))); Rsbf = fit(D, **dict(OPT_B, hf_samples=('SBF',)))
delta_sn = 5 * np.log10(Rsn['H0'] / H0t); delta_sbf = 5 * np.log10(Rsbf['H0'] / H0t)
R_off = fit(D, **dict(OPT_B, hf_offset=delta_all))
vo = V('Hypothetical: all HF mags +%.3f mag (NOT adopted)' % delta_all, R_off, 'hypothetical'); var.append(vo)
var.append(V('Hybrid: file ladder + R22 a_B', R_hyb, 'hybrid'))
var.append(V('Hybrid + SN σ_int', R_hyb_si, 'hybrid'))
var.append(V('Hybrid, no clipping', R_hyb_all, 'hybrid'))
save('variants.json', dict(variants=var, delta_all=delta_all, delta_sn=delta_sn, delta_sbf=delta_sbf, beta=par(R_slope, 'beta_HF'), birge=birge, sigint=SIGINT))
for v in var: print('  %-62s H0=%7.2f±%5.2f M_SN=%.3f chi2=%6.1f/%d' % (v['label'], v['H0'], v['sH0'], v['M_SN'], v['chi2'], v['dof']))
print('  offsets for H0=73: all %.3f  SN %.3f  SBF %.3f ; beta=%.3f±%.3f' % (delta_all, delta_sn, delta_sbf, *par(R_slope, 'beta_HF')))

# leave-one-out on the hybrid
loo = []
for h in [r[0] for r in D['sneia_calibrators'] if r[0] not in OPT_H.get('drop_hosts', ())]:
    R = fit(D, **dict(OPT_H, drop_hosts=tuple(OPT_H.get('drop_hosts', ())) + (h,)))
    loo.append(dict(drop='SN in ' + h, H0=R['H0'], sH0=R['sH0'], chi2=R['chi2'], dof=R['dof']))
for r in D['host_measurements']:
    if r[2] == 'MW':
        continue
    R = fit(D, **dict(OPT_H, drop_primary=(tuple(r[:3]),)))
    loo.append(dict(drop='%s %s/%s' % tuple(r[:3]), H0=R['H0'], sH0=R['sH0'], chi2=R['chi2'], dof=R['dof']))
save('loo_hybrid.json', loo)

# forest plot
fig, ax = plt.subplots(figsize=(10, 8.6))
ordered = [v for v in var if v['group'] != 'hybrid'] + [v for v in var if v['group'] == 'hybrid']
cmap = {'baseline': INK, 'HF convention': BLUE, 'HF sample': AQUA, 'calibration': ORANGE, 'hypothetical': GREY, 'hybrid': MAGENTA}
for j, v in enumerate(ordered[::-1]):
    ax.errorbar(v['H0'], j, xerr=v['sH0'], fmt='o', color=cmap[v['group']], ms=7, mec='white', mew=1.1, elinewidth=1.8)
    ax.text(155, j, 'χ²/ν=%.1f/%d' % (v['chi2'], v['dof']), va='center', fontsize=7.5, color=INK)
ax.set_yticks(range(len(ordered))); ax.set_yticklabels([v['label'] for v in ordered[::-1]], fontsize=8)
ax.axvspan(LIT['H0DN_H0'][0] - LIT['H0DN_H0'][1], LIT['H0DN_H0'][0] + LIT['H0DN_H0'][1], color=MAGENTA, alpha=0.15)
ax.axvspan(LIT['Planck_H0'][0] - LIT['Planck_H0'][1], LIT['Planck_H0'][0] + LIT['Planck_H0'][1], color=GREY, alpha=0.25)
ax.text(LIT['H0DN_H0'][0], len(ordered) - 0.2, 'H0DN 73.50±0.81', fontsize=8, ha='left', color=INK)
ax.text(LIT['Planck_H0'][0], len(ordered) + 0.5, 'Planck 67.4±0.5', fontsize=8, ha='right', color=INK)
ax.set_xlim(55, 170); ax.set_ylim(-1, len(ordered) + 1.2)
ax.set_xlabel(r'$H_0$ [km s$^{-1}$ Mpc$^{-1}$]')
ax.legend(handles=[plt.Line2D([], [], marker='o', ls='', color=c, label=k) for k, c in cmap.items()], loc='center', bbox_to_anchor=(0.42, 0.45), frameon=True, fontsize=8)
ax.set_title('Analysis variants: file-internal Hubble flow vs hybrid (literature a_B)', fontsize=10, loc='left')
fig.savefig(os.path.join(IMG, 'fig5_variants_forest.png')); plt.close(fig)

# ============================================================ 6. Hubble diagrams
print('== 6. Hubble diagrams')
fig, axes = plt.subplots(2, 2, figsize=(12, 7.5), gridspec_kw=dict(height_ratios=[2.2, 1]), sharex='col')
for c, (lab, rows, Mk, col) in enumerate([('SN Ia', D['hubble_flow_sneia'], 'M_SN', BLUE), ('SBF', D['hubble_flow_sbf'], 'M_SBF', MAGENTA)]):
    z = np.array([r[0] for r in rows]); m = np.array([r[1] for r in rows]); e = np.array([r[2] for r in rows]); v = np.array([r[3] for r in rows])
    M, sM = par(R_base, Mk)
    mu = m - M; sv = 5 * v / (LN10 * C_KMS * z); err = np.sqrt(e ** 2 + sv ** 2 + sM ** 2)
    zz = np.linspace(0.015, 0.09, 200)
    def mu_model(zv, H0): return 5 * np.log10(C_KMS * zv * fz(zv) / H0) + 25
    ax = axes[0, c]
    ax.vlines(z, mu - err, mu + err, color=col, lw=1.6, linestyles='solid')
    ax.plot(z, mu, 'o', color=col, ms=8, mec='white', mew=1.2, label='HF data, μ = m − %s (file calib.)' % Mk)
    ax.plot(zz, mu_model(zz, R_base['H0']), color=col, lw=2, label='best fit H0=%.1f' % R_base['H0'])
    ax.plot(zz, mu_model(zz, 73.5), color=INK, lw=2, ls='--', label='H0 = 73.5')
    if c == 0:
        aB = LIT['R22_aB'][0]
        ax.plot(zz, mu_model(zz, 10 ** (0.2 * M + aB + 5)), color=GREY, lw=1.6, ls=':', label='R22 a_B with file M_SN (H0=%.1f)' % (10 ** (0.2 * M + aB + 5)))
    ax.set_xscale('log'); ax.set_ylabel('distance modulus μ [mag]'); ax.legend(frameon=False, fontsize=8, loc='upper left')
    ax.set_title('Hubble flow: %s (%d objects)' % (lab, len(z)), fontsize=10, loc='left')
    ax = axes[1, c]
    ax.axhline(0, color=INK, ls='--', lw=1.2)
    rr_ = mu - mu_model(z, 73.5)
    ax.vlines(z, rr_ - err, rr_ + err, color=col, lw=1.6, linestyles='solid')
    ax.plot(z, rr_, 'o', color=col, ms=7, mec='white', mew=1.2)
    ax.set_ylim(-1.8, 0.3)
    ax.plot(zz, mu_model(zz, R_base['H0']) - mu_model(zz, 73.5), color=col, lw=1.5)
    ax.set_ylabel('μ − μ(H0=73.5) [mag]'); ax.set_xlabel('redshift z')
    ax.set_xticks([0.02, 0.03, 0.05, 0.08]); ax.set_xticklabels(['0.02', '0.03', '0.05', '0.08'])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
fig.savefig(os.path.join(IMG, 'fig4_hubble_diagrams.png')); plt.close(fig)

# ============================================================ 7. validation: MCMC, mocks, injections, budget
print('== 7. validation')
import emcee
def run_mcmc(R, seed, nsteps=5000):
    s = R['sys']; Ci = R['Ci']; npar = s.p; nw = 4 * npar
    def lnp(q):
        r = s.y - s.L @ q; return -0.5 * r @ Ci @ r
    rng = np.random.default_rng(seed)
    p0 = R['q'] + rng.normal(size=(nw, npar)) * np.sqrt(np.diag(R['cov'])) * 0.5
    sam = emcee.EnsembleSampler(nw, npar, lnp); sam._random = np.random.mtrand.RandomState(seed)
    sam.run_mcmc(p0, nsteps, progress=False)
    ch = sam.get_chain(discard=1000, thin=5, flat=True)
    k = R['P']['5logH0']; H = 10 ** (ch[:, k] / 5)
    return dict(H0_med=float(np.median(H)), H0_sd=float(np.std(H)), H0_16=float(np.percentile(H, 16)), H0_84=float(np.percentile(H, 84)),
                M_med=float(np.median(ch[:, R['P']['M_SN']])), M_sd=float(np.std(ch[:, R['P']['M_SN']])), H=H)
mc_b = run_mcmc(R_base, 1); mc_h = run_mcmc(R_hyb, 2)
print('  MCMC baseline H0 %.2f±%.2f (analytic %.2f±%.2f); hybrid %.2f±%.2f (analytic %.2f±%.2f)' % (mc_b['H0_med'], mc_b['H0_sd'], R_base['H0'], R_base['sH0'], mc_h['H0_med'], mc_h['H0_sd'], R_hyb['H0'], R_hyb['sH0']))

def mocks(R, n, seed, truth_H0=73.5, truth_M=-19.25):
    s = R['sys']; rng = np.random.default_rng(seed)
    qt = R['q'].copy(); qt[R['P']['5logH0']] = 5 * np.log10(truth_H0); qt[R['P']['M_SN']] = truth_M
    if 'M_SBF' in R['P']:
        qt[R['P']['M_SBF']] = -3.0
    Lc = np.linalg.cholesky(s.C); A = R['A']; Ci = R['Ci']; k = R['P']['5logH0']
    sd = np.sqrt(R['cov'][k, k])
    out = np.empty((n, 3))
    for i in range(n):
        y = s.L @ qt + Lc @ rng.normal(size=s.n)
        q = A @ y; r = y - s.L @ q
        out[i] = (10 ** (q[k] / 5), (q[k] - qt[k]) / sd, r @ Ci @ r)
    return out
mk_b = mocks(R_base, 5000, 11); mk_h = mocks(R_hyb, 5000, 12)
val = {}
for lab, mk, R in [('baseline', mk_b, R_base), ('hybrid', mk_h, R_hyb)]:
    k = R['P']['5logH0']; sd = np.sqrt(R['cov'][k, k])
    pulls = mk[:, 1]
    val[lab] = dict(mean_H0=float(mk[:, 0].mean()), median_H0=float(np.median(mk[:, 0])), bias_pct=float(100 * (np.median(mk[:, 0]) / 73.5 - 1)),
                    pull_mean=float(pulls.mean()), pull_sd=float(pulls.std()), cover68=float(np.mean(abs(pulls) < 1)),
                    chi2_mean=float(mk[:, 2].mean()), dof=R['dof'], sH0_rel_pred=float(LN10 / 5 * sd * 100))
    print('  mocks %s:' % lab, val[lab])

# injection tests on noiseless mock truth (baseline topology, both HF samples)
s = R_base['sys']
qt = R_base['q'].copy(); qt[R_base['P']['5logH0']] = 5 * np.log10(73.5); qt[R_base['P']['M_SN']] = -19.25; qt[R_base['P']['M_SBF']] = -3.0
y0 = s.L @ qt
types = s.types()
def refit(y):
    q = R_base['A'] @ y; r = y - s.L @ q
    return 10 ** (q[R_base['P']['5logH0']] / 5), float(r @ R_base['Ci'] @ r)
inj = []
for lab, dy in [('none', np.zeros(s.n)),
                ('HF offset −0.30 mag (all HF)', np.where(np.char.startswith(types, 'hf'), -0.30, 0.0)),
                ('HF offset −0.30 mag (SN HF only)', np.where(types == 'hf_sn', -0.30, 0.0)),
                ('One SN calibrator −1.0 mag (NGC1309-like)', np.array([-1.0 if (i['type'] == 'sn_cal' and i['host'] == 'NGC1365') else 0 for i in s.info])),
                ('HF slope tilt 1.25 (SN HF)', np.array([0.25 * (5 * np.log10(i['z']) - np.mean([5 * np.log10(j['z']) for j in s.info if j['type'] == 'hf_sn'])) if i['type'] == 'hf_sn' else 0 for i in s.info])),
                ('Anchor N4258 shifted +0.10', np.array([0.10 if (i['type'] == 'primary' and i['anchor'] == 'N4258') else 0 for i in s.info]))]:
    H, c2 = refit(y0 + dy)
    inj.append(dict(test=lab, H0=H, dH0_pct=100 * (H / 73.5 - 1), chi2=c2))
    print('  inject %-45s H0=%.2f chi2=%.2f' % (lab, H, c2))
save('validation.json', dict(mcmc_base={k: v for k, v in mc_b.items() if k != 'H'}, mcmc_hyb={k: v for k, v in mc_h.items() if k != 'H'}, mocks=val, injections=inj))

def budget(R):
    s = R['sys']; a = R['A'][R['P']['5logH0']]
    tot = a @ s.C @ a
    parts = {k: float(a @ M @ a) for k, M in s.C_blocks.items() if np.any(M)}
    return dict(total_var=float(tot), sum_parts=float(sum(parts.values())), parts=parts, total_mag=float(np.sqrt(tot)), total_pct=float(100 * LN10 / 5 * np.sqrt(tot)))
bud = {'baseline': budget(R_base), 'hybrid': budget(R_hyb), 'hybrid_sigint': budget(R_hyb_si)}
save('error_budget.json', bud)
for k, b in bud.items():
    print('  budget %s: total %.4f mag (%.2f%%), check sum %.2e' % (k, b['total_mag'], b['total_pct'], b['total_var'] - b['sum_parts']))
    for p, v in sorted(b['parts'].items(), key=lambda x: -x[1]): print('     %-18s %.5f  (%.1f%%)' % (p, v, 100 * v / b['total_var']))

# validation figure
fig, axes = plt.subplots(1, 5, figsize=(20, 4.0), gridspec_kw=dict(width_ratios=[1, 1, 1, 1.25, 0.8]))
ax = axes[0]
ax.hist(mk_b[:, 1], bins=60, density=True, color=BLUE, alpha=0.55, label='baseline topology')
ax.hist(mk_h[:, 1], bins=60, density=True, color=MAGENTA, alpha=0.45, label='hybrid topology')
xx = np.linspace(-4, 4, 200); ax.plot(xx, stats.norm.pdf(xx), color=INK, lw=1.5, label='N(0,1)')
ax.set_xlabel('pull of 5logH0'); ax.legend(frameon=False, fontsize=7.5); ax.set_title('(a) 5000 mocks: pulls', fontsize=10, loc='left')
ax = axes[1]
for mk, R, col in [(mk_b, R_base, BLUE), (mk_h, R_hyb, MAGENTA)]:
    ax.hist(mk[:, 2], bins=60, density=True, color=col, alpha=0.5)
    xx = np.linspace(0, 50, 300); ax.plot(xx, stats.chi2.pdf(xx, R['dof']), color=col, lw=1.5)
    ax.axvline(R['chi2'], color=col, ls='--', lw=1.5)
ax.set_xlabel(r'$\chi^2$'); ax.set_title('(b) mock χ² vs real data (dashed)', fontsize=10, loc='left'); ax.set_xlim(0, 60)
ax.text(0.97, 0.72, 'real data: baseline χ²=%.1f/%d (blue)\nhybrid χ²=%.1f/%d (pink)' % (R_base['chi2'], R_base['dof'], R_hyb['chi2'], R_hyb['dof']), transform=ax.transAxes, ha='right', va='top', fontsize=7.5)
ax = axes[2]
ax.hist(mc_h['H'], bins=60, density=True, color=MAGENTA, alpha=0.5, label='emcee (hybrid)')
xx = np.linspace(60, 80, 300); ax.plot(xx, stats.norm.pdf(xx, R_hyb['H0'], R_hyb['sH0']), color=INK, lw=1.5, label='analytic GLS')
ax.set_xlabel('H0'); ax.legend(frameon=False, fontsize=7.5); ax.set_title('(c) MCMC vs analytic', fontsize=10, loc='left')
ax = axes[3]
ax.barh(range(len(inj)), [r['dH0_pct'] for r in inj[::-1]], color=BLUE, height=0.55)
ax.set_yticks(range(len(inj))); ax.set_yticklabels([r['test'] for r in inj[::-1]], fontsize=8)
ax.axvline(0, color=GREY, lw=1); ax.set_xlabel('ΔH0 [%] (truth 73.5)'); ax.set_title('(d) injections: effect on H0', fontsize=10, loc='left')
ax = axes[4]
ax.barh(range(len(inj)), [r['chi2'] for r in inj[::-1]], color=ORANGE, height=0.55)
ax.set_yticks(range(len(inj))); ax.set_yticklabels([]); ax.set_xlabel('χ² of noiseless mock'); ax.set_title('(e) ... and on χ²', fontsize=10, loc='left')
fig.tight_layout(); fig.savefig(os.path.join(IMG, 'fig6_validation.png')); plt.close(fig)

# error budget figure
fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
names = {'anchor': 'anchors', 'method_anchor': 'method–anchor', 'stat_primary': 'primary stat', 'stat_sn_cal': 'SN calibrators',
         'stat_sbf_cal': 'SBF calib.+depth', 'stat_link': 'Fornax link (depth)', 'stat_hf_sn': 'HF SNe', 'stat_hf_sbf': 'HF SBF', 'stat_ext': 'external a_B'}
for ax, key, title in zip(axes, ['baseline', 'hybrid'], ['file-internal baseline', 'hybrid (file ladder + R22 a_B)']):
    b = bud[key]; items = sorted(b['parts'].items(), key=lambda x: -x[1])
    ax.barh(range(len(items)), [100 * LN10 / 5 * np.sqrt(max(v, 0)) for k, v in items], color=BLUE, height=0.6)
    ax.set_yticks(range(len(items))); ax.set_yticklabels([names.get(k, k) for k, v in items]); ax.invert_yaxis()
    ax.set_xlabel('contribution to σ(H0) [%] (sqrt of variance share)')
    ax.set_title('%s: total %.2f%%' % (title, b['total_pct']), fontsize=10, loc='left')
fig.tight_layout(); fig.savefig(os.path.join(IMG, 'fig7_error_budget.png')); plt.close(fig)

# ============================================================ 8. Planck comparison
print('== 8. Planck')
pl = LIT['Planck_H0']
comp = []
for lab, H, sH in [('This work: file-internal baseline', R_base['H0'], R_base['sH0']),
                   ('This work: hybrid (file ladder + R22 a_B)', R_hyb['H0'], R_hyb['sH0']),
                   ('This work: hybrid + SN σ_int', R_hyb_si['H0'], R_hyb_si['sH0']),
                   ('H0DN baseline (quoted)', *LIT['H0DN_H0']), ('R22 SH0ES (quoted)', *LIT['R22_H0']), ('Breuval+24 w/ SMC (quoted)', *LIT['B24_H0'])]:
    comp.append(dict(label=lab, H0=H, sH0=sH, tension=(H - pl[0]) / np.hypot(sH, pl[1])))
save('planck_comparison.json', comp)
for c in comp: print('  %-45s %.2f±%.2f  %.1fσ' % (c['label'], c['H0'], c['sH0'], c['tension']))
fig, ax = plt.subplots(figsize=(8.5, 3.6))
cols = [BLUE, MAGENTA, MAGENTA, INK, GREY, GREY]
for j, (c, col) in enumerate(zip(comp[::-1], cols[::-1])):
    ax.errorbar(c['H0'], j, xerr=c['sH0'], fmt='o', color=col, ms=8, mec='white', mew=1.2, elinewidth=1.8)
    ax.text(c['H0'] + c['sH0'] + 0.6, j + 0.12, '%.2f ± %.2f  (%.1fσ)' % (c['H0'], c['sH0'], c['tension']), fontsize=8, color=INK)
ax.axvspan(pl[0] - pl[1], pl[0] + pl[1], color=GREY, alpha=0.3); ax.text(pl[0], len(comp) - 0.35, 'Planck ΛCDM 67.4±0.5', ha='center', fontsize=8)
ax.set_yticks(range(len(comp))); ax.set_yticklabels([c['label'] for c in comp[::-1]], fontsize=8.5)
ax.set_xlabel(r'$H_0$ [km s$^{-1}$ Mpc$^{-1}$]'); ax.set_xlim(62, 140); ax.set_ylim(-0.7, len(comp))
ax.set_title('Comparison with the early-Universe (CMB) value; tension = ΔH0/√(σ²+0.5²)', fontsize=10, loc='left')
fig.savefig(os.path.join(IMG, 'fig8_planck.png')); plt.close(fig)

# zoomed Planck panel for the hybrid region
print('done.')
