import os, json, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
O = lambda f: os.path.join(ROOT, 'outputs', f); IMG = lambda f: os.path.join(ROOT, 'report/images', f)
C = dict(lin='#2a78d6', log='#eb6834', mb='#1baf7a', pt='#6250d6', mir='#e34948', model='#8a8a85', a2a='#6250d6', g2d='#eda100', g1d='#008300', exp='#1a1a1a')
GEO = {'all2all': ('all-to-all', C['a2a']), '2d': ('2D NN', C['g2d']), '1d': ('1D NN', C['g1d'])}
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True, 'grid.color': '#e1e0d9',
                     'grid.linewidth': 0.6, 'lines.linewidth': 2, 'legend.frameon': False, 'savefig.dpi': 150})
P = pd.read_csv(O('pooled_estimates.csv')); inst = pd.read_csv(O('per_instance_estimates.csv'))
G = pd.read_csv(O('model_grid.csv')); R = json.load(open(O('model_results.json')))
def pk(fam, meth): return P[(P.family == fam) & (P.method == meth)].sort_values(['N', 'd'])
def band(ax, N, x='d', sel=None, label='gate-count model (±1σ, ±2σ MC)'):
    g = G[G.N == N] if x == 'd' else G[G.d == sel]
    xx = g[x]; ax.fill_between(xx, g.q025, g.q975, color=C['model'], alpha=0.15, lw=0)
    ax.fill_between(xx, g.q16, g.q84, color=C['model'], alpha=0.3, lw=0, label=label); ax.plot(xx, g.F_model, color=C['model'], lw=1.5)

# ---------- Fig 1: data overview
fig, ax = plt.subplots(1, 3, figsize=(14, 4))
import pickle
recs = pickle.load(open(O('data_cache.pkl'), 'rb'))
for d, col in [(8, C['lin']), (20, C['log'])]:
    lx = np.concatenate([[np.log(r['ideal_probs'][k] * 2 ** 40)] * v for r in recs if r['family'] == 'N40_verification' and r['type'] == 'XEB' and r['d'] == d for k, v in r['counts'].items()])
    ax[0].hist(lx, bins=30, range=(-6, 3.5), density=True, histtype='step', color=col, lw=2, label=f'N=40, d={d} (1000 shots)')
u = np.linspace(-6, 3.5, 300); ex = np.exp(u)
for F, ls in [(0, ':'), (1, '--')]:
    ax[0].plot(u, ex * (1 - F + F * ex) * np.exp(-ex), color=C['model'], ls=ls, lw=1.5, label=f'Porter–Thomas, F={F}')
ax[0].set_xlabel(r'$\ln(2^N p_U(s))$ of sampled bitstrings'); ax[0].set_ylabel('density'); ax[0].legend(fontsize=8); ax[0].set_title('(a) ideal probabilities of measured samples')
trb = inst[inst.method == 'TRB_bitsurv']; tf = pd.read_csv(O('trb_fits.csv'))
cols = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#6250d6']
for (N, g), col in zip(trb.groupby('N'), cols):
    m = g.groupby('d').F.agg(['mean', 'std', 'size'])
    ax[1].errorbar(m.index, m['mean'], m['std'] / np.sqrt(m['size']), fmt='o', ms=5, color=col, label=f'N={N}', capsize=2)
    f = tf[tf.N == N].iloc[0]; dd = np.linspace(0, 100, 200); ax[1].plot(dd, 0.5 + f.B * f.p ** dd, color=col, lw=1.2)
ax[1].set_xlabel('transport/1Q-RB depth'); ax[1].set_ylabel('per-qubit bit survival'); ax[1].legend(fontsize=8, ncol=2); ax[1].set_title('(b) transport 1Q RB')
au = pd.read_csv(O('audit_table.csv'))
lab = []; val = []
for (fam, typ), g in au.groupby(['family', 'type']):
    lab.append(f'{fam}\n{typ}'); val.append(g.circuits.sum())
ax[2].barh(range(len(lab)), val, color=C['lin'], height=0.6); ax[2].set_yticks(range(len(lab))); ax[2].set_yticklabels(lab, fontsize=7)
for i, v in enumerate(val): ax[2].text(v + 5, i, str(v), va='center', fontsize=8)
ax[2].set_xlabel('unique circuits (after dedup)'); ax[2].set_title('(c) dataset inventory'); ax[2].grid(axis='y', visible=False)
plt.tight_layout(); plt.savefig(IMG('fig1_data_overview.png')); plt.close()

# ---------- Fig 2: d=12 N-scan (a) estimators linear scale, (b) log scale vs MPS approximability
fig, axs = plt.subplots(1, 2, figsize=(13, 4.8))
m = R['mirror']
for k, ax in enumerate(axs):
    band(ax, None, x='N', sel=12)
    off = {'linXEB': -0.9, 'logXEB': -0.3, 'PTMLE_pooled': 0.3, 'MB': 0.9}
    for meth, col, lab in [('linXEB', C['lin'], 'linear XEB'), ('logXEB', C['log'], 'log-XEB'), ('PTMLE_pooled', C['pt'], 'PT-MLE (pooled)'), ('MB', C['mb'], 'mirror benchmark')]:
        if k == 1 and meth in ('logXEB', 'PTMLE_pooled'): continue
        q = pk('N_scan_depth12', meth); ax.errorbar(q.N + off[meth], q.F, q.sigma_boot, fmt='o', ms=6, color=col, capsize=2, label=lab)
    q = pk('N56_depths', 'MB'); q = q[q.d == 12]; ax.errorbar(q.N + 0.9, q.F, q.sigma_boot, fmt='o', ms=6, color=C['mb'], capsize=2)
    q = pk('N40_verification', 'linXEB'); q = q[q.d == 12]; ax.errorbar(q.N - 1.5, q.F, q.sigma_boot, fmt='s', mfc='white', ms=6, color=C['lin'], capsize=2, label='linear XEB (N40 depth-scan set)')
    ax.errorbar(56 + 1.8, m['F12_sqrt'], m['F12_sqrt_se'], fmt='D', ms=7, color=C['mir'], capsize=2, label=r'$\sqrt{S_{24}}$ (mirror inference)')
    ax.set_xlabel('number of qubits N'); ax.set_ylabel('fidelity')
M = pd.read_csv(O('mps_d12_Nscan.csv'))
for geo, (lab, col) in GEO.items():
    mm = M[(M.geometry == geo) & (M.chi == 64)]
    axs[1].plot(mm.N, np.exp(mm.logF), color=col, ls='--', marker='v', ms=6, lw=1.5, label=f'MPS χ=64, {lab}')
axs[0].set_ylim(0.25, 0.95); axs[0].set_title('(a) d = 12: fidelity estimators vs N'); axs[0].legend(fontsize=7.5, loc='upper right')
axs[1].set_yscale('log'); axs[1].set_ylim(1e-14, 2); axs[1].set_title('(b) d = 12: experiment vs MPS (χ=64) approximation'); axs[1].legend(fontsize=7.5, loc='lower left')
plt.tight_layout(); plt.savefig(IMG('fig2_d12_Nscan.png')); plt.close()

# ---------- Fig 3: N=40 depth scan
fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
li = inst[(inst.family == 'N40_verification') & (inst.method == 'linXEB')]
rng = np.random.default_rng(0)
ax[0].scatter(li.d + rng.uniform(-0.5, 0.5, len(li)), li.F, s=8, color=C['lin'], alpha=0.25, lw=0, label='per-instance linear XEB (20 shots)')
for meth, col, lab, o in [('linXEB', C['lin'], 'linear XEB (50 circ.)', -0.25), ('logXEB', C['log'], 'log-XEB', 0.0), ('MB', C['mb'], 'mirror benchmark', 0.25)]:
    q = pk('N40_verification', meth); ax[1].errorbar(q.d + o, q.F, q.sigma_boot, fmt='o', ms=6, color=col, capsize=2, label=lab)
    if meth == 'linXEB': ax[0].errorbar(q.d, q.F, q.sigma_boot, fmt='o', ms=7, color=C['exp'], capsize=3, label='pooled linear XEB', zorder=5)
q = pk('N_scan_depth12', 'linXEB'); q = q[q.N == 40]; ax[1].errorbar(q.d - 0.5, q.F, q.sigma_boot, fmt='s', mfc='white', ms=6, color=C['lin'], capsize=2, label='linear XEB (N-scan set)')
rg = R['depth_regressions']['N40_linXEB']; dd = np.linspace(6, 22, 50)
ax[1].plot(dd, np.exp(rg['lnA'] - rg['lam'] * dd), color=C['lin'], lw=1.2, ls='--', label=f"exp fit (lin XEB): λ={rg['lam']:.3f}±{rg['lam_se']:.3f}")
rg = R['depth_regressions']['N40_MB']; ax[1].plot(dd, np.exp(rg['lnA'] - rg['lam'] * dd), color=C['mb'], lw=1.2, ls='--', label=f"exp fit (MB): λ={rg['lam']:.3f}±{rg['lam_se']:.3f}")
for a in ax: band(a, 40); a.set_xlim(6, 22); a.set_xlabel('circuit depth d'); a.set_ylabel('fidelity')
ax[0].set_ylim(-1, 3); ax[0].set_title('(a) N = 40: per-instance estimates'); ax[1].set_ylim(0.15, 0.75); ax[1].set_title('(b) N = 40: pooled estimators')
ax[0].legend(fontsize=8); ax[1].legend(fontsize=7.5); plt.tight_layout(); plt.savefig(IMG('fig3_N40_depth_scan.png')); plt.close()

# ---------- Fig 4: N=56 depth scan with mirror inference
fig, ax = plt.subplots(figsize=(7.5, 4.8))
band(ax, 56)
q = pk('N56_depths', 'MB'); ax.errorbar(q.d, q.F, q.sigma_boot, fmt='o', ms=6, color=C['mb'], capsize=2, label='MB return probability (50 circ.)')
rg = R['depth_regressions']['N56_MB_d8-20']; dd = np.linspace(0, 26, 60)
ax.plot(dd, np.exp(rg['lnA'] - rg['lam'] * dd), color=C['mb'], ls='--', lw=1.2, label=f"regression d=8–20: λ={rg['lam']:.4f}±{rg['lam_se']:.4f}, A={np.exp(rg['lnA']):.2f}")
ax.errorbar(12.35, m['F12_sqrt'], m['F12_sqrt_se'], fmt='D', ms=7, color=C['mir'], capsize=2, label=r'$\sqrt{S_{24}}$ → inferred F(d=12)')
ax.errorbar(11.65, m['F12_sqrtAspam'], m['F12_sqrtAspam_se'], fmt='D', ms=7, mfc='white', color=C['mir'], capsize=2, label=r'$\sqrt{A_{SPAM}S_{24}}$')
ax.set_yscale('log'); ax.set_xlabel('circuit depth d'); ax.set_ylabel('fidelity'); ax.set_xlim(0, 26); ax.set_ylim(0.07, 1.1)
ax.set_title('N = 56: mirror-benchmark fidelity vs depth'); ax.legend(fontsize=7.5, loc='lower left'); plt.tight_layout(); plt.savefig(IMG('fig4_N56_depth_scan.png')); plt.close()

# ---------- Fig 5: XEB vs MB
J = pd.read_csv(O('xeb_vs_mb.csv'))
fig, ax = plt.subplots(figsize=(5.2, 5))
ax.errorbar(J.F_MB, J.F, J.sigma_boot, J.sigma_boot_MB, fmt='o', color=C['lin'], ms=6, capsize=2, label='linear XEB')
ax.errorbar(J.F_MB + 0.004, J.F_log, J.sigma_boot_log, J.sigma_boot_MB, fmt='s', color=C['log'], ms=5, capsize=2, label='log-XEB')
ax.plot([0, 1], [0, 1], color=C['model'], lw=1); xb = R['xeb_vs_mb']
ax.text(0.22, 0.85, f"lin: χ²={xb['chi2_lin']:.1f}/{xb['n']}\nlog: χ²={xb['chi2_log']:.1f}/{xb['n']}\nslope={xb['slope_lin_on_MB']:.3f}±{xb['slope_se']:.3f}", fontsize=9)
ax.set_xlim(0.2, 0.95); ax.set_ylim(0.2, 0.95); ax.set_xlabel('MB fidelity'); ax.set_ylabel('XEB fidelity'); ax.set_title('XEB vs MB at matched (N, d)'); ax.legend(loc='lower right')
plt.tight_layout(); plt.savefig(IMG('fig5_xeb_vs_mb.png')); plt.close()

# ---------- Fig 6: model validation
Cv = pd.read_csv(O('model_coverage.csv'))
fig, ax = plt.subplots(1, 2, figsize=(12, 4.3))
grp = {('N_scan_depth12', 'linXEB'): ('calibration: d=12 lin XEB N≤40', C['exp'], 'o'), ('N40_verification', 'linXEB'): ('N40 lin XEB', C['lin'], 'o'),
       ('N40_verification', 'MB'): ('N40 MB', C['mb'], 's'), ('N_scan_depth12', 'MB'): ('d=12 MB N-scan', C['log'], '^'), ('N56_depths', 'MB'): ('N56 MB', C['pt'], 'D')}
i = 0
for (fam, meth), (lab, col, mk) in grp.items():
    g = Cv[(Cv.family == fam) & (Cv.method == meth)]
    ax[0].scatter(np.arange(i, i + len(g)), g.z, color=col, marker=mk, s=30, label=lab); i += len(g) + 1
ax[0].axhspan(-1, 1, color=C['model'], alpha=0.25, lw=0); ax[0].axhspan(-2, 2, color=C['model'], alpha=0.12, lw=0); ax[0].axhline(0, color=C['model'], lw=1)
ax[0].set_ylabel('z = (F_exp − F_model)/σ_tot'); ax[0].set_xticks([]); ax[0].set_title(f"(a) held-out coverage: {R['frac_1sigma']*100:.0f}% in 1σ, {R['frac_2sigma']*100:.0f}% in 2σ"); ax[0].legend(fontsize=7.5, ncol=2, loc='lower left')
g = G[G.N == 56]
ax[1].fill_between(g.d, g.F_model_e2q_m30, g.F_model_e2q_p30, color=C['model'], alpha=0.2, lw=0, label='ε₂Q ±30%')
ax[1].fill_between(g.d, g.q16, g.q84, color=C['model'], alpha=0.45, lw=0, label='full MC 1σ (ε₂Q, ε_mem, SPAM)')
ax[1].plot(g.d, g.F_model, color=C['model'], lw=1.5); ax[1].plot(g.d, g.F_model_allfit, color=C['exp'], lw=1, ls=':', label='ε₂Q fit to all data')
q = pk('N56_depths', 'MB'); ax[1].errorbar(q.d, q.F, q.sigma_boot, fmt='o', color=C['pt'], ms=5, capsize=2, label='N56 MB (held out)')
ax[1].set_yscale('log'); ax[1].set_xlabel('depth d'); ax[1].set_ylabel('fidelity'); ax[1].set_title('(b) N=56: prediction & ε₂Q sensitivity'); ax[1].legend(fontsize=8)
plt.tight_layout(); plt.savefig(IMG('fig6_model_validation.png')); plt.close()

# ---------- Fig 7: synthetic validation
Sv = pd.read_csv(O('synthetic_validation.csv'))
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for meth, col, lab in [('linXEB', C['lin'], 'linear XEB'), ('logXEB', C['log'], 'log-XEB'), ('PTMLE', C['pt'], 'PT-MLE (per-circuit avg.)')]:
    g = Sv[Sv.method == meth]; ax[0].errorbar(g.F_true, g.bias, g.sd_F / np.sqrt(300), fmt='o-', color=col, ms=6, lw=1.5, label=lab)
    ax[1].plot(g.F_true, g.cov1, 'o-', color=col, ms=6, lw=1.5, label=lab)
ax[0].axhline(0, color=C['model'], lw=1); ax[0].set_xlabel('true fidelity'); ax[0].set_ylabel('bias of pooled estimate'); ax[0].legend(fontsize=8); ax[0].set_title('(a) bias (N=14, 50×20 shots, 300 trials)')
ax[1].axhline(0.683, color=C['model'], lw=1, ls='--'); ax[1].set_ylim(0, 1); ax[1].set_xlabel('true fidelity'); ax[1].set_ylabel('1σ coverage'); ax[1].set_title('(b) coverage of circuit-level σ')
plt.tight_layout(); plt.savefig(IMG('fig7_synthetic_validation.png')); plt.close()

# ---------- Fig 8: scrambling
if os.path.exists(O('scrambling_N16.csv')):
    Sc = pd.read_csv(O('scrambling_N16.csv'))
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for geo, (lab, col) in GEO.items():
        g = Sc[Sc.geometry == geo].groupby('d')[['collision', 'S_half']].agg(['mean', 'std'])
        ax[0].errorbar(g.index, g[('collision', 'mean')], g[('collision', 'std')], fmt='o-', color=col, ms=4, lw=1.5, label=lab)
        ax[1].errorbar(g.index, g[('S_half', 'mean')], g[('S_half', 'std')], fmt='o-', color=col, ms=4, lw=1.5, label=lab)
    ax[0].axhline(2, color=C['model'], ls='--', lw=1, label='Porter–Thomas (2)'); ax[0].set_yscale('log'); ax[0].set_xlabel('depth'); ax[0].set_ylabel(r'$2^N\sum p^2$'); ax[0].legend(); ax[0].set_title('(a) collision probability, N=16')
    page = 8 - 1 / (2 * np.log(2)) * 2 ** 8 / 2 ** 8
    ax[1].axhline(7.2788, color=C['model'], ls='--', lw=1, label='Page value'); ax[1].set_xlabel('depth'); ax[1].set_ylabel('half-cut entropy (bits)'); ax[1].legend(); ax[1].set_title('(b) half-cut entanglement, N=16')
    plt.tight_layout(); plt.savefig(IMG('fig8_scrambling_N16.png')); plt.close()

# ---------- Fig 9: classical gap at N=56
if os.path.exists(O('mps_N56_depth.csv')):
    M = pd.read_csv(O('mps_N56_depth.csv'))
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
    q = pk('N56_depths', 'MB')
    ax[0].errorbar(q.d, np.log10(q.F), q.sigma_boot / q.F / np.log(10), fmt='o', color=C['exp'], ms=6, capsize=2, label='experiment (MB), N=56', zorder=5)
    g = G[G.N == 56]; ax[0].plot(g.d, np.log10(g.F_model), color=C['model'], lw=1.5, label='gate-count model')
    for geo, (lab, col) in GEO.items():
        for chi, ls in [(16, ':'), (64, '-'), (128, '--')]:
            mm = M[(M.geometry == geo) & (M.chi == chi)].groupby('d').logF.mean()
            if len(mm): ax[0].plot(mm.index, mm.values / np.log(10), color=col, ls=ls, lw=1.6, label=f'MPS {lab}, χ={chi}')
    ax[0].set_ylim(-45, 0.5); ax[0].set_xlabel('depth d'); ax[0].set_ylabel(r'$\log_{10}$ fidelity'); ax[0].set_title('(a) N=56: experiment vs MPS approximation'); ax[0].legend(fontsize=7, ncol=2, loc='lower left')
    if os.path.exists(O('contraction_width_N56.csv')):
        W = pd.read_csv(O('contraction_width_N56.csv'))
        for geo, (lab, col) in GEO.items():
            w = W[W.geometry == geo].groupby('d').width_greedy.mean(); ax[1].plot(w.index, w.values, 'o-', color=col, ms=5, lw=1.6, label=lab)
        ax[1].axhline(56, color=C['model'], ls='--', lw=1, label='N = 56 (state-vector width)')
        ax[1].set_xlabel('depth d'); ax[1].set_ylabel(r'greedy contraction width (log2 max tensor)'); ax[1].set_title('(b) TN contraction-width heuristic (upper bound)'); ax[1].legend(fontsize=8)
    plt.tight_layout(); plt.savefig(IMG('fig9_classical_gap_N56.png')); plt.close()
print('figures done')
