"""Figures for Lyapunov theory checks, equivalence, overshoot, ADMM and kappa sweep."""
import json, pickle, os, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from common import OUT, IMG

plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.grid': True, 'grid.color': '#e6e6e3', 'grid.linewidth': 0.6, 'legend.frameon': False})
C = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948']
T = json.load(open(f'{OUT}/theory.json'))

# ---------------- continuous flow ----------------
fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
for i, mu in enumerate(['1.0', '0.1']):
    c = T['continuous'][mu]; t = np.array(c['t']); E = np.array(c['E'])
    ax[0].semilogy(t, E / E[0], color=C[i], lw=2.2, label=f'AG flow $\\mathcal{{E}}(t)/\\mathcal{{E}}(0)$, $\\mu$={mu}')
    ax[0].semilogy(t, np.maximum(np.array(c['gap_gf']) / c['gap_gf'][0], 1e-30), color=C[i], ls='--', lw=1.4,
                   label=f'gradient flow $(f-f^\\star)/(f_0-f^\\star)$, $\\mu$={mu}')
    m = E > 1e-12 * E[0]
    ax[1].plot(t[m], np.log(E[m] / E[0]) + t[m], color=C[i], lw=2, label=f'$\\mu$={mu}')
    ax[2].plot(t, c['gam'], color=C[i], lw=2, label=f'$\\mu$={mu}')
ax[0].semilogy(t, np.exp(-t), color='k', ls=':', lw=1.2, label='$e^{-t}$ (theorem)')
ax[0].set_ylim(1e-14, 2); ax[0].set_xlabel('t'); ax[0].legend(fontsize=7.5)
ax[0].set_title('(a) strong Lyapunov decay along the AG flow (ridge, random $x_0$)', fontsize=9.5)
ax[1].axhline(0, color='k', lw=0.8); ax[1].set_xlabel('t'); ax[1].set_ylabel(r'$\log(\mathcal{E}(t)/\mathcal{E}(0))+t$')
ax[1].set_title('(b) must stay $\\leq 0$  (masked where $\\mathcal{E}<10^{-12}\\mathcal{E}_0$)', fontsize=9.5); ax[1].legend()
ax[2].set_xlabel('t'); ax[2].set_ylabel(r'$\gamma(t)$'); ax[2].set_yscale('log')
ax[2].set_title(r"(c) $\gamma'=\mu-\gamma$: $\gamma(t)=\mu+(\gamma_0-\mu)e^{-t}$", fontsize=9.5); ax[2].legend()
plt.tight_layout(); plt.savefig(f'{IMG}/fig6_continuous_lyapunov.png', dpi=150); plt.close()

# ---------------- discrete Lyapunov ----------------
fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
for i, mu in enumerate(['1.0', '0.1', '0.01']):
    d = T['discrete'][mu]
    for name, ls in [('VOS-I', '--'), ('VOS-II', '-')]:
        r = np.array(d[f'{name}_g0=mu']['ratio'])
        ax[0].plot(np.arange(len(r)), r, color=C[i], ls=ls, lw=1.3, label=f'{name}, $\\mu$={mu}')
    E = np.array(d['VOS-II_g0=mu']['E']); K = len(d['VOS-II_g0=mu']['ratio'])
for name, ls in [('VOS-I', '--'), ('VOS-II', '-')]:
    for i, mu in enumerate(['1.0', '0.1', '0.01']):
        d = T['discrete'][mu]
        E = np.array(d[f'{name}_g0=L']['E']); step = {'1.0': 1, '0.1': 3, '0.01': 7}[mu]
        ax[1].semilogy(np.arange(len(E)) * step, E / E[0], color=C[i], ls=ls, lw=1.5, label=f'{name}, $\\mu$={mu}')
ax[0].axhline(1, color='k', lw=0.8); ax[0].set_ylim(0.4, 1.05); ax[0].set_xlabel('iteration k')
ax[0].set_ylabel(r'$(1+\alpha_k)\,\mathcal{E}_{k+1}/\mathcal{E}_k$'); ax[0].set_title(r'(a) per-step contraction, $\gamma_0=\mu$ (must be $\leq1$)', fontsize=9.5)
ax[0].legend(fontsize=7, ncol=2, loc='lower right')
ax[1].set_ylim(1e-13, 2); ax[1].set_xlabel('iteration k'); ax[1].set_ylabel(r'$\mathcal{E}_k/\mathcal{E}_0$')
ax[1].set_title(r'(b) discrete Lyapunov value, $\gamma_0=L$, elastic net', fontsize=9.5); ax[1].legend(fontsize=7, ncol=2)
mus = [1.0, 0.1, 0.01]
r2 = [T['discrete'][str(m)]['VOS-II_g0=mu']['rate_last'] for m in mus]
r1 = [T['discrete'][str(m)]['VOS-I_g0=mu']['rate_last'] for m in mus]
pr = [T['discrete'][str(m)]['one_minus_sqrtq'] for m in mus]
ax[2].semilogx(mus, 1 - np.array(pr), 'k:', lw=1.2, label=r'$\sqrt{\mu/L}$')
ax[2].semilogx(mus, 1 - np.array(r2), 'o', color=C[0], ms=9, label=r'VOS-II: $1-1/(1+\alpha)$')
ax[2].semilogx(mus, 1 - np.array(r1), 's', color=C[1], ms=6, label=r'VOS-I: $1-1/(1+\alpha)$')
ax[2].set_yscale('log'); ax[2].set_xlabel(r'$\mu$'); ax[2].set_ylabel('guaranteed contraction per step')
ax[2].set_title(r'(c) certified rate $1/(1+\alpha)$ vs $1-\sqrt{\mu/L}$', fontsize=9.5); ax[2].legend(fontsize=8)
plt.tight_layout(); plt.savefig(f'{IMG}/fig7_discrete_lyapunov.png', dpi=150); plt.close()

# ---------------- equivalence & overshoot ----------------
fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
rd = np.array(T['equivalence']['lasso_vos2_vs_fista_reldiff'])
ax[0].semilogy(np.arange(1, len(rd)), np.maximum(rd[1:], 1e-17), color=C[0], lw=2)
ax[0].set_xlabel('iteration k'); ax[0].set_ylabel(r'$\|x_k^{VOS-II}-x_k^{FISTA}\|/\|x_k^{FISTA}\|$')
ax[0].set_title('(a) $\\mu=0$: VOS-AG-II $\\to$ FISTA (transient only)\n'
                f'$\\mu$=0.1, $\\gamma_0=\\mu$: max diff to Nesterov = {T["equivalence"]["EN_mu0.1_vos2_vs_nesterov_maxrel"]:.1e}', fontsize=9.5)
for j, mu in enumerate(['0.1', '0.01']):
    a = ax[1 + j]
    for i, (nm, c, ls) in enumerate([('PG', '#52514e', '--'), ('FISTA', '#eb6834', '-'), ('FISTA+grad-restart', '#eb6834', '-.'),
                                    ('VOS-AG-II (g0=mu)', '#2a78d6', '-'), ('VOS-AG-II (g0=L)', '#1baf7a', '-'),
                                    ('VOS-AG-I (g0=mu)', '#6250d6', ':')]):
        o = T['overshoot'][f'{mu}_{nm}']; g = np.maximum(np.array(o['gap']), 1e-16)
        a.semilogy(np.arange(len(g)), g, color=c, ls=ls, lw=2 if 'VOS-AG-II (g0=mu)' == nm else 1.3,
                   label=f'{nm} ({o["n_increase"]} increases)')
    a.set_xlabel('iteration k'); a.set_ylabel(r'$F(x_k)-F^\star$'); a.set_ylim(1e-15, 1e5)
    a.set_title(f'({"bc"[j]}) elastic net $\\mu$={mu}, random $x_0$: overshoot', fontsize=9.5); a.legend(fontsize=7)
plt.tight_layout(); plt.savefig(f'{IMG}/fig8_equivalence_overshoot.png', dpi=150); plt.close()

# ---------------- ADMM ----------------
if os.path.exists(f'{OUT}/admm_linear.json'):
    AD = json.load(open(f'{OUT}/admm_linear.json'))
    D = pickle.load(open(f'{OUT}/lasso_hist.pkl', 'rb'))
    pred = AD['pred']
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    bs = sorted(float(b) for b in AD['res'])
    for i, be in enumerate([0.316, 1.0, 5.62, 17.8, 56.2, 177.8]):
        bb = min(bs, key=lambda x: abs(np.log(x / be))); V = np.array(AD['res'][str(bb)]['V'])
        ax[0].semilogy(np.arange(len(V)) * 5, V / V[0], color=C[i], lw=1.6, label=f'$\\beta$={bb:.3g}')
    ax[0].set_xlim(0, 800); ax[0].set_ylim(1e-30, 2); ax[0].set_xlabel('iteration k')
    ax[0].set_ylabel(r'$V_k/V_0$, $V=\beta\|u-u^\star\|^2+\beta\|z-z^\star\|^2$')
    ax[0].set_title('(a) ADMM Lyapunov function: monotone for every $\\beta$', fontsize=9.5); ax[0].legend(fontsize=8)
    rv = [AD['res'][str(b)]['rate_V'] for b in bs]
    rt = [AD['res'][str(b)].get('rho_T', np.nan) ** 2 for b in bs]
    ax[1].semilogx(bs, rv, 'o-', color=C[0], ms=7, label='empirical rate of $V_k$ (fit)')
    ax[1].semilogx(bs, rt, 'x--', color=C[1], ms=9, mew=2, label=r'$\rho(T_\beta)^2$, linearised on supp$(x^\star)$')
    ax[1].axhline(pred['gap_rate_pg'], color='#52514e', ls=':', label=f'PG local $(1-\\mu_S/L)^2$={pred["gap_rate_pg"]:.3f}')
    ax[1].axhline(pred['gap_rate_nest_muS'], color='#2a78d6', ls=':', label=f'Nesterov($\\mu_S$) local $\\rho^2$={pred["gap_rate_nest_muS"]:.3f}')
    ax[1].axvline(pred['sqrt_muS_LS'], color='#999', lw=0.8)
    ax[1].set_xlabel(r'$\beta$'); ax[1].set_ylabel('per-iteration contraction'); ax[1].set_ylim(0, 1.02)
    ax[1].set_title('(b) local linear rate of ADMM vs $\\beta$', fontsize=9.5); ax[1].legend(fontsize=7.5, loc='lower left')
    sw = D['sweep']; sb = sorted(sw)
    for key, c, nm in [('it_1e6', C[2], '$10^{-6}$'), ('it_1e10', C[0], '$10^{-10}$')]:
        y = [sw[b][key] if sw[b][key] is not None else np.nan for b in sb]
        ax[2].loglog(sb, y, 'o-', color=c, label=f'iterations to rel. gap {nm}')
    ax[2].axvline(pred['sqrt_muS_LS'], color='#999', lw=0.8)
    ax[2].text(pred['sqrt_muS_LS'] * 1.1, 600, r'$\sqrt{\mu_S L_S}$', fontsize=9)
    ax[2].set_xlabel(r'$\beta$'); ax[2].set_ylabel('iterations'); ax[2].legend(fontsize=8)
    ax[2].set_title('(c) ADMM penalty sweep (Lasso, $x_0=0$)', fontsize=9.5)
    plt.tight_layout(); plt.savefig(f'{IMG}/fig9_admm.png', dpi=150); plt.close()

# ---------------- kappa sweep ----------------
if os.path.exists(f'{OUT}/kappa.json'):
    KJ = json.load(open(f'{OUT}/kappa.json'))
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.3)); slopes = {}
    for j, kind in enumerate(['ridge', 'elastic']):
        rows = [v for k, v in KJ.items() if k.startswith(kind)]
        if not rows: continue
        kap = np.array([r['kappa'] for r in rows])
        for i, (nm, c, mk) in enumerate([('PG', '#52514e', 's'), ('FISTA', '#eb6834', '^'), ('VOS-I', '#6250d6', 'D'), ('VOS-II', '#2a78d6', 'o')]):
            y = np.array([r[nm] if r[nm] is not None else np.nan for r in rows], float)
            ok = ~np.isnan(y)
            sl = float(np.polyfit(np.log(kap[ok]), np.log(y[ok]), 1)[0]) if ok.sum() >= 3 else np.nan
            slopes[f'{kind}_{nm}'] = sl
            ax[j].loglog(kap[ok], y[ok], mk + '-', color=c, ms=6, label=f'{nm} (slope {sl:.2f})')
            if nm == 'PG' and kind == 'ridge':
                pp = np.array([r['pg_pred'] for r in rows])
                ax[j].loglog(kap[~ok], pp[~ok], mk, mfc='none', color=c, ms=7, label='PG (predicted, not run)')
        ax[j].set_xlabel(r'$\kappa=L/\mu$'); ax[j].set_ylabel('iterations to rel. gap $10^{-8}$')
        ax[j].set_title(['(a) ridge, random $x_0$', '(b) elastic net ($\\lambda=0.01\\lambda_{max}$), random $x_0$'][j], fontsize=9.5)
        ax[j].legend(fontsize=8)
    json.dump(slopes, open(f'{OUT}/kappa_slopes.json', 'w'), indent=1)
    plt.tight_layout(); plt.savefig(f'{IMG}/fig10_kappa_sweep.png', dpi=150); plt.close()
print('done')
