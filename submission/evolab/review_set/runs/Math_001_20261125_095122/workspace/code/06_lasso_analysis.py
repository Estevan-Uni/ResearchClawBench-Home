"""Tables and figures for the headline Lasso experiment."""
import json, pickle, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from common import OUT, IMG, first_hit

plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.grid': True, 'grid.color': '#e6e6e3', 'grid.linewidth': 0.6, 'legend.frameon': False})
R = json.load(open(f'{OUT}/reference.json'))
D = pickle.load(open(f'{OUT}/lasso_hist.pkl', 'rb'))
H = D['H']; Fs = R['ref']['Fs']; F0 = R['facts']['F0']; dF0 = F0 - Fs
CERT = R['ref']['dgap']            # certified accuracy of F*
FLOOR = 1e-15
ST = {  # fixed colour per method (categorical slots, never cycled)
    'PG': dict(c='#52514e', ls='--', lw=1.8),
    'FISTA': dict(c='#eb6834', ls='-', lw=1.3),
    'FISTA+grad-restart': dict(c='#eb6834', ls='-.', lw=1.5),
    'VOS-AG-I': dict(c='#6250d6', ls='-', lw=1.3),
    'VOS-AG-I+restart': dict(c='#6250d6', ls=':', lw=1.6),
    'VOS-AG-II': dict(c='#2a78d6', ls='-', lw=1.1),
    'VOS-AG-II+restart': dict(c='#2a78d6', ls='-', lw=2.4),
    'VOS-AG-II monotone': dict(c='#1baf7a', ls='-', lw=1.6),
    'VOS-AG-II(mu=muS)+restart': dict(c='#e87ba4', ls='-', lw=1.6),
    'ADMM': dict(c='#008300', ls='-', lw=1.6),
    'Fast ADMM (restart)': dict(c='#eda100', ls='--', lw=1.6),
}
LAB = {'VOS-AG-II(mu=muS)+restart': r'VOS-AG-II($\hat\mu=\mu_S$)+restart', 'ADMM': f'ADMM ($\\beta$={D["best_beta"]:.1f})',
       'Fast ADMM (restart)': f'Fast ADMM ($\\beta$={D["best_beta"]:.1f})'}
lab = lambda k: LAB.get(k, k)
tab = {}

# ---------------- Table 1: raw objective at fixed budgets ----------------
ks = [0, 10, 25, 50, 100, 200, 400, 800]
T1 = ['PG', 'FISTA', 'VOS-AG-II', 'VOS-AG-II+restart', 'VOS-AG-II monotone', 'ADMM']
tab['table1'] = {m: {int(k): dict(F=float(H[m]['F'][k]), gap=float(H[m]['gap'][k])) for k in ks} for m in T1}

# ---------------- Table 2: iterations to tolerance ----------------
rel_t = [10.0 ** -e for e in range(3, 13)]
abs_t = [10.0 ** -e for e in range(3, 12)]
fpr_t = [10.0 ** -e for e in range(4, 11)]
tab['table2'] = {}
for m in H:
    h = H[m]; k = h['k']
    tab['table2'][m] = dict(rel={f'{t:.0e}': first_hit(k, h['gap'] / dF0, t) for t in rel_t},
                            abs={f'{t:.0e}': first_hit(k, h['gap'], t) for t in abs_t},
                            fpr={f'{t:.0e}': first_hit(k, h['fpr'], t) for t in fpr_t} if 'fpr' in h else None,
                            mv_rel1e10=(int(h['mv'][first_hit(k, h['gap'] / dF0, 1e-10)])
                                        if first_hit(k, h['gap'] / dF0, 1e-10) is not None else None))

# ---------------- phase analysis (support identification + fitted rates) ----------------
def phases(h):
    k, g, sm = h['k'], np.maximum(h['gap'], 1e-300), h['smis']
    bad = np.nonzero(sm != 0)[0]
    kid = int(k[bad[-1] + 1]) if len(bad) and bad[-1] + 1 < len(k) else (0 if not len(bad) else None)
    out = dict(k_id=kid)
    # early sublinear phase, log-log slope on k in [5, 50]
    w = (k >= 5) & (k <= 50)
    out['loglog_slope_5_50'] = float(np.polyfit(np.log(k[w]), np.log(g[w]), 1)[0])
    if kid is not None:
        w1 = (k >= max(10, kid // 3)) & (k <= kid)
        out['rate_pre'] = float(np.exp(np.polyfit(k[w1], np.log(g[w1]), 1)[0])) if w1.sum() > 3 else None
        w2 = (k >= kid) & (g > 1e-11)
        out['rate_post'] = float(np.exp(np.polyfit(k[w2], np.log(g[w2]), 1)[0])) if w2.sum() > 3 else None
        out['gap_at_kid'] = float(g[kid])
    return out
tab['phases'] = {m: phases(H[m]) for m in H}
# statistical accuracy: ||x_k - x*|| <= 0.1 ||x* - x_true||  and relerr within 1% of final
xs = np.load(f'{OUT}/xstar.npy')
d_stat = R['ref']['relerr']
for m in H:
    h = H[m]
    nxt = np.linalg.norm(xs) / 1.0
    tab['phases'][m]['k_stat'] = first_hit(h['k'], np.abs(h['relerr'] - d_stat) / d_stat, 0.01)
    tab['phases'][m]['k_stat_dist'] = first_hit(h['k'], h['dstar'] / (d_stat * np.linalg.norm(np.load(f'{OUT}/xstar.npy')) ), 0.1)
tab['restarts'] = D['NR']; tab['best_beta'] = D['best_beta']; tab['cert'] = CERT; tab['dF0'] = dF0
json.dump(tab, open(f'{OUT}/lasso_tables.json', 'w'), indent=1)

# markdown versions
L = []
L.append('| method | ' + ' | '.join(f'k={k}' for k in ks) + ' |')
L.append('|---|' + '---|' * len(ks))
for m in T1:
    L.append(f'| {m} F | ' + ' | '.join(f'{H[m]["F"][k]:.10f}' for k in ks) + ' |')
    L.append(f'| {m} F−F* | ' + ' | '.join(f'{max(H[m]["gap"][k], 0):.2e}' for k in ks) + ' |')
L.append('')
for kind, tl in [('rel', rel_t), ('abs', abs_t), ('fpr', fpr_t)]:
    L.append(f'**{kind}**')
    L.append('| method | ' + ' | '.join(f'{t:.0e}' for t in tl) + ' |')
    L.append('|---|' + '---|' * len(tl))
    for m in H:
        dd = tab['table2'][m][kind]
        if dd is None: continue
        L.append(f'| {m} | ' + ' | '.join(str(dd[f'{t:.0e}']) if dd[f'{t:.0e}'] is not None else '>800' for t in tl) + ' |')
    L.append('')
L.append('**phases**')
for m in H:
    L.append(f'{m}: {tab["phases"][m]}')
open(f'{OUT}/lasso_tables.md', 'w').write('\n'.join(L))
print('\n'.join(L))

# ---------------- Figure A: headline (VOS-AG-II+restart vs PG) ----------------
fig, ax = plt.subplots(1, 3, figsize=(17, 4.8), gridspec_kw=dict(width_ratios=[1, 1.15, 0.8]))
for m, nm in [('VOS-AG-II+restart', 'VOS-AG-II + restart (accelerated, this work)'), ('PG', 'proximal gradient (baseline)')]:
    h = H[m]; st = ST[m]
    ax[0].semilogy(h['k'], h['F'], color=st['c'], ls=st['ls'], lw=2.2, label=nm)
    ax[1].semilogy(h['k'], np.maximum(h['gap'], FLOOR), color=st['c'], ls=st['ls'], lw=2.2, label=nm)
ax[0].axhline(Fs, color='#d1495b', lw=1, ls=':', label=f'$F^\\star$ = {Fs:.6f}')
ax[0].set_ylim(Fs * 0.9, F0 * 1.3)
ax[0].set_xlabel('iteration k'); ax[0].set_ylabel(r'$F(x_k)=\frac{1}{2}\|Ax_k-b\|^2+\lambda\|x_k\|_1$')
ax[0].set_title('(a) raw objective (log scale)')
ax[0].set_xlim(0, 400); ax[0].legend(loc='upper right')
ax[1].axhspan(FLOOR, CERT, color='#eeeeee', zorder=0)
ax[1].text(260, 3e-14, 'below certified accuracy of $F^\\star$', fontsize=8, color='#777')
ax[1].set_xlabel('iteration k'); ax[1].set_ylabel(r'$F(x_k)-F^\star$')
ax[1].set_title('(b) suboptimality gap (log scale)'); ax[1].set_xlim(0, 400)
ax[1].set_ylim(FLOOR, 2e3)
ph = tab['phases']
for m in ['PG', 'VOS-AG-II+restart']:
    kid = ph[m]['k_id']
    ax[1].plot([kid], [max(H[m]['gap'][kid], FLOOR)], 'o', ms=7, mfc='white', mec=ST[m]['c'], mew=1.8)
ax[1].annotate(f'PG support identified (k={ph["PG"]["k_id"]})', (ph['PG']['k_id'], H['PG']['gap'][ph['PG']['k_id']]),
               xytext=(205, 1e-1), fontsize=8, arrowprops=dict(arrowstyle='->', color='#777'))
ax[1].annotate(f'VOS support identified (k={ph["VOS-AG-II+restart"]["k_id"]})',
               (ph['VOS-AG-II+restart']['k_id'], H['VOS-AG-II+restart']['gap'][ph['VOS-AG-II+restart']['k_id']]),
               xytext=(105, 1e-9), fontsize=8, arrowprops=dict(arrowstyle='->', color='#777'))
ax[1].legend(loc='upper right')
for m, nm in [('VOS-AG-II+restart', 'VOS-AG-II + restart'), ('PG', 'proximal gradient')]:
    h = H[m]; w = (h['k'] >= 1) & (h['k'] <= 120)
    ax[2].loglog(h['k'][w], np.maximum(h['gap'][w], FLOOR), color=ST[m]['c'], ls=ST[m]['ls'], lw=2.2, label=nm)
kk = np.array([2, 120.]); ax[2].loglog(kk, H['PG']['gap'][2] * 2 / kk, color='#999', lw=1, ls=':', label=r'reference $\propto 1/k$')
ax[2].loglog(kk, H['PG']['gap'][2] * 4 / kk**2, color='#bbb', lw=1, ls='-.', label=r'reference $\propto 1/k^2$')
ax[2].set_xlabel('iteration k (log)'); ax[2].set_ylabel(r'$F(x_k)-F^\star$')
ax[2].set_title('(c) early phase, log-log'); ax[2].legend(fontsize=8, loc='lower left')
plt.tight_layout(); plt.savefig(f'{IMG}/fig2_headline_vos_vs_pg.png', dpi=160); plt.close()

# ---------------- Figure B: all methods, iterations and matvecs ----------------
ADMM_SETUP = 1000 + 167      # A A^T (2 m^2 n flops) + Cholesky (m^3/3) in matvec-equivalents (4e6 flops)
fig, ax = plt.subplots(1, 2, figsize=(13.5, 5))
order = ['PG', 'FISTA', 'FISTA+grad-restart', 'VOS-AG-I', 'VOS-AG-I+restart', 'VOS-AG-II', 'VOS-AG-II+restart',
         'VOS-AG-II monotone', 'VOS-AG-II(mu=muS)+restart', 'ADMM', 'Fast ADMM (restart)']
for m in order:
    h = H[m]; st = ST[m]; g = np.maximum(h['gap'] / dF0, FLOOR / dF0)
    ax[0].semilogy(h['k'], g, color=st['c'], ls=st['ls'], lw=st['lw'], label=lab(m))
    mv = 2.5 * h['k'] if 'ADMM' in m else 2 * h['k']
    ax[1].semilogy(mv, g, color=st['c'], ls=st['ls'], lw=st['lw'], label=lab(m))
ax[1].semilogy(2.5 * H['ADMM']['k'] + ADMM_SETUP, np.maximum(H['ADMM']['gap'] / dF0, FLOOR / dF0), color='#008300',
               ls=':', lw=1.2, label='ADMM incl. factorisation cost')
for a in ax:
    a.axhspan(FLOOR / dF0, CERT / dF0, color='#eeeeee', zorder=0)
    a.set_ylabel(r'relative gap $(F(x_k)-F^\star)/(F(x_0)-F^\star)$'); a.set_ylim(FLOOR / dF0, 3)
ax[0].set_xlabel('iteration k'); ax[0].set_xlim(0, 400); ax[0].set_title('(a) all methods vs iteration')
ax[1].set_xlabel('matrix-vector products (ADMM: 2 + triangular solves ≈ 2.5 / iter)'); ax[1].set_xlim(0, 1600)
ax[1].set_title('(b) all methods vs cost')
ax[0].legend(fontsize=7.5, ncol=2, loc='upper right')
plt.tight_layout(); plt.savefig(f'{IMG}/fig3_all_methods.png', dpi=150); plt.close()

# ---------------- Figure C: non-restarted VOS-AG-II oscillation ----------------
fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
for m in ['FISTA', 'VOS-AG-II', 'VOS-AG-II monotone', 'VOS-AG-II+restart', 'PG']:
    h = H[m]; st = ST[m]
    ax[0].semilogy(h['k'], np.maximum(h['gap'], FLOOR), color=st['c'], ls=st['ls'], lw=st['lw'], label=lab(m))
ax[0].set_xlim(0, 300); ax[0].set_xlabel('iteration k'); ax[0].set_ylabel(r'$F(x_k)-F^\star$')
ax[0].set_title('(a) non-restarted VOS-AG-II ≈ FISTA: ripples from vanishing damping'); ax[0].legend(fontsize=8)
# momentum / damping: gamma_k and effective momentum for mu=0
from common import alpha_II
gam = 100.0; G = []; MOM = []
for k in range(300):
    a = alpha_II(100.0, 0.0, gam); G.append(gam); gam = gam / (1 + a)
G = np.array(G)
ax[1].loglog(np.arange(1, 301), G, color='#2a78d6', lw=2, label=r'$\gamma_k$ (VOS-AG-II, $\mu=0$)')
kk = np.arange(1, 301); ax[1].loglog(kk, 100 * 4 / (kk + 1) ** 2 * 1.0, color='#999', ls=':', label=r'$4L/(k+1)^2$')
ax[1].set_xlabel('iteration k'); ax[1].set_title(r'(b) $\gamma_k\to0$ when $\mu=0$: damping $\mu/\gamma$ vanishes')
ax[1].legend(fontsize=8)
plt.tight_layout(); plt.savefig(f'{IMG}/fig4_oscillation.png', dpi=150); plt.close()

# ---------------- Figure D: recovery & support ----------------
fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
for m in ['PG', 'FISTA', 'VOS-AG-II+restart', 'VOS-AG-I', 'ADMM']:
    h = H[m]; st = ST[m]
    ax[0].semilogy(h['k'], h['relerr'], color=st['c'], ls=st['ls'], lw=st['lw'], label=lab(m))
    ax[1].semilogy(h['k'], np.maximum(h['dstar'], 1e-16), color=st['c'], ls=st['ls'], lw=st['lw'], label=lab(m))
    ax[2].plot(h['k'], h['smis'], color=st['c'], ls=st['ls'], lw=st['lw'], label=lab(m))
ax[0].axhline(d_stat, color='#d1495b', ls=':', lw=1); ax[0].text(250, d_stat * 1.15, r'$\|x^\star-x_{true}\|/\|x_{true}\|$', fontsize=8)
ax[0].set_ylabel(r'$\|x_k-x_{true}\|/\|x_{true}\|$'); ax[0].set_title('(a) recovery error vs ground truth')
ax[1].set_ylabel(r'$\|x_k-x^\star\|$'); ax[1].set_title('(b) distance to the Lasso solution')
ax[2].set_ylabel(r'$|\mathrm{supp}(x_k)\,\triangle\,\mathrm{supp}(x^\star)|$'); ax[2].set_yscale('symlog', linthresh=1)
ax[2].set_title('(c) support mismatch')
for a in ax: a.set_xlabel('iteration k'); a.set_xlim(0, 400)
ax[0].legend(fontsize=8)
plt.tight_layout(); plt.savefig(f'{IMG}/fig5_recovery_support.png', dpi=150); plt.close()
print('figs done')
