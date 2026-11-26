"""Fig 2: optimal-mismatch validation; Fig 3: LJ stability maps; table of delta*."""
import json, os, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import icoshell as ic

IMG = '../report/images/'
load = lambda f: json.load(open(f)) if os.path.exists(f) else {}
LJ = load('../outputs/scans_LJ.json'); MO = load('../outputs/scans_Morse.json'); HA = load('../outputs/scans_harm.json')
JO = load('../outputs/joint_seeds.json')
TCOL = {'conservative': '#1f77b4', 'class-raising': '#2ca02c', 'class-lowering': '#d62728', 'non-NN': '#999999'}


def nn_rows(S):
    rows = {}
    for core, v in S.items():
        for c, r in v['results'].items():
            if r['nn'] and not r['edge_of_grid']:
                key = (tuple(v['core_path'][-1]), tuple(r['shell']))
                rows[key] = r
    return rows


# merge extra-candidate scans into their cores
for extra, base in [('c03b', 'c03')]:
    if extra in LJ and base in LJ:
        LJ[base]['results'].update(LJ[extra]['results'])
if 'c11x' in LJ and 'c11' in LJ:
    LJ['c11']['results']['(0, 2)']['delta_opt'] = LJ['c11x']['results']['(0, 2)']['delta_opt']
    LJ['c11']['results']['(0, 2)']['edge_of_grid'] = False
LJx = {k: v for k, v in LJ.items() if k != 'c11x' and k != 'c03b'}
Rl, Rm, Rh = nn_rows(LJx), nn_rows(MO), nn_rows(HA)
# merge c11x (extended grid) result
table = []
for key in sorted(set(Rl) | set(Rm) | set(Rh), key=lambda k: (ic.T_of(*k[0]), ic.T_of(*k[1]))):
    a, b = key
    f = ic.delta_star(ic.T_of(*a), ic.T_of(*b))
    table.append(dict(frm=list(a), to=list(b), step=ic.step_type(a, b), T=ic.T_of(*a), Tp=ic.T_of(*b),
                      cos=ic.step_cos(a, b), formula=f, limit=ic.delta_star_limit(ic.T_of(*a), ic.T_of(*b)) if ic.T_of(*a) else None,
                      LJ=Rl.get(key, {}).get('delta_opt'), Morse10=Rm.get(key, {}).get('delta_opt'),
                      harm=Rh.get(key, {}).get('delta_opt')))
# joint optimisation deltas
jrows = []
for n, v in JO.items():
    p = [tuple(x) for x in v['path']]
    for i in range(1, len(p)):
        jrows.append(dict(path=n, frm=list(p[i - 1]), to=list(p[i]), joint=v['deltas'][i - 1],
                          formula=ic.delta_star(ic.T_of(*p[i - 1]), ic.T_of(*p[i]))))
mae = {}
for k in ['LJ', 'Morse10', 'harm']:
    e = [abs(r[k] - r['formula']) for r in table if r[k] is not None]
    mae[k] = dict(MAE=float(np.mean(e)) if e else None, n=len(e), max=float(np.max(e)) if e else None)
e = [abs(r['joint'] - r['formula']) for r in jrows]
mae['joint'] = dict(MAE=float(np.mean(e)) if e else None, n=len(e))
# excluding the central-atom step (the formula ignores 2nd-neighbour contraction there)
e = [abs(r['LJ'] - r['formula']) for r in table if r['LJ'] is not None and r['T'] > 0]
mae['LJ_noncentral'] = dict(MAE=float(np.mean(e)), n=len(e))
json.dump(dict(table=table, joint=jrows, mae=mae), open('../outputs/mismatch_table.json', 'w'), indent=1)
print(mae)

fig, axs = plt.subplots(1, 3, figsize=(16, 5))
ax = axs[0]
mk = {'LJ': 'o', 'Morse10': 's', 'harm': '^'}
for k, m in mk.items():
    for r in table:
        if r[k] is not None:
            ax.scatter(r['formula'], r[k], marker=m, s=45, facecolors=TCOL[r['step']] if k == 'LJ' else 'none',
                       edgecolors=TCOL[r['step']], lw=1.3)
for r in jrows:
    ax.scatter(r['formula'], r['joint'], marker='*', s=70, c='k', alpha=0.6)
lim = [0, 0.58]
ax.plot(lim, lim, 'k--', lw=1)
for k, m in mk.items():
    ax.scatter([], [], marker=m, facecolors='gray' if k == 'LJ' else 'none', edgecolors='gray',
               label='%s (MAE %.4f, n=%d)' % (k, mae[k]['MAE'] or np.nan, mae[k]['n']))
ax.scatter([], [], marker='*', c='k', label='joint NM path opt. (MAE %.4f, n=%d)' % (mae['joint']['MAE'] or np.nan, mae['joint']['n']))
for t, c in TCOL.items():
    if t != 'non-NN':
        ax.plot([], [], 's', c=c, label=t)
ax.set_xlabel('analytic vertex-contact $\\delta^*$'); ax.set_ylabel('numerical $\\delta^*$ (argmin E($\\delta$))')
ax.legend(fontsize=7.5, loc='upper left'); ax.set_title('(a) formula vs relaxed-energy optimum (NN steps)', fontsize=10)
ax.set_xlim(lim); ax.set_ylim(lim)
# (b) delta* vs sqrt T, by step type, with formula curves for families
ax = axs[1]
Ts = np.linspace(1, 36, 300)
for r in table:
    if r['LJ'] is not None:
        ax.scatter(np.sqrt(r['T']), r['LJ'], c=TCOL[r['step']], s=40, zorder=3)
        ax.annotate('%s$\\to$%s' % (tuple(r['frm']), tuple(r['to'])), (np.sqrt(r['T']), r['LJ']), fontsize=6,
                    xytext=(3, 3), textcoords='offset points')
# formula families
fam = {'conservative (h,k)$\\to$(h,k+1)': [((0, k), (0, k + 1)) for k in range(1, 6)] + [((1, k), (1, k + 1)) for k in range(2, 6)],
       'class-raising (h,k)$\\to$(h+1,k)': [((0, k), (1, k)) for k in range(1, 6)] + [((1, k), (2, k)) for k in range(2, 6)],
       'reverse (h,k)$\\to$(h-1,k+1)': [((1, k), (0, k + 1)) for k in range(1, 6)] + [((2, k), (1, k + 1)) for k in range(2, 6)]}
for (lab, steps), c in zip(fam.items(), ['#1f77b4', '#2ca02c', '#d62728']):
    x = [np.sqrt(ic.T_of(*a)) for a, b in steps]; y = [ic.delta_star(ic.T_of(*a), ic.T_of(*b)) for a, b in steps]
    o = np.argsort(x)
    ax.plot(np.array(x)[o], np.array(y)[o], 'x', c=c, ms=6, label=lab + ' formula')
    yl = [ic.delta_star_limit(ic.T_of(*a), ic.T_of(*b)) for a, b in steps]
    ax.plot(np.array(x)[o], np.array(yl)[o], '.', c=c, ms=4, alpha=0.5)
ax.scatter([], [], c='gray', s=40, label='LJ numerical')
ax.plot([], [], '.', c='gray', label='large-T limit (1-S cos$\\theta$)/(S$\\sqrt{T}$)')
ax.set_xlabel('$\\sqrt{T}$ of inner shell'); ax.set_ylabel('$\\delta^*$')
ax.legend(fontsize=7, loc='upper right'); ax.set_title('(b) $\\delta^*$ by step type and shell size', fontsize=10)
ax.set_ylim(-0.02, 0.6)
# (c) example E(delta) curves for core (1,2)
ax = axs[2]
for core, col in [('c12', None)]:
    if core not in LJ:
        continue
    for c, r in LJ[core]['results'].items():
        g = np.array(r['grid']); E = np.array(r['E']); D = np.array(r['Delta'])
        ax.plot(g, np.where(np.array(r['shift']) < 3, D, np.nan), marker='o', ms=3, c=TCOL[r['step']], label='(1,2)$\\to$%s %s ($\\delta^*$=%.3f)' % (c, r['step'], r['delta_opt']))
        bad = np.array(r['shift']) > 3
        ax.scatter(g[bad], D[bad], marker='x', c='k', s=30, zorder=4)
        if r['delta_formula'] > 0:
            ax.axvline(r['delta_formula'], c=TCOL[r['step']], ls=':', lw=1)
ax.scatter([], [], marker='x', c='k', label='reconstructed (>3$^\\circ$)')
ax.set_xlabel('outer-shell mismatch $\\delta$'); ax.set_ylabel('$\\Delta$=(E-N e$_b$)/N$^{2/3}$ [$\\epsilon$]')
ax.set_title('(c) LJ stability vs $\\delta$ on the 117-atom (1,2) core\n(dotted: formula $\\delta^*$)', fontsize=10)
ax.legend(fontsize=7)
plt.tight_layout(); plt.savefig(IMG + 'fig2_mismatch_validation.png', dpi=140); plt.close()

# ---- Fig 3: stability maps (winner vs delta) for all cores
cores = [c for c in ['c01', 'c11', 'c02', 'c12', 'c03', 'c22', 'c13'] if c in LJ]
fig, axs = plt.subplots(2, 4, figsize=(18, 8))
axs = axs.ravel()
winners = {}
for ax, core in zip(axs, cores):
    v = LJ[core]
    win = None
    for c, r in v['results'].items():
        g = np.array(r['grid']); D = np.array(r['Delta'])
        hk = tuple(r['shell'])
        okm = np.array(r['shift']) < 3
        Dp = np.where(okm, D, np.nan)
        ax.plot(g, Dp, marker='o', ms=3, c=TCOL[r['step']], ls='-' if r['nn'] else '--',
                label='%s %s %s' % (c, ic.shell_class(*hk), r['step']))
        ax.scatter(g[~okm], D[~okm], marker='x', c=TCOL[r['step']], s=18, alpha=0.6)
    G = np.array(list(v['results'].values())[0]['grid'])
    Dm = np.array([[r['Delta'][i] if r['shift'][i] < 3 else np.inf for i in range(len(G))] for r in v['results'].values()])
    names = list(v['results'].keys())
    wi = np.argmin(Dm, axis=0)
    winners[core] = [[float(G[i]), names[wi[i]]] for i in range(len(G))]
    ymin = ax.get_ylim()[0]
    for i in range(len(G)):
        r = list(v['results'].values())[wi[i]]
        ax.axvspan(G[i] - 0.01, G[i] + 0.01, color=TCOL[r['step']], alpha=0.12, lw=0)
    p = v['core_path'][-1]
    ax.set_title('core %s (N=%d): %s' % (tuple(p), int(sum(ic.N_of(*q) for q in v['core_path'])),
                                       ic.shell_class(*p)), fontsize=10)
    ax.set_xlabel('$\\delta$'); ax.set_ylabel('$\\Delta$ [$\\epsilon$]'); ax.legend(fontsize=7)
for ax in axs[len(cores):]:
    ax.axis('off')
axs[-1].text(0.0, 0.5, 'Shading: lowest-$\\Delta$ shell at each $\\delta$\n(blue conservative, green class-raising,\n'
             'red reverse/class-lowering, grey non-NN).\nDashed: non-NN control shells.\n'
             '$\\Delta$=(E+8.61N)/N$^{2/3}$, equal-$\\epsilon$ LJ.', fontsize=10, transform=axs[-1].transAxes)
plt.tight_layout(); plt.savefig(IMG + 'fig3_stability_maps_LJ.png', dpi=130); plt.close()
# windows where each shell wins
wins = {}
for core, w in winners.items():
    seg = []
    for d, n in w:
        if seg and seg[-1][0] == n:
            seg[-1][2] = d
        else:
            seg.append([n, d, d])
    wins[core] = seg
json.dump(wins, open('../outputs/stability_windows.json', 'w'), indent=1)
for k, v in wins.items():
    print(k, v)
