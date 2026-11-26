"""Benchmark-style growth figures:
 (i)  dominant shell class vs number of deposited atoms (MD) + lattice-model class frequencies
 (ii) running mean mismatch of deposited species vs deposition step
 (iii) step-type statistics: lattice model (single step from the MD seed shell, and 5-step paths)
       with MD outcome fractions overlaid (Wilson 95% CI)
 (iv) snapshots of grown clusters coloured by species, with hand."""
import json, re, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import icoshell as ic
import lattice_model as lm
from growth_analysis import summary, dominant, tup, wilson, SEEDCLS, parse, excess

IMG = '../report/images/'
COL = {'MC': '#1f77b4', 'BG': '#ff7f0e', 'Ch1': '#2ca02c', 'Ch2': '#d62728', 'Ch3': '#9467bd'}
LMJ = json.load(open('../outputs/lattice_model.json'))
SEEDSHELL = {'E1': (0, 1), 'E3a': (0, 1), 'E3c': (0, 1), 'E3b': (1, 1), 'E2': (1, 2), 'E2b': (1, 3)}


def fam(n):
    return n.split('_')[0]


def run_classes(n, r):
    out = [(0, SEEDCLS[fam(n)], ic.class_index(*SEEDSHELL[fam(n)]))]
    for h in r['hist']:
        cls, best, ex = dominant(h, SEEDCLS[fam(n)])
        ci = ic.class_index(*best) if ex >= 0.3 else ic.class_index(*SEEDSHELL[fam(n)])
        out.append((h['n'], cls, ci))
    return out


# ------------------------------------------------------------- (i)
scen = [('small $\\delta$: E1 $\\delta$=0; E2 $\\delta$=0, 0.03; E2b $\\delta$=0, 0.02', ['E1_d=0_T=0.22_tau=60_n=10', 'E2_d=0.0_T=0.22_tau=40_n=2', 'E2_d=0.03_T=0.22_tau=40_n=2',
                                          'E2b_d=0.0_T=0.22_tau=30_n=1_N=212', 'E2b_d=0.02_T=0.22_tau=30_n=1_N=212']),
        ('E3a $\\delta$=0.22; E2 $\\delta$=0.08; E2b $\\delta$=0.06, 0.08', ['E3a_d=0.22_T=0.22_tau=60_n=4_N=32', 'E2_d=0.08_T=0.22_tau=40_n=2', 'E2b_d=0.06_T=0.22_tau=30_n=2_N=192', 'E2b_d=0.08_T=0.22_tau=30_n=2_N=192']),
        ('E3c Cu/Ag/Cu on Ag$_{13}$; E3b $\\delta$=0.09', ['E3c_T=0.22_tau=60_n=3', 'E3b_d=0.09_T=0.22_tau=40_n=3'])]
fig, axs = plt.subplots(1, 4, figsize=(19, 4.6))
for ax, (title, names) in zip(axs[:3], scen):
    off = 0
    for n in names:
        if n not in summary:
            continue
        for k, r in enumerate(summary[n]['runs']):
            rc = run_classes(n, r)
            x = [a for a, b, c in rc]; y = np.array([c for a, b, c in rc]) + off
            ax.step(x, y, where='post', c='gray', lw=0.8, alpha=0.6)
            ax.scatter(x, y, c=[COL.get(b, 'k') for a, b, c in rc], s=22, zorder=3)
            off += 0.025
    ax.set_yticks([0, 1, 2]); ax.set_yticklabels(['class 0 (MC)', 'class 1 (BG(1,1)/Ch1)', 'class 2 (BG(2,2)/Ch2)'], fontsize=8)
    ax.set_ylim(-0.3, 2.95); ax.set_xlabel('deposited atoms'); ax.set_title('(i) %s' % title, fontsize=10)
for c, col in COL.items():
    axs[0].scatter([], [], c=col, label=c)
axs[0].legend(fontsize=7, loc='upper left')
# lattice model class frequencies by shell step, alkali and mixed scenario
ax = axs[3]
for key, ls in [('thermo_pr0.1_alkali', '-'), ('thermo_pr0.1_mixed', '--'), ('thermo_pr0.1_d0.08', ':')]:
    cf = LMJ['runs'][key]['class_freq_by_step']
    for c in ['MC', 'BG', 'Ch1', 'Ch2']:
        y = [d.get(c, 0) / 500 for d in cf]
        if max(y) > 0:
            ax.plot(range(len(y)), y, ls, c=COL[c], marker='o', ms=3, label='%s %s' % (key.split('_')[-1], c))
ax.set_xlabel('shell step'); ax.set_ylabel('fraction of 500 model paths')
ax.set_title('(i\') lattice model: class frequency per step', fontsize=10); ax.legend(fontsize=6.5, ncol=2)
plt.tight_layout(); plt.savefig(IMG + 'fig7_class_vs_deposition.png', dpi=140); plt.close()

# ------------------------------------------------------------- (ii) running mismatch
fig, axs = plt.subplots(1, 2, figsize=(12, 4.3))
ax = axs[0]
for n in summary:
    r0 = summary[n]['runs'][0]
    js = json.load(open('../outputs/growth/%s_s0.json' % n))
    D = np.array(js['D']); ns = js['nseed']
    dref = D[ns - 1]
    dl = D[ns:] / dref - 1
    rm = np.cumsum(dl) / np.arange(1, len(dl) + 1)
    p_ = parse(n)
    ax.plot(np.arange(1, len(dl) + 1), rm, label='%s %s T=%s $\\tau$=%s' % (fam(n), 'seq Cu/Ag/Cu' if fam(n) == 'E3c' else '$\\delta$=' + p_['d'], p_['T'], p_['tau']))
ax.set_xlabel('deposition step (atoms)'); ax.set_ylabel('running mean mismatch of deposited atoms\n(vs. outer seed species)')
ax.legend(fontsize=6.5, ncol=2); ax.set_title('(ii) running mean mismatch (MD runs)', fontsize=10)
ax = axs[1]
seqs = {'Na$\\to$K$\\to$Rb$\\to$Cs': [0.2204, 0.0925, 0.0685], 'Na$\\to$K$\\to$K$\\to$Rb$\\to$Cs': [0.2204, 0.0, 0.0925, 0.0685]}
for lab, s in seqs.items():
    ax.plot(range(1, len(s) + 1), np.cumsum(s) / np.arange(1, len(s) + 1), '-o', label=lab + ' (shell-wise)')
# dataset growth_results (reference, as given)
gr = [(0, 'MC', 0.00), (10, 'MC', 0.01), (20, 'MC', 0.02), (30, 'MC', 0.025), (40, 'MC', 0.03), (50, 'MC', 0.035), (0, 'Ch1', 0.00), (10, 'Ch1', 0.12), (20, 'Ch1', 0.14), (30, 'Ch1', 0.138), (40, 'Ch1', 0.136), (50, 'Ch1', 0.135), (0, 'MC', 0.00), (10, 'MC', 0.08), (20, 'Ch1', 0.14), (30, 'Ch1', 0.15), (40, 'Ch1', 0.145), (50, 'Ch1', 0.142)]
ax2 = ax.twiny()
for i in range(3):
    seg = gr[6 * i:6 * i + 6]
    ax2.plot([a for a, b, c in seg], [c for a, b, c in seg], ':', c='k', alpha=0.5)
    ax2.scatter([a for a, b, c in seg], [c for a, b, c in seg], c=[COL[b] for a, b, c in seg], s=14, marker='s')
ax2.set_xlabel('dataset growth_results step (dotted, squares coloured by listed class)', fontsize=8)
ax.set_xlabel('shell step'); ax.set_ylabel('running mean mismatch'); ax.legend(fontsize=7, loc='lower right')
ax.set_title('(ii\') sequential-species design paths vs dataset reference', fontsize=10, pad=28)
plt.tight_layout(); plt.savefig(IMG + 'fig8_running_mismatch.png', dpi=140); plt.close()

# ------------------------------------------------------------- (iii) step statistics
lam = LMJ['lam_thermo']


def single_step_probs(seed, dl, lam, p_r=0.1):
    opts = lm.steps_from(seed)
    w = np.array([-lam * ic.N_of(*o) * (dl - lm.dstar(seed, o)[0]) ** 2 for o in opts])
    w = np.exp(w - w.max()); w /= w.sum()
    p = dict(conservative=0.0, **{'mismatch-driven': 0.0}, reverse=0.0)
    for o, wi in zip(opts, w):
        p[lm.stype(seed, o)] += (1 - p_r) * wi + p_r / len(opts)
    return p


WIN = json.load(open('../outputs/stability_windows.json'))
SCJ = json.load(open('../outputs/scans_LJ.json'))
CORE_OF = {(0, 1): 'c01', (1, 1): 'c11', (1, 2): 'c12', (1, 3): 'c13'}


def window_winner(seed, dl):
    """lowest-Delta shell on the LJ stability map at the nearest grid delta (T=0 energetics)"""
    core = CORE_OF[seed]
    res = dict(SCJ[core]['results'])
    best, bv = None, np.inf
    for c, r in res.items():
        g = np.array(r['grid']); D = np.array(r['Delta']); ok = np.array(r['shift']) < 3
        if dl < g.min() or dl > g.max():
            continue
        v = np.interp(dl, g[ok], D[ok])
        if v < bv:
            best, bv = tuple(r['shell']), v
    p = dict(conservative=0.0, **{'mismatch-driven': 0.0}, reverse=0.0)
    p[lm.stype(seed, best)] = 1.0
    return p, best


md_cases = []
for n, s in summary.items():
    f = fam(n)
    if f == 'E3c':
        continue
    seed = SEEDSHELL[f]
    dl = float(parse(n)['d'])
    cnt = dict(conservative=0, **{'mismatch-driven': 0}, reverse=0, unresolved=0)
    for r in s['runs']:
        st = r['id_anneal']
        if not st['resolved']:
            cnt['unresolved'] += 1
        else:
            cnt[lm.stype(seed, st['best'])] += 1
    md_cases.append(dict(name=n, seed=seed, delta=dl, counts=cnt, n=s['n'],
                         model_thermo=single_step_probs(seed, dl, lam), model_soft=single_step_probs(seed, dl, lam / 30),
                         model_window=window_winner(seed, dl)[0], window_shell=window_winner(seed, dl)[1]))
json.dump(md_cases, open('../outputs/md_vs_model_steps.json', 'w'), indent=1, default=str)

fig, axs = plt.subplots(1, 2, figsize=(17, 5.6), gridspec_kw=dict(width_ratios=[1.1, 1]))
ax = axs[0]
TC = {'conservative': '#1f77b4', 'mismatch-driven': '#2ca02c', 'reverse': '#d62728', 'random': '#bbbbbb'}
keys = ['thermo_pr0.1_d0.00', 'thermo_pr0.1_d0.04', 'thermo_pr0.1_d0.08', 'thermo_pr0.1_d0.14', 'thermo_pr0.1_d0.20',
        'thermo_pr0.1_alkali', 'thermo_pr0.1_mixed', 'soft_pr0.1_d0.08', 'soft_pr0.1_d0.14']
bottom = np.zeros(len(keys) + 1)
labels = [k.replace('thermo_pr0.1_', 'th ').replace('soft_pr0.1_', 'soft ') for k in keys] + ['dataset\npath_selection']
ds = dict(conservative=325, **{'mismatch-driven': 125}, random=50, reverse=100)
for t in ['conservative', 'mismatch-driven', 'reverse', 'random']:
    v = [LMJ['runs'][k]['stats'][t] / sum(LMJ['runs'][k]['stats'].values()) for k in keys] + [ds[t] / 600]
    ax.bar(range(len(v)), v, bottom=bottom, color=TC[t], label=t)
    bottom += v
ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=35, fontsize=8, ha='right')
ax.set_ylabel('fraction of steps'); ax.legend(fontsize=8, loc='upper center', bbox_to_anchor=(0.5, -0.25), ncol=4)
ax.set_title('(iii) lattice-model step statistics (500 paths x 5 steps, p$_r$=0.1)', fontsize=10)
ax = axs[1]
x = np.arange(len(md_cases))
for i, c in enumerate(md_cases):
    nres = c['n']
    for j, t in enumerate(['conservative', 'mismatch-driven', 'reverse']):
        p, lo, hi = wilson(c['counts'][t], nres)
        ax.errorbar(i + (j - 1) * 0.22, p, yerr=[[max(p - lo, 0)], [max(hi - p, 0)]], fmt='o', c=TC[t], capsize=2, ms=6,
                    label=('MD ' + t) if i == 0 else None)
        ax.scatter(i + (j - 1) * 0.22, c['model_thermo'][t], marker='_', s=260, c='k', zorder=4,
                   label='model (thermo $\\lambda$)' if (i == 0 and j == 0) else None)
        ax.scatter(i + (j - 1) * 0.22, c['model_soft'][t], marker='x', s=30, c='k', zorder=4,
                   label='model (soft $\\lambda$/30)' if (i == 0 and j == 0) else None)
        ax.scatter(i + (j - 1) * 0.22, c['model_window'][t], marker='D', s=40, facecolors='none', edgecolors='m', zorder=5,
                   label='T=0 LJ stability-map winner' if (i == 0 and j == 0) else None)
ax.set_xticks(x)
def short(c):
    p = parse(c['name'])
    s = '%s %s δ=%s' % (fam(c['name']), str(tuple(c['seed'])).replace(' ', ''), p['d'])
    if fam(c['name']) == 'E1':
        s += ' T=%s τ=%s' % (p['T'], p['tau'])
    elif p.get('tau') not in ('40', '30', '60'):
        s += ' τ=%s' % p['tau']
    return s + ' (n=%d)' % c['n']
ax.set_xticklabels([short(c) for c in md_cases], fontsize=7.5, rotation=55, ha='right')
ax.set_ylabel('fraction of runs'); ax.set_ylim(-0.03, 1.08)
ax.legend(fontsize=7, loc='center left', bbox_to_anchor=(1.0, 0.5))
ax.set_title('(iii\') MD step outcomes (95% CI) vs single-step model', fontsize=10)
plt.tight_layout(); plt.savefig(IMG + 'fig9_step_statistics.png', dpi=140); plt.close()

# ------------------------------------------------------------- (iv) snapshots
picks = [('E1_d=0_T=0.22_tau=60_n=10', 0), ('E3a_d=0.22_T=0.22_tau=60_n=4_N=32', 0),
         ('E3b_d=0.09_T=0.22_tau=40_n=3', 0), ('E3b_d=0.09_T=0.22_tau=40_n=3', 2),
         ('E2_d=0.03_T=0.22_tau=40_n=2', 0), ('E2_d=0.08_T=0.22_tau=40_n=2', 0),
         ('E2b_d=0.08_T=0.22_tau=30_n=2_N=192', 0), ('E2b_d=0.02_T=0.22_tau=30_n=1_N=212', 0)]
picks = [p for p in picks if p[0] in summary and any(r['seed'] == p[1] for r in summary[p[0]]['runs'])]
import md_growth as md
fig = plt.figure(figsize=(4.2 * len(picks), 4.6))
for i, (n, sd) in enumerate(picks):
    js = json.load(open('../outputs/growth/%s_s%d.json' % (n, sd)))
    X = np.array(js['X_anneal']); D = np.array(js['D'])
    c = md.auto_centre(X) if fam(n) in ('E1', 'E3a', 'E3c') else 0
    R, rms = md.core_frame(X, c)
    Y = (X - X[c]) @ R.T
    # rotate so that the five-fold axis ICO_V[0] points to the viewer (z)
    v = ic.ICO_V[0]; zax = np.array([0, 0, 1.0])
    ax_ = np.cross(v, zax); s_ = np.linalg.norm(ax_); c_ = v @ zax
    K = np.array([[0, -ax_[2], ax_[1]], [ax_[2], 0, -ax_[0]], [-ax_[1], ax_[0], 0]])
    Rz = np.eye(3) + K + K @ K * ((1 - c_) / s_ ** 2)
    Y = Y @ Rz.T
    a = fig.add_subplot(1, len(picks), i + 1)
    o = np.argsort(Y[:, 2])
    uD = np.unique(np.round(D, 4))
    cm = plt.get_cmap('viridis')
    cols = {d: cm(j / max(len(uD) - 1, 1)) for j, d in enumerate(uD)}
    from matplotlib.patches import Circle
    zmin, zmax = Y[:, 2].min(), Y[:, 2].max()
    for k in o:
        if Y[k, 2] < 0:
            continue
        shade = 0.45 + 0.55 * (Y[k, 2] - 0) / (zmax + 1e-9)
        col = np.array(cols[np.round(D[k], 4)][:3]) * shade
        a.add_patch(Circle((Y[k, 0], Y[k, 1]), 0.5 * D[k], fc=np.clip(col, 0, 1), ec='k', lw=0.3))
    Rm = np.abs(Y[:, :2]).max() + 1
    a.set_xlim(-Rm, Rm); a.set_ylim(-Rm, Rm)
    st = summary[n]['runs'][[r['seed'] for r in summary[n]['runs']].index(sd)]['id_anneal']
    chi = ', '.join('%s=%+.2f' % (k, v) for k, v in st['chi'].items())
    a.set_title('%s s%d  $\\delta$=%s\nbest %s (%s, cov %.2f)\n%s' % (fam(n), sd, parse(n).get('d', 'seq'), st['best'],
                ic.shell_class(*tup(str(st['best']))), st['cov'][str(tuple(st['best']))] if str(tuple(st['best'])) in st['cov'] else np.nan, chi),
                fontsize=8)
    a.set_aspect('equal'); a.axis('off')
fig.suptitle('(iv) annealed grown clusters viewed down a five-fold axis (front hemisphere); colour = species diameter (dark small -> yellow large)', fontsize=10)
plt.tight_layout(rect=(0, 0, 1, 0.92)); plt.savefig(IMG + 'fig10_snapshots.png', dpi=130); plt.close()
for c in md_cases:
    print(c['name'], c['counts'], {k: round(v, 2) for k, v in c['model_thermo'].items()})
