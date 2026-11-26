"""Fig 4: real-element design chart (dataset radii) + Gupta sensitivity."""
import json, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import icoshell as ic

IMG = '../report/images/'
G = json.load(open('../outputs/gupta_results.json'))
DS = json.load(open('../outputs/design.json'))
LJ = json.load(open('../outputs/scans_LJ.json'))
WIN = json.load(open('../outputs/stability_windows.json'))
TCOL = {'conservative': '#1f77b4', 'class-raising': '#2ca02c', 'class-lowering': '#d62728', 'non-NN': '#999999'}

fig, axs = plt.subplots(1, 3, figsize=(19, 5.8), gridspec_kw=dict(width_ratios=[1.1, 1.15, 0.9]))
# (a) design chart: LJ stability windows for Mackay cores 13/55/147 and the zig-zag cores, with element pairs
ax = axs[0]
cores = ['c01', 'c02', 'c03', 'c11', 'c12', 'c22', 'c13']
for i, core in enumerate(cores):
    for name, lo, hi in WIN[core]:
        r = LJ[core]['results'][name] if name in LJ[core]['results'] else LJ['c03b']['results'][name]
        ax.barh(i, hi - lo + 0.02, left=lo - 0.01, color=TCOL[r['step']], alpha=0.35, edgecolor='k', lw=0.3)
        hk = tuple(r['shell'])
        ax.text((lo + hi) / 2, i, '%s\n%s' % (name.replace(' ', ''), ic.shell_class(*hk)), ha='center', va='center', fontsize=6.5)
labels = []
for core in cores:
    p = LJ[core]['core_path']
    labels.append('core %s N=%d' % (tuple(p[-1]), sum(ic.N_of(*q) for q in p)))
ax.set_yticks(range(len(cores))); ax.set_yticklabels(labels, fontsize=8)
pairs = sorted([p for p in DS['pairs'] if p['delta'] < 0.37], key=lambda p: p['delta'])
for j, p in enumerate(pairs):
    ax.axvline(p['delta'], c='k', lw=0.6, ls=':')
    ax.text(p['delta'], len(cores) - 0.4 + 0.33 * (j % 3), '%s/%s' % (p['shell'], p['core']), rotation=0, fontsize=6.5, ha='center')
ax.set_xlim(-0.02, 0.38); ax.set_ylim(-0.6, len(cores) + 0.8)
ax.set_xlabel('mismatch $\\delta$ = r$_{out}$/r$_{in}$ - 1')
ax.set_title('(a) LJ winning outer shell vs $\\delta$ (shaded windows)\n dotted: element pairs from dataset radii', fontsize=10)

# (b) Gupta Delta per candidate for both cross-parameter sets
ax = axs[1]
cases = [c for c in G['cases']]
groups = []
for c in cases:
    key = (c['core'], c['shell'], c['Ncore'])
    if key not in groups:
        groups.append(key)
x0 = 0; xt, xl = [], []
for key in groups:
    cs = [c for c in cases if (c['core'], c['shell'], c['Ncore']) == key]
    outs = []
    for c in cs:
        if tuple(c['outer']) not in outs:
            outs.append(tuple(c['outer']))
    for k, o in enumerate(outs):
        for m, (mode, hatch) in enumerate([('avg', ''), ('lit', '//')]):
            cc = [c for c in cs if tuple(c['outer']) == o and c['mode'] == mode][0]
            best = min(c2['Delta'] for c2 in cs if c2['mode'] == mode)
            col = TCOL[cc['step']]
            ax.bar(x0 + k * 0.5 + m * 0.24, cc['Delta'] - best, 0.22, color=col, hatch=hatch, edgecolor='k', lw=0.4,
                   alpha=0.4 if cc['shift'] > 3 else 1.0)
            if cc['Delta'] == best:
                ax.text(x0 + k * 0.5 + m * 0.24, 0.02, '*', ha='center', fontsize=11)
        xt.append(x0 + k * 0.5 + 0.12); xl.append('%s%d' % (str(o).replace(' ', ''), ic.N_of(*o)))
    ax.text(x0 + 0.25 * (len(outs) - 1) + 0.12, -0.2, '%s%d@%s' % (key[0], key[2], key[1]), ha='right', fontsize=7,
            transform=ax.get_xaxis_transform(), weight='bold', rotation=35)
    x0 += 0.5 * len(outs) + 0.6
ax.set_xticks(xt); ax.set_xticklabels(xl, rotation=90, fontsize=6.5)
ax.set_ylabel('$\\Delta$ - min$\\Delta$  (eV)')
from matplotlib.patches import Patch
hs = [Patch(fc='w', ec='k', label='avg cross params'), Patch(fc='w', ec='k', hatch='//', label='"lit" cross params')]
hs += [Patch(fc=TCOL[t], label=t) for t in ['conservative', 'class-raising', 'non-NN']]
ax.legend(handles=hs, fontsize=7, loc='upper left')
ax.set_title('(b) Gupta: excess energy relative to best shell (*: best;\n faded: outer shell reconstructed >3$^\\circ$)', fontsize=10)
# (c) task example: Ni147@Ag132 vs Ni147@Ag192 (and others), both metrics, both sets
ax = axs[2]
sel = [c for c in cases if c['core'] == 'Ni' and c['shell'] == 'Ag' and c['Ncore'] == 147]
outs = []
for c in sel:
    if tuple(c['outer']) not in outs:
        outs.append(tuple(c['outer']))
for m, mode in enumerate(['avg', 'lit']):
    y = [[c for c in sel if tuple(c['outer']) == o and c['mode'] == mode][0] for o in outs]
    ax.plot(range(len(outs)), [c['Delta'] for c in y], '-o', label='$\\Delta$, %s' % mode)
    ax.plot(range(len(outs)), [c['eps_shell'] * 5 + 1 for c in y], '--s', label='5$\\times\\epsilon_{shell}$+1, %s' % mode)
ax.set_xticks(range(len(outs))); ax.set_xticklabels(['Ni147@Ag%d\n%s %s' % (ic.N_of(*o), o, ic.shell_class(*o)) for o in outs], fontsize=8)
ax.set_ylabel('eV (lower = more stable)'); ax.legend(fontsize=7.5)
ax.set_title('(c) task example: Ni$_{147}$@Ag$_{192}$ vs alternatives (Gupta)', fontsize=10)
plt.tight_layout(); plt.savefig(IMG + 'fig4_elements_gupta.png', dpi=140); plt.close()

# Gupta vs nearest-delta* rule agreement
RAD = dict([('Ag', 1.44), ('Cu', 1.28), ('Ni', 1.24)])
import lattice_model as lm
agree = []
for key in groups:
    core, shl, Nc = key
    dl = RAD[shl] / RAD[core] - 1
    seed = {13: (0, 1), 55: (0, 2), 147: (0, 3)}[Nc]
    nn = [tuple(c['outer']) for c in cases if (c['core'], c['shell'], c['Ncore']) == key and c['mode'] == 'avg' and c['step'] != 'non-NN']
    pred = min(nn, key=lambda o: abs(lm.dstar(seed, o)[0] - dl))
    import design
    win = design.window_rule(seed, dl)
    for mode in ['avg', 'lit']:
        cs = [c for c in cases if (c['core'], c['shell'], c['Ncore']) == key and c['mode'] == mode]
        bD = tuple(min(cs, key=lambda c: c['Delta'])['outer'])
        bE = tuple(min(cs, key=lambda c: c['eps_shell'])['outer'])
        agree.append(dict(core=core, shell=shl, Ncore=Nc, delta=dl, mode=mode, rule=pred, window=win, best_Delta=bD, best_eps=bE,
                          ok_Delta=bool(bD == pred), ok_eps=bool(bE == pred),
                          okw_Delta=bool(bD == win), okw_eps=bool(bE == win)))
json.dump(agree, open('../outputs/gupta_vs_rule.json', 'w'), indent=1, default=str)
for a in agree:
    print(a)
print('window agreement Delta:', {m: sum(a['okw_Delta'] for a in agree if a['mode'] == m) for m in ['avg', 'lit']},
      'eps:', {m: sum(a['okw_eps'] for a in agree if a['mode'] == m) for m in ['avg', 'lit']})
print('agreement Delta:', {m: sum(a['ok_Delta'] for a in agree if a['mode'] == m) for m in ['avg', 'lit']},
      'eps:', {m: sum(a['ok_eps'] for a in agree if a['mode'] == m) for m in ['avg', 'lit']}, 'of', len(groups))
