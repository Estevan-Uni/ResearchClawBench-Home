"""Fig 1: hexagonal (h,k) lattice coloured by chirality class (dataset shell_colors),
magic-number paths, 3D shell renderings; table (h,k)->T,N,class; convention audit."""
import json, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import icoshell as ic

COL = {'MC': '#1f77b4', 'BG': '#ff7f0e', 'Ch1': '#2ca02c', 'Ch2': '#d62728', 'Ch3': '#9467bd',
       'Ch4': '#8c564b', 'Ch5': '#e377c2', 'C': 'k'}


def xy(h, k):
    return h * ic.A1 + k * ic.A2


fig = plt.figure(figsize=(15, 7.4))
ax = fig.add_axes([0.04, 0.27, 0.44, 0.66])
for h in range(0, 6):
    for k in range(0, 6):
        p = xy(h, k)
        c = ic.shell_class(h, k)
        hd = ic.hand(h, k)
        ax.scatter(*p, s=420, c=COL[c] if hd >= 0 else 'white', edgecolors=COL[c], linewidths=2.5, zorder=3)
        ax.text(p[0], p[1] + 0.27, '(%d,%d)' % (h, k), ha='center', fontsize=7.5)
        ax.text(p[0], p[1], str(ic.N_of(h, k)), ha='center', va='center', fontsize=6.5,
                color='white' if hd >= 0 and c != 'C' else 'k', zorder=4)
# paths
mk = np.array([xy(0, k) for k in range(5)])
zz = np.array([xy(*p) for p in [(0, 0), (0, 1), (1, 1), (1, 2), (2, 2), (2, 3)]])
alt = np.array([xy(*p) for p in [(0, 0), (0, 1), (0, 2), (0, 3), (1, 3), (2, 3)]])
ax.plot(mk[:, 0], mk[:, 1], '-', color='#1f77b4', lw=3, alpha=0.5, zorder=1, label='Mackay path (0,k): 1,13,55,147,309')
ax.plot(zz[:, 0], zz[:, 1], '-', color='#ff7f0e', lw=3, alpha=0.6, zorder=1,
        label='zig-zag (0,0)(0,1)(1,1)(1,2)(2,2)(2,3): 1,13,45,117,239,431')
ax.plot(alt[:, 0] + 0.05, alt[:, 1], '--', color='#d62728', lw=2, alpha=0.6, zorder=1,
        label='Mackay core + chiral line: ...147@132@192')
for c in ['MC', 'BG', 'Ch1', 'Ch2', 'Ch3', 'Ch4']:
    ax.scatter([], [], s=120, c=COL[c], label=c)
ax.scatter([], [], s=120, c='white', edgecolors='gray', linewidths=2, label='hollow: h>k (opposite hand)')
ax.legend(loc='upper center', fontsize=8, framealpha=0.9, ncol=3, bbox_to_anchor=(0.5, -0.12))
ax.set_aspect('equal'); ax.set_xlim(-0.8, 8.6); ax.set_ylim(-0.6, 5.0)
ax.set_xlabel('h a$_1$ + k a$_2$  (x)'); ax.set_ylabel('y')
ax.set_title('(a) Shell lattice: (h,k), N=10T+2, class = Ch$_{min(h,k)}$ (MC: min=0, BG: h=k)', fontsize=10)
# 3D renderings
shells = [(0, 2), (1, 1), (1, 2), (2, 1), (2, 2), (1, 3)]
for i, (h, k) in enumerate(shells):
    a3 = fig.add_axes([0.50 + (i % 3) * 0.165, 0.50 - (i // 3) * 0.46, 0.16, 0.40], projection='3d')
    X = ic.shell(h, k)
    vm = ic.vertex_mask(X)
    # view down a 5-fold axis
    a3.scatter(*X[~vm].T, s=18, c=COL[ic.shell_class(h, k)], depthshade=True)
    a3.scatter(*X[vm].T, s=30, c='k')
    a3.view_init(elev=90 - np.degrees(np.arccos(ic.ICO_V[0, 2])), azim=np.degrees(np.arctan2(ic.ICO_V[0, 1], ic.ICO_V[0, 0])))
    a3.set_axis_off()
    a3.set_title('(%d,%d) %s  N=%d' % (h, k, ic.shell_class(h, k), len(X)) +
                 ('' if ic.hand(h, k) == 0 else (' hand %+d' % ic.hand(h, k))), fontsize=9)
fig.text(0.51, 0.975, '(b) ideal CK shells (black: 12 five-fold vertices)', fontsize=10)
plt.savefig('../report/images/fig1_lattice_classes.png', dpi=150)

# ---- table and audit
rows = []
for h in range(0, 6):
    for k in range(h, 6):
        if (h, k) == (0, 0):
            continue
        rows.append(dict(hk=[h, k], T=ic.T_of(h, k), N=ic.N_of(h, k), cls=ic.shell_class(h, k),
                         chiral=ic.hand(h, k) != 0))
seqs = dict(mackay=[1] + list(np.cumsum([ic.N_of(0, k) for k in range(1, 5)]) + 1),
            zigzag=list(np.cumsum([ic.N_of(*p) for p in [(0, 0), (0, 1), (1, 1), (1, 2), (2, 2), (2, 3)]])))
ds_mackay = [1, 13, 55, 147, 309]; ds_b5 = [1, 13, 45, 117, 239, 431]


def conv(h, k, which):
    m, n = min(h, k), max(h, k)
    if which == 'A':      # adopted: Ch_min
        return 'MC' if m == 0 else ('BG' if m == n else 'Ch%d' % m)
    if which == 'B':      # distance from the BG line
        return 'MC' if m == 0 else ('BG' if m == n else 'Ch%d' % (n - m))
    if which == 'C':      # pure anti-Mackay count, BG not separated
        return 'MC' if m == 0 else 'Ch%d' % m


def shell_of_N(N):
    for h in range(8):
        for k in range(h, 8):
            if (h, k) != (0, 0) and 10 * ic.T_of(h, k) + 2 == N:
                return (h, k)
    return None


items = [('multicomponent_clusters Na13@Rb32 outer', 32, 'Ch1'),
         ('multicomponent_clusters K13@Cs42 outer', 42, 'Ch2'),
         ('multicomponent_clusters Ag13@Cu45 outer', 45, 'Ch1'),
         ('initial_seeds Na13@Rb32', 32, 'Ch1'),
         ('core 13 of all clusters (MC)', 12, 'MC')]
audit = []
for name, N, lab in items:
    s = shell_of_N(N)
    audit.append(dict(item=name, N=N, label=lab, hk=s,
                      **{w: (conv(*s, w) if s else 'N not 10T+2') for w in 'ABC'},
                      **{'ok_' + w: bool(s and conv(*s, w) == lab) for w in 'ABC'}))
# mismatch-type labels: which NN steps (with LJ-like formula delta*) have the stated class change
SC = json.load(open('../outputs/scans_LJ.json')) if __import__('os').path.exists('../outputs/scans_LJ.json') else {}
steps = []
for h in range(0, 5):
    for k in range(h, 5):
        if (h, k) == (0, 0):
            continue
        for c in ic.outward_nn_steps(h, k):
            steps.append(((h, k), c, ic.delta_star(ic.T_of(h, k), ic.T_of(*c))))
mm_items = [('optimal_mismatch_ranges MC->MC', 'MC', 'MC', 0.03, 0.05),
            ('optimal_mismatch_ranges MC->Ch1', 'MC', 'Ch1', 0.12, 0.16),
            ('optimal_mismatch_ranges MC->Ch2', 'MC', 'Ch2', 0.19, 0.22),
            ('optimal_mismatch_ranges MC->BG', 'MC', 'BG', 0.08, 0.10),
            ('mismatch_params Ch1->Ch2 0.21', 'Ch1', 'Ch2', 0.19, 0.23)]
for name, a, b, lo, hi in mm_items:
    rec = dict(item=name, range=[lo, hi])
    for w in 'ABC':
        cand = [(list(s), list(c), round(float(d), 3)) for s, c, d in steps
                if conv(*s, w) == a and conv(*c, w) == b]
        rec[w] = cand
        rec['ok_' + w] = any(lo - 0.02 <= d <= hi + 0.02 for _, _, d in cand)
    audit.append(rec)
score = {w: sum(r['ok_' + w] for r in audit) for w in 'ABC'}
json.dump(dict(table=rows, sequences={k: [int(x) for x in v] for k, v in seqs.items()},
               dataset_sequences_match=dict(mackay=[int(x) for x in seqs['mackay']] == ds_mackay,
                                            b5=[int(x) for x in seqs['zigzag']] == ds_b5),
               audit=audit, score=score, n_items=len(audit)),
          open('../outputs/lattice_table_audit.json', 'w'), indent=1, default=str)
print(seqs, score, len(audit))
for r in audit:
    print(r)
