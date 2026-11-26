import json, numpy as np
from figstyle import *
from gen2004 import VARIANTS
from geo.problem import Problem, build_diagram
from geo.numeric import circumcenter
import matplotlib.patches as mp

def draw(ax, P, title, variant):
    def seg(a, b, ext=0.0, **kw):
        A, B = P[a], P[b]; d = B - A
        A2, B2 = A - ext * d, B + ext * d
        ax.plot([A2.real, B2.real], [A2.imag, B2.imag], **kw)
    def circ(c, r, **kw):
        ax.add_patch(mp.Circle((c.real, c.imag), r, fill=False, **kw))
    def line_through(ks, **kw):
        a0, b0 = P[ks[0]], P[ks[1]]
        u = (b0 - a0) / abs(b0 - a0)
        ts = [((P[k] - a0) * u.conjugate()).real for k in ks]
        t0, t1 = min(ts), max(ts); m = 0.1 * (t1 - t0)
        z0, z1 = a0 + u * (t0 - m), a0 + u * (t1 + m)
        ax.plot([z0.real, z1.real], [z0.imag, z1.imag], **kw)
    line_through(['a', 'b', 'm'], color=MUTED, lw=1.2); line_through(['a', 'c', 'n'], color=MUTED, lw=1.2)
    zs = sorted([P['b'], P['c'], P['p']], key=lambda z: ((z - P['b']) * (P['c'] - P['b']).conjugate()).real)
    dd = zs[-1] - zs[0]
    ax.plot([(zs[0] - 0.08 * dd).real, (zs[-1] + 0.08 * dd).real], [(zs[0] - 0.08 * dd).imag, (zs[-1] + 0.08 * dd).imag], color=INK, lw=1.6)
    circ(P['o'], abs(P['o'] - P['m']), color=SERIES[0], lw=1.8)
    for o, x in (('o1', 'b'), ('o2', 'c')):
        circ(P[o], abs(P[o] - P[x]), color=SERIES[1], lw=1.5)
    ca = circumcenter(P['a'], P['m'], P['n'])
    circ(ca, abs(ca - P['a']), color=SERIES[2], lw=1.3, ls='--')
    seg('a', 'r', 0.1, color=INK2, lw=1, ls=':'); seg('o', 'r', 0.1, color=INK2, lw=1, ls=':')
    for k in ['a', 'b', 'c', 'o', 'm', 'n', 'r', 'p']:
        z = P[k]; ax.plot(z.real, z.imag, 'o', ms=6, color=INK, zorder=5)
        ax.annotate(k.upper(), (z.real, z.imag), xytext=(5, 5), textcoords='offset points', fontsize=11, fontweight='bold')
    ax.plot(ca.real, ca.imag, 's', ms=6, color=SERIES[2], zorder=5)
    ax.annotate('aux', (ca.real, ca.imag), xytext=(5, -12), textcoords='offset points', color=INK2)
    ax.set_aspect('equal'); ax.set_title(title, fontsize=10.5); ax.grid(False)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_visible(False)
    xs = [P[k].real for k in 'abcomnrp']; ys = [P[k].imag for k in 'abcomnrp']
    pad = 0.6
    ax.set_xlim(min(xs) - pad, max(xs) + pad); ax.set_ylim(min(ys) - pad, max(ys) + pad)

fig, axs = plt.subplots(1, 2, figsize=(13, 6.2))
def best(text):
    import itertools
    bestq, bestP = -1, None
    for s in range(300):
        try: P = build_diagram(Problem('v', text), seed=s, max_tries=5)
        except Exception: continue
        pts = [P[k] for k in 'abcomnrp']
        ext = max(abs(x - y) for x, y in itertools.combinations(pts, 2))
        q = min(abs(x - y) for x, y in itertools.combinations(pts, 2)) / ext
        if q > bestq: bestq, bestP = q, P
    return bestP
P0 = best(VARIANTS['V0_original'])
P3 = best(VARIANTS['V3_only_OM_eq_ON'])
draw(axs[0], P0, 'Original IMO 2004 P1: O = midpoint of BC,\nM, N on the circle with diameter BC', 'V0')
draw(axs[1], P3, 'Generalisation (verified): O arbitrary, M on AB arbitrary,\nN on AC with ON = OM (circle need not pass through B or C)', 'V3')
fig.text(0.5, 0.02, 'blue: circle centred O through M, N   orange: circles (BMR), (CNR)   green dashed: circle (AMN), centre = aux point used by the proof   '
         'In both, the second intersection P of (BMR) and (CNR) lies on line BC.', ha='center', color=INK2, fontsize=9)
plt.savefig('/workspace/report/images/fig7_imo2004p1_generalisation.png')
