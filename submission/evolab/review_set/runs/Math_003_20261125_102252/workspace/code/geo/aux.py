"""Auxiliary-point candidate generation, realisation and features."""
import itertools
import math
import random

import numpy as np

from . import numeric as N
from .problem import DEFS, Problem

AUX_NAMES = ('aux', 'aux2', 'aux3')

FAMILIES = ['midpoint', 'mirror', 'foot', 'reflect', 'circumcenter', 'orthocenter', 'incenter',
            'parallelogram', 'line_line', 'line_circle', 'pline_line', 'tline_line', 'circle_circle',
            'tangent_line']


def realize(constructs, P, rng=None):
    """All candidate coordinates for a clause given as list of (cname, full_args)."""
    rng = rng or random.Random(0)
    loci, pts = [], None
    for cname, full in constructs:
        kind, val = N.sketch(cname, full, P, rng)
        if kind == 'pts':
            pts = val
        else:
            loci += val
    if pts is not None:
        return pts[:1]
    if len(loci) < 2:
        return []
    return N.intersect(loci[0], loci[1])


def statement_lines_circles(prob, P):
    lines, circles = set(), set()
    for ci, cname, pred, args in prob.premises():
        if pred == 'coll':
            for a, b in itertools.combinations(args, 2):
                lines.add(tuple(sorted((a, b))))
        elif pred in ('para', 'perp'):
            lines.add(tuple(sorted(args[:2])))
            lines.add(tuple(sorted(args[2:4])))
        elif pred == 'cong':
            a, b, c, d = args
            if a == c:
                circles.add((a, b)); circles.add((a, d))
            elif a == d:
                circles.add((a, b)); circles.add((a, c))
            elif b == c:
                circles.add((b, a)); circles.add((b, d))
            elif b == d:
                circles.add((b, a)); circles.add((b, c))
    for cname in ():
        pass
    return lines, circles


def generate(prob, P, name='aux', quota=None, seed=0, extra_lines=None):
    """Return list of dict(family, clause, args, z) deduplicated by coordinates."""
    if name in P:
        raise ValueError('aux name %s collides with an existing point' % name)
    pts = list(P.keys())
    Z = {k: P[k] for k in pts}
    goal_pts = set(prob.goal[1]) if prob.goal else set()
    slines, scircles = statement_lines_circles(prob, P)
    pairs = [tuple(sorted(p)) for p in itertools.combinations(pts, 2)]
    out = []
    seen = {}
    for k, z in Z.items():
        seen[(round(z.real, 6), round(z.imag, 6))] = k
    scale = max(abs(z) for z in Z.values()) + 1

    def add(fam, clause, args):
        try:
            prob_cons = _parse_clause(clause)
            cands = realize(prob_cons, Z)
        except (N.DegenerateError, ZeroDivisionError, ValueError, NotImplementedError):
            return
        for z in cands:
            if not (abs(z) < 4 * scale):
                continue
            if min(abs(z - w) for w in Z.values()) < 1e-3:
                continue
            key = (round(z.real, 6), round(z.imag, 6))
            if key in seen:
                continue
            seen[key] = clause
            out.append(dict(family=fam, clause=clause, args=list(args), z=z))

    for a, b in pairs:
        add('midpoint', '%s = midpoint %s %s %s' % (name, name, a, b), (a, b))
    for a, b in itertools.permutations(pts, 2):
        add('mirror', '%s = mirror %s %s %s' % (name, name, a, b), (a, b))
    for a in pts:
        for b, c in pairs:
            if a in (b, c) or N.n_coll(Z[a], Z[b], Z[c], 1e-6):
                continue
            add('foot', '%s = foot %s %s %s %s' % (name, name, a, b, c), (a, b, c))
            add('reflect', '%s = reflect %s %s %s %s' % (name, name, a, b, c), (a, b, c))
    for a, b, c in itertools.combinations(pts, 3):
        if N.n_coll(Z[a], Z[b], Z[c], 1e-4):
            continue
        add('circumcenter', '%s = circle %s %s %s %s' % (name, name, a, b, c), (a, b, c))
        add('orthocenter', '%s = orthocenter %s %s %s %s' % (name, name, a, b, c), (a, b, c))
        add('incenter', '%s = incenter %s %s %s %s' % (name, name, a, b, c), (a, b, c))
    for a, b, c in itertools.permutations(pts, 3):
        if N.n_coll(Z[a], Z[b], Z[c], 1e-4):
            continue
        add('parallelogram', '%s = parallelogram %s %s %s %s' % (name, a, b, c, name), (a, b, c))
    # lines: statement lines first, then all pairs
    L = sorted(slines) + [p for p in pairs if p not in slines]
    Ls = sorted(slines) if len(pts) > 13 else L
    for l1, l2 in itertools.combinations(L, 2):
        if l1 not in slines and l2 not in slines and len(pts) > 11:
            continue
        add('line_line', '%s = on_line %s %s %s, on_line %s %s %s' % (name, name, l1[0], l1[1], name, l2[0], l2[1]),
            l1 + l2)
    circles = set(scircles)
    for o in pts:
        for x in pts:
            if x != o and (o, x) not in circles:
                # circles centred at points with at least two equidistant points
                d = abs(Z[x] - Z[o])
                if sum(1 for y in pts if y not in (o, x) and abs(abs(Z[y] - Z[o]) - d) < 1e-7) >= 1:
                    circles.add((o, x))
    canon_c = {}
    for o, x in sorted(circles):
        canon_c.setdefault((o, round(abs(Z[x] - Z[o]), 6)), (o, x))
    circles = sorted(canon_c.values())
    for o, x in circles:
        for l in L:
            add('line_circle', '%s = on_line %s %s %s, on_circle %s %s %s' % (name, name, l[0], l[1], name, o, x),
                l + (o, x))
    for (o1, x1), (o2, x2) in itertools.combinations(circles, 2):
        if o1 == o2:
            continue
        add('circle_circle', '%s = on_circle %s %s %s, on_circle %s %s %s' % (name, name, o1, x1, name, o2, x2),
            (o1, x1, o2, x2))
    SL = sorted(slines)
    for a in pts:
        for b, c in SL:
            if a in (b, c):
                continue
            for l in SL:
                if l == (b, c) or a in l:
                    continue
                add('pline_line', '%s = on_pline %s %s %s %s, on_line %s %s %s' % (name, name, a, b, c, name, l[0], l[1]),
                    (a, b, c) + l)
                add('tline_line', '%s = on_tline %s %s %s %s, on_line %s %s %s' % (name, name, a, b, c, name, l[0], l[1]),
                    (a, b, c) + l)
    for o, x in circles:
        for l in Ls:
            add('tangent_line', '%s = on_tline %s %s %s %s, on_line %s %s %s' % (name, name, x, o, x, name, l[0], l[1]),
                (x, o) + l)
    if quota is not None:
        rng = random.Random(seed)
        byf = {}
        for c in out:
            byf.setdefault(c['family'], []).append(c)
        out = []
        for fam in FAMILIES:
            lst = byf.get(fam, [])
            if len(lst) > quota:
                lst = rng.sample(lst, quota)
            out += lst
    return out


def _parse_clause(clause):
    lhs, rhs = clause.split('=', 1)
    cons = []
    for part in rhs.split(','):
        t = part.split()
        cons.append((t[0], t[1:]))
    return cons


# ------------------------------------------------------------------ features
FEAT_NAMES = None


def features(cands, prob, P, names_order=None):
    """Hand-crafted features (problem-agnostic) for each candidate."""
    pts = list(P.keys())
    Z = np.array([P[k] for k in pts])
    n = len(pts)
    goal_pts = set(prob.goal[1]) if prob.goal else set()
    order = {p: i for i, p in enumerate(pts)}
    late = set(pts[-3:])
    scale = np.median(np.abs(Z[:, None] - Z[None, :])[np.triu_indices(n, 1)])
    iu = np.triu_indices(n, 1)
    # existing segment lengths
    seglen = np.abs(Z[:, None] - Z[None, :])[iu]
    F = []
    for c in cands:
        z = c['z']
        f = [1.0 if c['family'] == fam else 0.0 for fam in FAMILIES]
        args = c['args']
        f.append(sum(a in goal_pts for a in args) / max(1, len(args)))
        f.append(float(any(a in goal_pts for a in args)))
        f.append(sum(a in late for a in args) / max(1, len(args)))
        f.append(np.mean([order[a] for a in args]) / n)
        f.append(len(set(args)) / 6.0)
        dz = Z - z
        dist = np.abs(dz)
        # collinear with pairs
        cr = (dz[:, None].conj() * dz[None, :]).imag[iu] / (dist[:, None] * dist[None, :])[iu]
        ncol = float(np.sum(np.abs(cr) < 1e-7))
        f.append(math.log1p(ncol))
        # equidistant pairs
        eqd = float(np.sum(np.abs(dist[:, None] - dist[None, :])[iu] < 1e-7 * scale))
        f.append(math.log1p(eqd))
        # right angles at z
        dot = (dz[:, None].conj() * dz[None, :]).real[iu] / (dist[:, None] * dist[None, :])[iu]
        f.append(math.log1p(float(np.sum(np.abs(dot) < 1e-7))))
        # length equal to an existing segment
        eql = float(np.sum(np.abs(dist[:, None] - seglen[None, :]) < 1e-7 * scale))
        f.append(math.log1p(eql))
        # concyclic with triples
        f.append(math.log1p(_n_concyclic(z, Z)))
        # distance to goal points (normalised)
        gd = [abs(z - P[g]) for g in goal_pts if g in P]
        f.append(min(gd) / scale if gd else 1.0)
        f.append(min(np.min(dist) / scale, 3.0))
        f.append(min(abs(z - Z.mean()) / scale, 5.0))
        f.append(float(len(cands)) ** 0 * math.log(n))
        F.append(f)
    return np.array(F, float)


def _n_concyclic(z, Z):
    n = len(Z)
    if n < 3:
        return 0.0
    idx = np.array(list(itertools.combinations(range(n), 3)))
    if len(idx) > 3000:
        idx = idx[np.random.RandomState(0).choice(len(idx), 3000, replace=False)]
    a, b, c = Z[idx[:, 0]], Z[idx[:, 1]], Z[idx[:, 2]]
    ax, ay, bx, by, cx, cy = a.real, a.imag, b.real, b.imag, c.real, c.imag
    d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    ok = np.abs(d) > 1e-9
    d = np.where(ok, d, 1)
    ux = ((ax * ax + ay * ay) * (by - cy) + (bx * bx + by * by) * (cy - ay) + (cx * cx + cy * cy) * (ay - by)) / d
    uy = ((ax * ax + ay * ay) * (cx - bx) + (bx * bx + by * by) * (ax - cx) + (cx * cx + cy * cy) * (bx - ax)) / d
    o = ux + 1j * uy
    r = np.abs(a - o)
    return float(np.sum(ok & (np.abs(np.abs(z - o) - r) < 1e-7 * np.maximum(1, r))))


FEAT_NAMES = FAMILIES + ['frac_goal_args', 'any_goal_arg', 'frac_late_args', 'mean_arg_order', 'n_distinct_args',
                         'log_ncoll_pairs', 'log_equidist_pairs', 'log_right_angles', 'log_eq_lengths',
                         'log_concyclic', 'dist_goal', 'dist_nearest', 'dist_centroid', 'log_npoints']


def heuristic_score(c, prob):
    """Hand-designed ordering (no learning): simple constructions on goal points first."""
    base = {'midpoint': 5, 'foot': 4.5, 'circumcenter': 4.5, 'mirror': 4, 'reflect': 4, 'orthocenter': 3.5,
            'incenter': 3, 'parallelogram': 3, 'line_line': 2.5, 'line_circle': 2.5, 'circle_circle': 2,
            'tangent_line': 1.5, 'pline_line': 1.0, 'tline_line': 1.0}[c['family']]
    goal_pts = set(prob.goal[1]) if prob.goal else set()
    g = sum(a in goal_pts for a in c['args'])
    return base + 0.8 * g - 0.1 * len(c['args'])
