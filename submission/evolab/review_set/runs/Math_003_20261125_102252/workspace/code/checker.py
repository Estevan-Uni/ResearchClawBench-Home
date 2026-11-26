"""Independent proof checker.

Does NOT import the prover (package `geo`).  It re-parses the statement with its
own reader of data/defs.txt, recomputes premise predicates, re-derives the linear
rows of every fact, verifies AR steps by exact rational linear algebra and DD steps
against rule schemas.  Non-degeneracy side conditions (ncoll, orientation sign) are
checked numerically on the coordinates stored in the proof (as AlphaGeometry does).
"""
import cmath
import itertools
import json
import math
import os
import sys
from fractions import Fraction as Fr

DEFS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'defs.txt')


# ------------------------------------------------------------ statement side
def read_defs(path=DEFS_PATH):
    L = open(path).read().split('\n')
    out = {}
    for i in range(0, len(L) - 4, 6):
        h = L[i].split()
        if not h:
            continue
        outs = []
        for g in L[i + 1].split(','):
            if ':' in g:
                outs += g.split(':')[0].split()
        preds = []
        for grp in L[i + 3].split(';'):
            for it in grp.split(','):
                it = it.strip()
                if ':' in it:
                    it = it.split(':', 1)[1].strip()
                if it:
                    t = it.split()
                    preds.append((t[0], t[1:]))
        out[h[0]] = (h[1:], outs, preds)
    return out


def statement_premises(text, defs):
    body = text.split('?')[0]
    prem = []
    points = []
    for cl in body.split(';'):
        cl = cl.strip()
        if not cl:
            continue
        lhs, rhs = cl.split('=', 1)
        outs = [t.split('@')[0] for t in lhs.split()]
        for o in outs:
            if o in points:
                raise CheckError('point %s defined twice' % o)
        points += outs
        for part in rhs.split(','):
            t = part.split()
            if not t:
                continue
            dargs, douts, dpreds = defs[t[0]]
            given = t[1:]
            if len(given) != len(dargs):
                miss = len(dargs) - len(given)
                slots = [a for a in dargs if a in douts] or dargs
                full, oi, gi, nf = [], 0, 0, 0
                for a in dargs:
                    if a in slots and nf < miss:
                        full.append(outs[oi]); oi += 1; nf += 1
                    else:
                        full.append(given[gi]); gi += 1
                given = full
            sub = dict(zip(dargs, given))
            for p, pa in dpreds:
                prem.append((p, [sub.get(a, a) for a in pa]))
    goal = text.split('?')[1].split() if '?' in text else None
    return prem, goal, points


class CheckError(Exception):
    pass


# ------------------------------------------------------------ fact semantics
def seg(a, b):
    return tuple(sorted((a, b)))


def comb(terms):
    d = {}
    for k, c in terms:
        d[k] = d.get(k, 0) + c
    return {k: Fr(c) for k, c in d.items() if c != 0}


def rows(kind, a, const=Fr(0)):
    """Rows: (system, {segment: coeff}, const). Angle const mod 1 (units of pi);
    length const in units of log 2."""
    const = Fr(const)
    R = []
    if kind == 'coll':
        for z in a[2:]:
            R.append(('A', comb([(seg(a[0], a[1]), 1), (seg(a[0], z), -1)]), Fr(0)))
            R.append(('A', comb([(seg(a[0], a[1]), 1), (seg(a[1], z), -1)]), Fr(0)))
    elif kind == 'para':
        R.append(('A', comb([(seg(a[0], a[1]), 1), (seg(a[2], a[3]), -1)]), Fr(0)))
    elif kind == 'perp':
        R.append(('A', comb([(seg(a[0], a[1]), 1), (seg(a[2], a[3]), -1)]), Fr(1, 2)))
    elif kind == 'eqangle':
        R.append(('A', comb([(seg(a[2], a[3]), 1), (seg(a[0], a[1]), -1), (seg(a[6], a[7]), -1),
                             (seg(a[4], a[5]), 1)]), const % 1))
    elif kind == 'cong':
        R.append(('L', comb([(seg(a[0], a[1]), 1), (seg(a[2], a[3]), -1)]), Fr(0)))
    elif kind == 'eqratio':
        R.append(('L', comb([(seg(a[0], a[1]), 1), (seg(a[2], a[3]), -1), (seg(a[4], a[5]), -1),
                             (seg(a[6], a[7]), 1)]), Fr(0)))
    elif kind == 'rconst':
        R.append(('L', comb([(seg(a[0], a[1]), 1), (seg(a[2], a[3]), -1)]), const))
    elif kind == 'midp':
        m, x, y = a
        R += rows('coll', [m, x, y]) + rows('cong', [m, x, m, y]) + rows('rconst', [x, y, m, x], 1)
    elif kind == 'cyclic':
        for q in itertools.combinations(a, 4):
            p0, p1, p2, p3 = q
            for (u, v, s, t) in ((p0, p1, p2, p3), (p0, p2, p1, p3), (p0, p3, p1, p2)):
                R += rows('eqangle', [s, u, s, v, t, u, t, v])
    elif kind in ('simtri', 'simtri2'):
        x, y, z, p, q, r = a
        if kind == 'simtri':
            R += rows('eqangle', [x, y, x, z, p, q, p, r]) + rows('eqangle', [y, x, y, z, q, p, q, r])
        else:
            R += rows('eqangle', [x, y, x, z, p, r, p, q]) + rows('eqangle', [y, x, y, z, q, r, q, p])
        R += rows('eqratio', [x, y, p, q, y, z, q, r]) + rows('eqratio', [x, y, p, q, x, z, p, r])
    else:
        raise CheckError('unknown fact kind %s' % kind)
    return [r for r in R if r[1]]


def same_row(r1, r2):
    """Rows equal up to sign (with constants)."""
    if r1[0] != r2[0]:
        return False
    s1, s2 = r1[1], r2[1]
    for sg in (1, -1):
        if s1 == {k: sg * v for k, v in s2.items()}:
            c1, c2 = r1[2], sg * r2[2]
            if r1[0] == 'A':
                if (c1 - c2) % 1 == 0:
                    return True
            elif c1 == c2:
                return True
    return False


def facts_equal(k1, a1, c1, k2, a2, c2):
    if k1 in ('coll', 'cyclic') or k2 in ('coll', 'cyclic'):
        return k1 == k2 and set(a1) == set(a2) and len(set(a1)) == len(a1)
    if k1 == 'midp' or k2 == 'midp':
        return k1 == k2 and a1[0] == a2[0] and set(a1[1:]) == set(a2[1:])
    if k1 in ('simtri', 'simtri2') or k2 in ('simtri', 'simtri2'):
        if k1 != k2:
            return False
        c1_ = set(zip(a1[:3], a1[3:]))
        c2_ = set(zip(a2[:3], a2[3:]))
        return c1_ == c2_ or c1_ == set((y, x) for x, y in c2_)
    R1, R2 = rows(k1, a1, c1), rows(k2, a2, c2)
    if len(R1) != 1 or len(R2) != 1:
        return False
    return same_row(R1[0], R2[0])


def in_span(target, rowlist):
    """Exact test target in Q-span of rowlist. Returns (ok, integral, const_ok)."""
    sysn, tv, tc = target
    rl = [r for r in rowlist if r[0] == sysn]
    keys = sorted(set(k for r in rl for k in r[1]) | set(tv))
    idx = {k: i for i, k in enumerate(keys)}
    m, n = len(rl), len(keys)
    M = [[Fr(0)] * (m + 1) for _ in range(n)]
    for j, r in enumerate(rl):
        for k, v in r[1].items():
            M[idx[k]][j] = v
    for k, v in tv.items():
        M[idx[k]][m] = v
    piv, row = [], 0
    for col in range(m):
        p = next((i for i in range(row, n) if M[i][col] != 0), None)
        if p is None:
            continue
        M[row], M[p] = M[p], M[row]
        inv = 1 / M[row][col]
        M[row] = [x * inv for x in M[row]]
        for i in range(n):
            if i != row and M[i][col] != 0:
                f = M[i][col]
                M[i] = [a - f * b for a, b in zip(M[i], M[row])]
        piv.append(col)
        row += 1
    if any(M[i][m] != 0 for i in range(row, n)):
        return False, False, False
    coef = [Fr(0)] * m
    for i, col in enumerate(piv):
        coef[col] = M[i][m]
    cval = sum(c * r[2] for c, r in zip(coef, rl))
    integral = all(c.denominator == 1 for c in coef)
    if sysn == 'A':
        if integral:
            const_ok = (cval - tc) % 1 == 0
        else:
            d = 1
            for c in coef:
                d = d * c.denominator // math.gcd(d, c.denominator)
            const_ok = ((cval - tc) * d) % 1 == 0  # equal modulo 1/d; branch fixed by the diagram
    else:
        const_ok = cval == tc
    return True, integral, const_ok


# ------------------------------------------------------------ numerics
class Num:
    def __init__(self, coords):
        self.P = {k: complex(*v) for k, v in coords.items()}

    def ncoll(self, a, b, c):
        A, B, C = self.P[a], self.P[b], self.P[c]
        s = max(abs(B - A), abs(C - A))
        return abs(((B - A).conjugate() * (C - A)).imag) / s / s > 1e-6

    def direction(self, a, b):
        return (cmath.phase(self.P[b] - self.P[a]) / math.pi) % 1

    def row_value_ok(self, row):
        sysn, tv, tc = row
        if sysn == 'A':
            v = sum(c * self.direction(*k) for k, c in tv.items())
            t = (v - float(tc)) % 1
            return min(t, 1 - t) < 1e-6
        v = sum(float(c) * math.log(abs(self.P[k[0]] - self.P[k[1]])) for k, c in tv.items())
        return abs(v - float(tc) * math.log(2)) < 1e-6

    def orient(self, a, b, c):
        A, B, C = self.P[a], self.P[b], self.P[c]
        return 1 if ((B - A).conjugate() * (C - A)).imag > 0 else -1


# ------------------------------------------------------------ checker
def check_proof(proof, defs=None, strict_numeric=True):
    defs = defs or read_defs()
    prem, goal, points = statement_premises(proof['statement'], defs)
    num = Num(proof['coords'])
    for p in points:
        if p not in num.P:
            raise CheckError('missing coordinates for %s' % p)
    steps = {}
    info = {'nonintegral_ar': 0, 'n_steps': 0, 'used_premises': []}
    for s in proof['steps']:
        sid, kind, a, c, rule = s['id'], s['kind'], s['args'], Fr(s['const']), s['rule']
        for x in a:
            if x not in num.P:
                raise CheckError('unknown point %s' % x)
        P = []
        for pid in s['prem']:
            if pid not in steps:
                raise CheckError('step %s cites unknown/later step %s' % (sid, pid))
            P.append(steps[pid])
        # every fact must be numerically true (sanity; catches forged facts)
        if kind in ('coll', 'cyclic'):
            pass
        else:
            for r in rows(kind, a, c):
                if strict_numeric and not num.row_value_ok(r):
                    raise CheckError('step %s numerically false: %s %s' % (sid, kind, a))
        check_rule(rule, kind, a, c, P, prem, num, info)
        steps[sid] = (kind, a, c)
        info['n_steps'] += 1
        if rule == 'premise':
            info['used_premises'].append((kind, a))
    last = steps.get(proof['goal_step'])
    if last is None:
        raise CheckError('goal step missing')
    g = proof['goal']
    gk, ga = g[0], g[1:]
    if goal is None or [gk] + ga != goal:
        raise CheckError('goal in proof differs from statement goal')
    gc = Fr(1, 2) if gk == 'perp' else Fr(0)
    if not facts_equal(last[0], last[1], last[2], gk, ga, gc):
        raise CheckError('final fact does not match goal')
    return info


def _find(P, kind, a, c=Fr(0)):
    return any(facts_equal(k, x, cc, kind, a, c) for k, x, cc in P)


def check_rule(rule, kind, a, c, P, prem, num, info):
    if rule == 'premise':
        for pk, pa in prem:
            if pk in ('coll', 'para', 'perp', 'cong', 'eqangle', 'cyclic', 'eqratio'):
                pc = Fr(1, 2) if pk == 'perp' else Fr(0)
                if pk == 'perp':
                    pc = Fr(0)  # rows() adds the 1/2 for perp itself
                if facts_equal(kind, a, c, pk, pa, pc if pk != 'perp' else Fr(0)):
                    return
            elif pk == 'rconst' and kind == 'rconst':
                q = Fr(int(pa[4]), int(pa[5]))
                lg = Fr(round(math.log2(q))) if q > 0 and 2 ** round(math.log2(q)) == q else None
                if lg is not None and facts_equal(kind, a, c, 'rconst', pa[:4], lg):
                    return
        raise CheckError('premise %s %s not in statement' % (kind, a))
    if rule == 'ar':
        if kind not in ('coll', 'para', 'perp', 'eqangle', 'cong', 'eqratio', 'rconst'):
            raise CheckError('AR cannot conclude %s' % kind)
        R = [r for k, x, cc in P for r in rows(k, x, cc)]
        for t in rows(kind, a, c):
            ok, integral, cok = in_span(t, R)
            if not ok:
                raise CheckError('AR step not in span: %s %s' % (kind, a))
            if not cok:
                raise CheckError('AR constant mismatch: %s %s' % (kind, a))
            if not integral and t[0] == 'A':
                info['nonintegral_ar'] += 1
        return
    if rule == 'para_coll':
        if kind == 'coll' and len(set(a)) == 3:
            for x in a:
                y, z = [w for w in a if w != x]
                if _find(P, 'para', [x, y, x, z]):
                    return
        raise CheckError('para_coll schema')
    if rule == 'midp':
        m, x, y = a
        if kind == 'midp' and _find(P, 'coll', [m, x, y]) and _find(P, 'cong', [m, x, m, y]):
            return
        raise CheckError('midp schema')
    if rule in ('iso', 'iso_conv'):
        if rule == 'iso':
            if kind != 'eqangle':
                raise CheckError('iso kind')
            A_, O, B = a[0], a[1], a[3]
            exp = ('eqangle', [A_, O, A_, B, B, A_, B, O])
            if not facts_equal(kind, a, c, *exp, Fr(0)) or not _find(P, 'cong', [O, A_, O, B]):
                raise CheckError('iso schema')
        else:
            if kind != 'cong' or a[0] != a[2]:
                raise CheckError('iso_conv kind')
            O, A_, B = a[0], a[1], a[3]
            if not _find(P, 'eqangle', [A_, O, A_, B, B, A_, B, O]):
                raise CheckError('iso_conv schema')
        if not num.ncoll(O, A_, B):
            raise CheckError('iso degenerate')
        return
    if rule in ('central', 'central_conv'):
        if rule == 'central':
            A_, B, C, O = a[0], a[1], a[3], a[4]
            if kind != 'eqangle' or not facts_equal(kind, a, c, 'eqangle', [A_, B, A_, C, O, B, B, C], Fr(1, 2)):
                raise CheckError('central form')
            # the cited congruences must connect OA, OB, OC
            if not _connected([seg(O, A_), seg(O, B), seg(O, C)], P):
                raise CheckError('central premises')
        else:
            if kind != 'cong' or a[0] != a[2]:
                raise CheckError('central_conv kind')
            O, A_, B = a[0], a[1], a[3]
            ok = False
            for k, x, cc in P:
                if k == 'eqangle' and x[0] == A_ and x[4] == O:
                    B_, C_ = x[1], x[3]
                    if B in (B_, C_) and facts_equal(k, x, cc, 'eqangle', [A_, B_, A_, C_, O, B_, B_, C_], Fr(1, 2)) \
                            and _find(P, 'cong', [O, B_, O, C_]) and num.ncoll(A_, B_, C_):
                        ok = True
            if not ok:
                raise CheckError('central_conv schema')
            return
        if not num.ncoll(A_, B, C):
            raise CheckError('central degenerate')
        return
    if rule == 'cyclic_eqangle':
        if kind != 'cyclic' or len(set(a)) != 4:
            raise CheckError('cyclic kind')
        p0, p1, p2, p3 = a
        for (u, v, s, t) in ((p0, p1, p2, p3), (p0, p2, p1, p3), (p0, p3, p1, p2)):
            if _find(P, 'eqangle', [s, u, s, v, t, u, t, v]):
                if num.ncoll(u, v, s) and num.ncoll(u, v, t):
                    return
        raise CheckError('cyclic_eqangle schema')
    if rule == 'cyclic_center':
        if kind != 'cyclic':
            raise CheckError('cyclic kind')
        for o in set(x for k, xs, cc in P for x in xs):
            if o in a:
                continue
            if _connected([seg(o, x) for x in a], P):
                return
        raise CheckError('cyclic_center schema')
    if rule == 'cyclic_merge':
        cy = [set(x) for k, x, cc in P if k == 'cyclic']
        if kind == 'cyclic' and len(cy) == 2 and len(cy[0] & cy[1]) >= 3 and set(a) == cy[0] | cy[1]:
            return
        raise CheckError('cyclic_merge schema')
    if rule == 'cyclic_sub':
        if kind == 'cyclic' and any(k == 'cyclic' and set(a) <= set(x) for k, x, cc in P) and len(set(a)) >= 4:
            return
        raise CheckError('cyclic_sub schema')
    if rule in ('sim_aa', 'sim_sas', 'sim_sss'):
        if kind not in ('simtri', 'simtri2'):
            raise CheckError('sim kind')
        x, y, z, p, q, r = a
        if not num.ncoll(x, y, z) or not num.ncoll(p, q, r):
            raise CheckError('sim degenerate')
        if kind == 'simtri':
            angs = [[x, y, x, z, p, q, p, r], [y, x, y, z, q, p, q, r], [z, x, z, y, r, p, r, q]]
        else:
            angs = [[x, y, x, z, p, r, p, q], [y, x, y, z, q, r, q, p], [z, x, z, y, r, q, r, p]]
        ratv = [[x, y, x, z, p, q, p, r], [y, x, y, z, q, p, q, r], [z, x, z, y, r, p, r, q]]
        if rule == 'sim_aa':
            if sum(_find(P, 'eqangle', g) for g in angs) >= 2:
                return
        elif rule == 'sim_sas':
            if any(_find(P, 'eqangle', angs[i]) and _find(P, 'eqratio', ratv[i]) for i in range(3)):
                return
        else:
            sides = [[x, y, p, q, y, z, q, r], [x, y, p, q, x, z, p, r]]
            same = num.orient(x, y, z) == num.orient(p, q, r)
            if all(_find(P, 'eqratio', s_) for s_ in sides) and same == (kind == 'simtri'):
                return
        raise CheckError('%s schema' % rule)
    if rule in ('bisector', 'bisector_conv'):
        if rule == 'bisector':
            if kind != 'eqratio':
                raise CheckError('bis kind')
            d, b, d2, cc_, A_, b2, A2, c2 = a
            ok = d == d2 and b == b2 and cc_ == c2 and A_ == A2 and _find(P, 'coll', [d, b, cc_]) and \
                _find(P, 'eqangle', [A_, b, A_, d, A_, d, A_, cc_])
        else:
            if kind != 'eqangle':
                raise CheckError('bis kind')
            A_, b, A2, d, A3, d2, A4, cc_ = a
            ok = A_ == A2 == A3 == A4 and d == d2 and _find(P, 'coll', [d, b, cc_]) and \
                _find(P, 'eqratio', [d, b, d, cc_, A_, b, A_, cc_])
            b2 = b
        if ok and num.ncoll(A_, b, cc_):
            return
        raise CheckError('bisector schema')
    if rule == 'perp_bisector':
        A_, B, o1, o2 = a
        if kind == 'perp' and o1 != o2 and _find(P, 'cong', [o1, A_, o1, B]) and _find(P, 'cong', [o2, A_, o2, B]):
            return
        raise CheckError('perp_bisector schema')
    if rule in ('chord', 'chord_conv'):
        cy = [set(x) for k, x, cc in P if k == 'cyclic']
        if rule == 'chord':
            if kind != 'cong':
                raise CheckError('chord kind')
            A_, B, P_, Q = a
            for k, e, cc in P:
                if k != 'eqangle':
                    continue
                C_, R_ = e[0], e[4]
                if not any({A_, B, P_, Q, C_, R_} <= s_ for s_ in cy):
                    continue
                if facts_equal(k, e, cc, 'eqangle', [C_, A_, C_, B, R_, P_, R_, Q], Fr(0)) or \
                        facts_equal(k, e, cc, 'eqangle', [C_, A_, C_, B, R_, Q, R_, P_], Fr(0)):
                    if C_ not in (A_, B) and R_ not in (P_, Q):
                        return
            raise CheckError('chord schema')
        else:
            if kind != 'eqangle':
                raise CheckError('chord kind')
            C_, A_, B, R_ = a[0], a[1], a[3], a[4]
            P_, Q = a[5], a[7]
            if C_ in (A_, B) or R_ in (P_, Q) or not any({A_, B, P_, Q, C_, R_} <= s_ for s_ in cy):
                raise CheckError('chord_conv schema')
            if _find(P, 'cong', [A_, B, P_, Q]):
                return  # sign of the angle equality fixed by the diagram (numeric check above)
            raise CheckError('chord_conv schema')
    raise CheckError('unknown rule %s' % rule)


def _connected(segs, P):
    """Do the cited cong facts connect all given segments (equal lengths)?"""
    parent = {s: s for s in segs}

    def find(x):
        while parent[x] != x:
            x = parent[x]
        return x
    for k, x, cc in P:
        if k == 'cong':
            s1, s2 = seg(x[0], x[1]), seg(x[2], x[3])
            if s1 in parent and s2 in parent:
                parent[find(s1)] = find(s2)
    return len(set(find(s) for s in segs)) == 1


def check_file(path):
    proof = json.load(open(path))
    return check_proof(proof)


if __name__ == '__main__':
    ok = 0
    for f in sys.argv[1:]:
        try:
            info = check_file(f)
            print('ACCEPT', f, info['n_steps'], 'steps')
            ok += 1
        except CheckError as e:
            print('REJECT', f, e)
    print('%d/%d accepted' % (ok, len(sys.argv[1:])))
