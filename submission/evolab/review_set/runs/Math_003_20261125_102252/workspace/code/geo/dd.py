"""DD+AR deduction engine (numerically guided).

* AR: angle variables theta_{ab} (direction of line ab, units of pi, mod 1) and
  log-length variables l_{ab}, one per point pair.  Every fact contributes linear
  rows "form = const".  The row space is maintained in RREF over GF(p); membership
  of a candidate form is tested in O(1) by evaluating it on K random solutions of
  the homogeneous system (a form lies in the row space iff it vanishes on the
  whole solution space).  The constant is read from the diagram during search and
  recomputed exactly by the traceback/checker.
* DD: a fixed set of geometric rules whose candidate instances are proposed by
  the numeric diagram and whose premises are verified by AR queries.
"""
import itertools
from fractions import Fraction as Fr

import numpy as np

from . import numeric as N

PMOD = 33554393  # prime < 2^25 (products fit in int64)
HALF = Fr(1, 2)


def vidx(i, j):
    if i > j:
        i, j = j, i
    return j * (j - 1) // 2 + i


def nvars_for(n):
    return n * (n - 1) // 2


class AR:
    def __init__(self, nv, seed=0, K=2):
        self.nv, self.K = nv, K
        self.B = np.zeros((0, nv), np.int64)
        self.piv = []
        self.rng = np.random.RandomState(seed)
        self.update_X()

    def copy(self):
        o = AR.__new__(AR)
        o.nv, o.K, o.B, o.piv = self.nv, self.K, self.B.copy(), list(self.piv)
        o.rng = np.random.RandomState(self.rng.randint(1 << 30))
        o.xa, o.xb = list(self.xa), list(self.xb)
        return o

    def extend(self, nv):
        if nv > self.nv:
            self.B = np.hstack([self.B, np.zeros((self.B.shape[0], nv - self.nv), np.int64)])
            self.nv = nv
            self.update_X()

    def update_X(self):
        X = np.zeros((self.nv, self.K), np.int64)
        free = np.ones(self.nv, bool)
        if self.piv:
            free[self.piv] = False
        X[free] = self.rng.randint(1, PMOD, size=(int(free.sum()), self.K))
        if self.piv:
            X[self.piv] = (-(self.B[:, free] @ X[free])) % PMOD
        self.xa = X[:, 0].tolist()
        self.xb = X[:, 1].tolist()

    def zero(self, terms):
        xa, xb = self.xa, self.xb
        s = 0
        for v, c in terms:
            s += c * xa[v]
        if s % PMOD:
            return False
        s = 0
        for v, c in terms:
            s += c * xb[v]
        return s % PMOD == 0

    def add(self, terms):
        """Insert a row; returns True if it was independent."""
        if self.zero(terms):
            return False
        v = np.zeros(self.nv, np.int64)
        for var, c in terms:
            v[var] = (v[var] + c) % PMOD
        if self.piv:
            v = (v - (v[self.piv] @ self.B) % PMOD) % PMOD
        nz = np.flatnonzero(v)
        if len(nz) == 0:
            return False
        j = int(nz[0])
        v = (v * pow(int(v[j]), PMOD - 2, PMOD)) % PMOD
        if self.piv:
            col = self.B[:, j].copy()
            self.B = (self.B - (np.outer(col, v) % PMOD)) % PMOD
        self.B = np.vstack([self.B, v[None, :]])
        self.piv.append(j)
        return True


class UFAR:
    """'DD without AR' backend: only transitivity of explicitly stated equalities.
    Two-term rows (x_u - x_v = c) merge variable classes; three/four-term rows
    (angle or ratio equalities) merge 'difference keys' D(u,v) = x_v - x_u.  No other
    linear combinations are formed."""

    def __init__(self, nv, seed=0, K=2):
        self.nv = nv
        self.rows = []
        self.update_X()

    def copy(self):
        import copy as _c
        return _c.deepcopy(self)

    def extend(self, nv):
        self.nv = max(nv, self.nv)

    def add(self, terms):
        if self.zero(terms):
            return False
        self.rows.append(list(terms))
        return True

    def _find(self, par, x):
        r = x
        while par.get(r, r) != r:
            r = par[r]
        while par.get(x, x) != r:
            par[x], x = r, par[x]
        return r

    @staticmethod
    def _pairs(terms):
        pos = [v for v, c in terms for _ in range(c) if c > 0]
        neg = [v for v, c in terms for _ in range(-c) if c < 0]
        return pos, neg

    def update_X(self):
        self.vp = {}
        self.kp = {}
        for it in range(3):
            changed = False
            for t in self.rows:
                pos, neg = self._pairs(t)
                if len(pos) == 1 and len(neg) == 1:
                    a, b = self._find(self.vp, pos[0]), self._find(self.vp, neg[0])
                    if a != b:
                        self.vp[a] = b
                        changed = True
            self.kp = {}
            f = lambda x: self._find(self.vp, x)
            for t in self.rows:
                pos, neg = self._pairs(t)
                if len(pos) == 2 and len(neg) == 2:
                    P1, P2 = map(f, pos)
                    N1, N2 = map(f, neg)
                    for (a, b), (c, d) in ((((N1, P1), (P2, N2))), (((N1, P2), (P1, N2)))):
                        self._ku((a, b), (c, d))
                        self._ku((b, a), (d, c))
            # a difference key equal to a trivial one merges variable classes (e.g. eqangle => para)
            for k in list(self.kp.keys()):
                pass
            for t in self.rows:
                pos, neg = self._pairs(t)
                if len(pos) == 2 and len(neg) == 2:
                    P1, P2 = map(f, pos)
                    N1, N2 = map(f, neg)
                    for a, b in ((P1, N1), (P1, N2), (P2, N1), (P2, N2)):
                        if a == b:
                            rest_p = [x for x in (P1, P2)]
                            rest_p.remove(a)
                            rest_n = [x for x in (N1, N2)]
                            rest_n.remove(b)
                            u, v = rest_p[0], rest_n[0]
                            if u != v:
                                self.vp[u] = v
                                changed = True
                            break
            if not changed:
                break

    def _kfind(self, k):
        r = k
        while self.kp.get(r, r) != r:
            r = self.kp[r]
        return r

    def _ku(self, k1, k2):
        a, b = self._kfind(k1), self._kfind(k2)
        if a != b:
            self.kp[a] = b

    def zero(self, terms):
        f = lambda x: self._find(self.vp, x)
        pos, neg = self._pairs(terms)
        if len(pos) == 1 and len(neg) == 1:
            return f(pos[0]) == f(neg[0])
        if len(pos) == 2 and len(neg) == 2:
            P1, P2 = map(f, pos)
            N1, N2 = map(f, neg)
            if (P1 == N1 and P2 == N2) or (P1 == N2 and P2 == N1):
                return True
            for (a, b), (c, d) in (((N1, P1), (P2, N2)), ((N1, P2), (P1, N2))):
                if self._kfind((a, b)) == self._kfind((c, d)):
                    return True
            return False
        if not pos and not neg:
            return True
        return False


def merge_terms(terms):
    d = {}
    for v, c in terms:
        d[v] = d.get(v, 0) + c
    return [(v, c) for v, c in d.items() if c != 0]


class Fact:
    __slots__ = ('id', 'kind', 'args', 'const', 'rule', 'prem', 'T', 'src')

    def __init__(self, id, kind, args, const, rule, prem, T, src=None):
        self.id, self.kind, self.args, self.const = id, kind, tuple(args), const
        self.rule, self.prem, self.T, self.src = rule, prem, T, src

    def __repr__(self):
        return '%s(%s)%s[%s]' % (self.kind, ','.join(self.args),
                                 '' if not self.const else '+%s' % self.const, self.rule)


def fact_key(kind, args, const=0):
    a = list(args)
    if kind == 'coll':
        return ('coll', frozenset(a))
    if kind in ('para', 'perp', 'cong'):
        return (kind, frozenset([frozenset(a[:2]), frozenset(a[2:4])]))
    if kind == 'cyclic':
        return ('cyclic', frozenset(a))
    if kind == 'midp':
        return ('midp', a[0], frozenset(a[1:]))
    if kind in ('simtri', 'simtri2'):
        t1, t2 = tuple(a[:3]), tuple(a[3:])
        corr1 = frozenset(zip(t1, t2))
        corr2 = frozenset(zip(t2, t1))
        return (kind, frozenset([corr1, corr2]))
    return (kind, tuple(a), const)


ALL_RULES = ('coll', 'midp', 'iso', 'central', 'cyclic', 'sim', 'bis', 'pbis', 'chord')


class Engine:
    def __init__(self, prob, coords, seed=0, rules=ALL_RULES, use_ar=True, extra_capacity=2,
                 sim_angle_only=False, ar_mode='full'):
        self.prob = prob
        self.names = list(coords.keys())
        self.idx = {n: i for i, n in enumerate(self.names)}
        self.coords = dict(coords)
        self.rules = set(rules)
        self.use_ar = use_ar
        nv = nvars_for(len(self.names) + extra_capacity)
        ARC = AR if ar_mode == 'full' else UFAR
        self.ar = {'a': ARC(nv, seed), 'l': ARC(nv, seed + 1)}
        self.facts = []
        self.keys = {}
        self.rows = []      # (sys, terms, const, fact_id)
        self.committed = 0
        self.T0 = 0
        self.goal_fact = None
        self.cyc_sets = {}  # numeric circle id -> list of (frozenset, fact_id)
        self.sim_done = set()
        self._gen_candidates()
        self._add_premises()
        self.commit()

    # ------------------------------------------------------------------ basics
    def th(self, a, b):
        return vidx(self.idx[a], self.idx[b])

    def fact_rows(self, kind, args, const):
        th = self.th
        a = args
        R = []
        if kind == 'coll':
            pts = list(a)
            for z in pts[2:]:
                R.append(('a', [(th(pts[0], pts[1]), 1), (th(pts[0], z), -1)], Fr(0)))
                R.append(('a', [(th(pts[0], pts[1]), 1), (th(pts[1], z), -1)], Fr(0)))
        elif kind == 'para':
            R.append(('a', [(th(a[0], a[1]), 1), (th(a[2], a[3]), -1)], Fr(0)))
        elif kind == 'perp':
            R.append(('a', [(th(a[0], a[1]), 1), (th(a[2], a[3]), -1)], HALF))
        elif kind == 'eqangle':
            R.append(('a', merge_terms([(th(a[2], a[3]), 1), (th(a[0], a[1]), -1),
                                        (th(a[6], a[7]), -1), (th(a[4], a[5]), 1)]), Fr(const) % 1))
        elif kind == 'cong':
            R.append(('l', [(th(a[0], a[1]), 1), (th(a[2], a[3]), -1)], Fr(0)))
        elif kind == 'eqratio':
            R.append(('l', merge_terms([(th(a[0], a[1]), 1), (th(a[2], a[3]), -1),
                                        (th(a[4], a[5]), -1), (th(a[6], a[7]), 1)]), Fr(0)))
        elif kind == 'rconst':
            R.append(('l', [(th(a[0], a[1]), 1), (th(a[2], a[3]), -1)], Fr(const)))
        elif kind == 'midp':
            m, x, y = a
            R += self.fact_rows('coll', (m, x, y), 0)
            R += self.fact_rows('cong', (m, x, m, y), 0)
            R += self.fact_rows('rconst', (x, y, m, x), 1)
        elif kind == 'cyclic':
            for q in itertools.combinations(a, 4):
                p0, p1, p2, p3 = q
                for (u, v, s, t) in ((p0, p1, p2, p3), (p0, p2, p1, p3), (p0, p3, p1, p2)):
                    R += self.fact_rows('eqangle', (s, u, s, v, t, u, t, v), 0)
        elif kind in ('simtri', 'simtri2'):
            x, y, z, p, q, r = a
            if kind == 'simtri':
                R += self.fact_rows('eqangle', (x, y, x, z, p, q, p, r), 0)
                R += self.fact_rows('eqangle', (y, x, y, z, q, p, q, r), 0)
            else:
                R += self.fact_rows('eqangle', (x, y, x, z, p, r, p, q), 0)
                R += self.fact_rows('eqangle', (y, x, y, z, q, r, q, p), 0)
            R += self.fact_rows('eqratio', (x, y, p, q, y, z, q, r), 0)
            R += self.fact_rows('eqratio', (x, y, p, q, x, z, p, r), 0)
        return [r for r in R if r[1]]

    def add_fact(self, kind, args, const, rule, prem, src=None):
        k = fact_key(kind, args, const)
        if k in self.keys:
            return self.keys[k], False
        f = Fact(len(self.facts), kind, args, const, rule, prem, len(self.rows), src)
        self.facts.append(f)
        self.keys[k] = f.id
        for sysn, terms, c in self.fact_rows(kind, args, const):
            self.rows.append((sysn, terms, c, f.id))
        self.new_count += 1
        return f.id, True

    def arf(self, kind, args, const=0):
        """Fact justified by AR at the current round time T0."""
        k = fact_key(kind, args, const)
        if k in self.keys:
            return self.keys[k]
        f = Fact(len(self.facts), kind, args, const, 'ar', None, self.T0)
        self.facts.append(f)
        self.keys[k] = f.id
        return f.id

    def coll_fact(self, a, b, c):
        """coll(a,b,c) from the AR-known parallelism ab || ac (rule para_coll)."""
        k = fact_key('coll', (a, b, c))
        if k in self.keys:
            return self.keys[k]
        nc = self.new_count
        fid, _ = self.add_fact('coll', (a, b, c), 0, 'para_coll', [self.arf('para', (a, b, a, c))])
        return fid

    def r_coll(self):
        for L in self.nlines:
            for a, b, c in itertools.combinations(L, 3):
                if fact_key('coll', (a, b, c)) in self.keys:
                    continue
                for x, y, z in ((a, b, c), (b, a, c), (c, a, b)):
                    if self.k_coll(x, y, z):
                        if not (self.k_coll(y, x, z)):
                            self.coll_fact(x, y, z)
                        break

    def commit(self):
        for i in range(self.committed, len(self.rows)):
            sysn, terms, c, fid = self.rows[i]
            if self.use_ar or sysn == 'x':
                self.ar[sysn].add(terms)
        self.committed = len(self.rows)
        self.ar['a'].update_X()
        self.ar['l'].update_X()
        self.T0 = len(self.rows)

    # queries ---------------------------------------------------------------
    def A0(self, terms):
        return self.ar['a'].zero(terms)

    def L0(self, terms):
        return self.ar['l'].zero(terms)

    def k_coll(self, a, b, c):
        return self.A0([(self.th(a, b), 1), (self.th(a, c), -1)])

    def k_para(self, a, b, c, d):
        return self.A0([(self.th(a, b), 1), (self.th(c, d), -1)])

    def k_cong(self, a, b, c, d):
        return self.L0([(self.th(a, b), 1), (self.th(c, d), -1)])

    def k_eqangle(self, a, b, c, d, p, q, r, s):
        th = self.th
        return self.A0(merge_terms([(th(c, d), 1), (th(a, b), -1), (th(r, s), -1), (th(p, q), 1)]))

    def k_eqratio(self, a, b, c, d, p, q, r, s):
        th = self.th
        return self.L0(merge_terms([(th(a, b), 1), (th(c, d), -1), (th(p, q), -1), (th(r, s), 1)]))

    # ------------------------------------------------------------- premises
    def _add_premises(self):
        self.new_count = 0
        self.premise_ids = []
        for pi, (ci, cname, pred, args) in enumerate(self.prob.premises()):
            if pred in ('coll', 'para', 'perp', 'cong', 'eqangle', 'cyclic', 'eqratio'):
                kind, a, c = pred, args, 0
            elif pred == 'rconst':
                import math
                kind, a = 'rconst', args[:4]
                c = Fr(int(args[4]), int(args[5]))
                if c.denominator & (c.denominator - 1) or c.numerator & (c.numerator - 1):
                    continue
                c = Fr(int(math.log2(c.numerator)) - int(math.log2(c.denominator)))
            else:
                continue
            if kind == 'coll' and len(set(a)) < 3:
                continue
            fid, new = self.add_fact(kind, a, c, 'premise', [], src=(ci, cname, pred, tuple(args), pi))
            self.premise_ids.append(fid)

    # ------------------------------------------------------------ candidates
    def _gen_candidates(self):
        names = self.names
        Z = np.array([self.coords[n] for n in names])
        n = len(names)
        self.Z = Z
        D = np.abs(Z[:, None] - Z[None, :])
        self.D = D
        tol = 1e-7
        # equidistance groups: center -> list of groups (lists of names)
        self.eqd = {}
        for o in range(n):
            order = [j for j in np.argsort(D[o]) if j != o]
            grp = [order[0]]
            groups = []
            for j in order[1:]:
                if abs(D[o, j] - D[o, grp[-1]]) < tol * max(1, D[o, j]):
                    grp.append(j)
                else:
                    if len(grp) >= 2:
                        groups.append([names[k] for k in grp])
                    grp = [j]
            if len(grp) >= 2:
                groups.append([names[k] for k in grp])
            if groups:
                self.eqd[names[o]] = groups
        # iso candidates (o, a, b)
        self.cand_iso = []
        for o, groups in self.eqd.items():
            for g in groups:
                for a, b in itertools.combinations(g, 2):
                    if not N.n_coll(self.coords[o], self.coords[a], self.coords[b], 1e-6):
                        self.cand_iso.append((o, a, b))
        # lines
        lines = {}
        for i, j in itertools.combinations(range(n), 2):
            s = [k for k in range(n) if N.n_coll(Z[i], Z[j], Z[k], 1e-8)]
            if len(s) >= 3:
                lines[frozenset(s)] = 1
        self.nlines = [[names[k] for k in sorted(s)] for s in lines]
        # circles through >=4 points
        circ = {}
        for i, j, k in itertools.combinations(range(n), 3):
            if N.n_coll(Z[i], Z[j], Z[k], 1e-6):
                continue
            o = N.circumcenter(Z[i], Z[j], Z[k])
            r = abs(Z[i] - o)
            if r > 100:
                continue
            key = (round(o.real, 5), round(o.imag, 5), round(r, 5))
            circ.setdefault(key, set()).update((i, j, k))
        self.ncircles = []
        seen = set()
        for key, s in circ.items():
            if len(s) >= 4:
                fs = frozenset(s)
                if fs in seen:
                    continue
                seen.add(fs)
                o = complex(key[0], key[1])
                center = None
                for c in range(n):
                    if abs(Z[c] - o) < 1e-6:
                        center = names[c]
                self.ncircles.append(([names[k] for k in sorted(s)], center))
        self.cyc_sets = {ci: [] for ci in range(len(self.ncircles))}
        # midpoints
        hz = {(round(z.real, 6), round(z.imag, 6)): names[k] for k, z in enumerate(Z)}
        self.cand_midp = []
        for i, j in itertools.combinations(range(n), 2):
            m = (Z[i] + Z[j]) / 2
            for dx in (0,):
                key = (round(m.real, 6), round(m.imag, 6))
                if key in hz:
                    self.cand_midp.append((hz[key], names[i], names[j]))
        # bisector candidates (a, d, b, c)
        self.cand_bis = []
        for L in self.nlines:
            Ls = set(L)
            for d in L:
                for b, c in itertools.combinations([x for x in L if x != d], 2):
                    for a in names:
                        if a in Ls:
                            continue
                        P = self.coords
                        r1 = abs(P[d] - P[b]) / abs(P[d] - P[c])
                        r2 = abs(P[a] - P[b]) / abs(P[a] - P[c])
                        if abs(r1 - r2) < 1e-7 * max(1, r1):
                            self.cand_bis.append((a, d, b, c))
        # similar triangles
        self._gen_sim()
        # equal chords in numeric circles
        self.cand_chord = []
        for ci, (S, center) in enumerate(self.ncircles):
            chords = list(itertools.combinations(S, 2))
            for (a, b), (p, q) in itertools.combinations(chords, 2):
                if abs(D[self.idx[a], self.idx[b]] - D[self.idx[p], self.idx[q]]) < 1e-7:
                    self.cand_chord.append((ci, a, b, p, q))
        # perp bisector candidates: pair (a,b) -> list of equidistant points
        pb = {}
        for o, a, b in self.cand_iso:
            pb.setdefault((a, b) if a < b else (b, a), []).append(o)
        self.cand_pb = {k: v for k, v in pb.items() if len(v) >= 2}

    def _gen_sim(self):
        names, Z = self.names, self.Z
        n = len(names)
        bucket = {}
        tris = []
        for i, j, k in itertools.combinations(range(n), 3):
            a, b, c = Z[i], Z[j], Z[k]
            if N.n_coll(a, b, c, 1e-5):
                continue
            tris.append((i, j, k))
        for t in tris:
            for perm in itertools.permutations(t):
                a, b, c = Z[perm[0]], Z[perm[1]], Z[perm[2]]
                z = (c - a) / (b - a)
                if abs(z) > 50:
                    continue
                key = (round(z.real * 1e5), round(z.imag * 1e5))
                bucket.setdefault(key, []).append(perm)
        cands = {}
        for t in tris:
            a, b, c = Z[t[0]], Z[t[1]], Z[t[2]]
            for perm in itertools.permutations(t):
                a, b, c = Z[perm[0]], Z[perm[1]], Z[perm[2]]
                z = (c - a) / (b - a)
                if abs(z) > 50:
                    continue
                for orient, zz in (('simtri', z), ('simtri2', z.conjugate())):
                    key = (round(zz.real * 1e5), round(zz.imag * 1e5))
                    for t2 in bucket.get(key, ()):
                        if set(t2) == set(perm):
                            continue
                        n1 = tuple(names[x] for x in perm)
                        n2 = tuple(names[x] for x in t2)
                        k = fact_key(orient, n1 + n2)
                        if k not in cands:
                            cands[k] = (orient, n1, n2)
        self.cand_sim = list(cands.values())

    # ------------------------------------------------------------------ rules
    def run_rules(self):
        R = self.rules
        if 'coll' in R:
            self.r_coll()
        if 'midp' in R:
            self.r_midp()
        if 'iso' in R:
            self.r_iso()
        if 'central' in R:
            self.r_central()
        if 'cyclic' in R:
            self.r_cyclic()
        if 'sim' in R:
            self.r_sim()
        if 'bis' in R:
            self.r_bis()
        if 'pbis' in R:
            self.r_pbis()
        if 'chord' in R:
            self.r_chord()

    def r_midp(self):
        for m, a, b in self.cand_midp:
            if fact_key('midp', (m, a, b)) in self.keys:
                continue
            if self.k_coll(m, a, b) and self.k_cong(m, a, m, b):
                self.add_fact('midp', (m, a, b), 0, 'midp',
                              [self.coll_fact(m, a, b), self.arf('cong', (m, a, m, b))])

    def r_iso(self):
        for o, a, b in self.cand_iso:
            c = self.k_cong(o, a, o, b)
            e = self.k_eqangle(a, o, a, b, b, a, b, o)
            if c and not e:
                self.add_fact('eqangle', (a, o, a, b, b, a, b, o), 0, 'iso', [self.arf('cong', (o, a, o, b))])
            elif e and not c:
                self.add_fact('cong', (o, a, o, b), 0, 'iso_conv',
                              [self.arf('eqangle', (a, o, a, b, b, a, b, o))])

    def central_terms(self, o, a, b, c):
        th = self.th
        return merge_terms([(th(a, c), 1), (th(a, b), -1), (th(b, c), -1), (th(o, b), 1)])

    def r_central(self):
        for o, groups in self.eqd.items():
            for g in groups:
                if len(g) < 3:
                    continue
                for a in g:
                    for b, c in itertools.combinations([x for x in g if x != a], 2):
                        known = self.A0(self.central_terms(o, a, b, c))
                        if known:
                            if not self.k_cong(o, a, o, b) and self.k_cong(o, b, o, c):
                                self.add_fact('cong', (o, a, o, b), 0, 'central_conv',
                                              [self.arf('cong', (o, b, o, c)),
                                               self.arf('eqangle', (a, b, a, c, o, b, b, c), HALF)])
                            continue
                        if self.k_cong(o, a, o, b) and self.k_cong(o, a, o, c):
                            self.add_fact('eqangle', (a, b, a, c, o, b, b, c), HALF, 'central',
                                          [self.arf('cong', (o, a, o, b)), self.arf('cong', (o, a, o, c))])

    def r_cyclic(self):
        for ci, (S, center) in enumerate(self.ncircles):
            proven = self.cyc_sets[ci]
            newsets = []
            for q in itertools.combinations(S, 4):
                qs = frozenset(q)
                if any(qs <= ps for ps, _ in proven) or any(qs <= ps for ps, _ in newsets):
                    continue
                a, b, c, d = q
                found = None
                for (u, v, s, t) in ((a, b, c, d), (a, c, b, d), (a, d, b, c)):
                    if self.k_eqangle(s, u, s, v, t, u, t, v):
                        found = ('cyclic_eqangle', [self.arf('eqangle', (s, u, s, v, t, u, t, v))])
                        break
                if found is None and center is not None and all(self.k_cong(center, a, center, x) for x in (b, c, d)):
                    found = ('cyclic_center', [self.arf('cong', (center, a, center, x)) for x in (b, c, d)])
                if found:
                    fid, new = self.add_fact('cyclic', tuple(sorted(q)), 0, found[0], found[1])
                    newsets.append((qs, fid))
            proven += newsets
            # merging
            changed = True
            while changed:
                changed = False
                for (s1, f1), (s2, f2) in itertools.combinations(list(proven), 2):
                    if len(s1 & s2) >= 3 and not (s1 <= s2 or s2 <= s1):
                        u = s1 | s2
                        fid, new = self.add_fact('cyclic', tuple(sorted(u)), 0, 'cyclic_merge', [f1, f2])
                        proven[:] = [(s, f) for s, f in proven if not (s <= u)] + [(u, fid)]
                        changed = True
                        break
                    if s1 <= s2 and s1 != s2:
                        proven[:] = [(s, f) for s, f in proven if s != s1]
                        changed = True
                        break

    def sim_conditions(self, orient, t1, t2):
        x, y, z = t1
        p, q, r = t2
        if orient == 'simtri':
            angs = [(x, y, x, z, p, q, p, r), (y, x, y, z, q, p, q, r), (z, x, z, y, r, p, r, q)]
        else:
            angs = [(x, y, x, z, p, r, p, q), (y, x, y, z, q, r, q, p), (z, x, z, y, r, q, r, p)]
        rats_v = [(x, y, x, z, p, q, p, r), (y, x, y, z, q, p, q, r), (z, x, z, y, r, p, r, q)]
        sides = [(x, y, p, q, y, z, q, r), (x, y, p, q, x, z, p, r)]
        return angs, rats_v, sides

    def r_sim(self):
        for orient, t1, t2 in self.cand_sim:
            k = fact_key(orient, t1 + t2)
            if k in self.sim_done or k in self.keys:
                continue
            angs, rats_v, sides = self.sim_conditions(orient, t1, t2)
            ka = [self.k_eqangle(*a) for a in angs]
            na = sum(ka)
            ks = [self.k_eqratio(*s) for s in sides]
            if na == 3 and all(ks):
                self.sim_done.add(k)
                continue
            rule = prem = None
            if na >= 2:
                i1, i2 = [i for i in range(3) if ka[i]][:2]
                rule, prem = 'sim_aa', [self.arf('eqangle', angs[i1]), self.arf('eqangle', angs[i2])]
            else:
                if na == 1:
                    v = ka.index(True)
                    if self.k_eqratio(*rats_v[v]):
                        rule, prem = 'sim_sas', [self.arf('eqangle', angs[v]), self.arf('eqratio', rats_v[v])]
                if rule is None and all(ks):
                    rule, prem = 'sim_sss', [self.arf('eqratio', sides[0]), self.arf('eqratio', sides[1])]
            if rule:
                self.add_fact(orient, t1 + t2, 0, rule, prem)

    def r_bis(self):
        for a, d, b, c in self.cand_bis:
            if not self.k_coll(d, b, c):
                continue
            ang = self.k_eqangle(a, b, a, d, a, d, a, c)
            rat = self.k_eqratio(d, b, d, c, a, b, a, c)
            if ang and not rat:
                self.add_fact('eqratio', (d, b, d, c, a, b, a, c), 0, 'bisector',
                              [self.coll_fact(d, b, c), self.arf('eqangle', (a, b, a, d, a, d, a, c))])
            elif rat and not ang:
                self.add_fact('eqangle', (a, b, a, d, a, d, a, c), 0, 'bisector_conv',
                              [self.coll_fact(d, b, c), self.arf('eqratio', (d, b, d, c, a, b, a, c))])

    def r_pbis(self):
        for (a, b), os_ in self.cand_pb.items():
            E = [o for o in os_ if self.k_cong(o, a, o, b)]
            for o2 in E[1:]:
                o1 = E[0]
                th = self.th
                if not self.A0([(th(a, b), 1), (th(o1, o2), -1)]):
                    self.add_fact('perp', (a, b, o1, o2), 0, 'perp_bisector',
                                  [self.arf('cong', (o1, a, o1, b)), self.arf('cong', (o2, a, o2, b))])

    def r_chord(self):
        for ci, a, b, p, q in self.cand_chord:
            proven = self.cyc_sets.get(ci, [])
            S = None
            for ps, fid in proven:
                if {a, b, p, q} <= ps and len(ps) >= 4:
                    S = (ps, fid)
                    break
            if S is None:
                continue
            ps, fid = S
            cands_c = [x for x in sorted(ps) if x not in (a, b)]
            cands_r = [x for x in sorted(ps) if x not in (p, q)]
            if not cands_c or not cands_r:
                continue
            c, r = cands_c[0], cands_r[0]
            if c == r and len(cands_r) > 1:
                r = cands_r[1]
            if c == r:
                continue
            e1 = (c, a, c, b, r, p, r, q)
            e2 = (c, a, c, b, r, q, r, p)
            k1, k2 = self.k_eqangle(*e1), self.k_eqangle(*e2)
            kc = self.k_cong(a, b, p, q)
            if (k1 or k2) and not kc:
                self.add_fact('cong', (a, b, p, q), 0, 'chord', [fid, self.arf('eqangle', e1 if k1 else e2)])
            elif kc and not (k1 or k2):
                P = self.coords
                e = e1 if N.n_eqangle(*[P[x] for x in e1], tol=1e-6) else e2
                self.add_fact('eqangle', e, 0, 'chord_conv', [fid, self.arf('cong', (a, b, p, q))])

    # ------------------------------------------------------------------ goal
    def check_goal(self, goal=None):
        goal = goal or self.prob.goal
        if goal is None:
            return None
        kind, a = goal
        a = tuple(a)
        ok = False
        if kind == 'cong':
            ok = self.k_cong(*a)
        elif kind == 'eqratio':
            ok = self.k_eqratio(*a)
        elif kind == 'eqangle':
            ok = self.k_eqangle(*a)
        elif kind == 'para':
            ok = self.k_para(*a)
        elif kind == 'perp':
            ok = self.k_para(*a)  # constant (1/2) read from diagram
        elif kind == 'coll':
            ok = all(self.k_coll(a[0], a[1], z) for z in a[2:])
        elif kind == 'cyclic':
            gs = frozenset(a)
            for ci, lst in self.cyc_sets.items():
                for ps, fid in lst:
                    if gs <= ps:
                        if ps == gs:
                            return fid
                        return self.add_fact_direct('cyclic', tuple(a), 'cyclic_sub', [fid])
            p0, p1, p2, p3 = a
            for (u, v, s, t) in ((p0, p1, p2, p3), (p0, p2, p1, p3), (p0, p3, p1, p2)):
                if self.k_eqangle(s, u, s, v, t, u, t, v):
                    return self.add_fact_direct('cyclic', tuple(a), 'cyclic_eqangle',
                                                [self.arf('eqangle', (s, u, s, v, t, u, t, v))])
            return None
        if ok:
            const = HALF if kind == 'perp' else 0
            if kind == 'coll':
                for z in a[2:]:
                    fid = self.coll_fact(a[0], a[1], z)
                return fid
            return self.arf(kind, a, const)
        return None

    def add_fact_direct(self, kind, args, rule, prem):
        k = fact_key(kind, args)
        if k in self.keys:
            return self.keys[k]
        self.new_count = 0
        fid, _ = self.add_fact(kind, args, 0, rule, prem)
        return fid

    def saturate(self, max_rounds=40, stop_at_goal=True, goal=None):
        self.rounds = 0
        for rnd in range(max_rounds):
            self.rounds = rnd + 1
            if stop_at_goal:
                g = self.check_goal(goal)
                if g is not None:
                    self.goal_fact = g
                    return g
            self.new_count = 0
            self.run_rules()
            self.commit()
            if self.new_count == 0:
                break
        g = self.check_goal(goal)
        if g is not None:
            self.goal_fact = g
        return g
