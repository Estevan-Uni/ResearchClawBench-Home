"""Numeric geometry primitives (complex-number based) and construction sketches.

Every construction of data/defs.txt that is needed is implemented here as a
function returning either explicit points or a list of 1-D loci (Line / Circle)
whose intersection defines the new point.
"""
import cmath
import math
import random

EPS = 1e-9


class Line:
    __slots__ = ('p', 'd')

    def __init__(self, p, d):
        if abs(d) < 1e-12:
            raise DegenerateError('zero direction')
        self.p = p
        self.d = d / abs(d)

    def sample(self, rng, scale=1.0):
        return self.p + self.d * rng.gauss(0, 1) * scale


class Circle:
    __slots__ = ('c', 'r')

    def __init__(self, c, r):
        if r < 1e-9:
            raise DegenerateError('zero radius')
        self.c = c
        self.r = r

    def sample(self, rng, scale=1.0):
        return self.c + self.r * cmath.exp(1j * rng.uniform(0, 2 * math.pi))


class DegenerateError(Exception):
    pass


def unit(z):
    if abs(z) < 1e-12:
        raise DegenerateError('unit of zero')
    return z / abs(z)


def line2(a, b):
    return Line(a, b - a)


def intersect(l1, l2):
    """Return list of intersection points of two loci."""
    if isinstance(l1, Line) and isinstance(l2, Line):
        cr = (l1.d.conjugate() * l2.d).imag
        if abs(cr) < 1e-9:
            raise DegenerateError('parallel lines')
        t = ((l2.p - l1.p).conjugate() * l2.d).imag / cr
        return [l1.p + l1.d * t]
    if isinstance(l1, Circle) and isinstance(l2, Line):
        l1, l2 = l2, l1
    if isinstance(l1, Line) and isinstance(l2, Circle):
        # p + t d, |p + t d - c| = r
        w = l1.p - l2.c
        b = (w.conjugate() * l1.d).real
        cc = abs(w) ** 2 - l2.r ** 2
        disc = b * b - cc
        if disc < -1e-12:
            raise DegenerateError('line misses circle')
        disc = math.sqrt(max(disc, 0.0))
        return [l1.p + l1.d * (-b + disc), l1.p + l1.d * (-b - disc)]
    # circle-circle
    d = abs(l2.c - l1.c)
    if d < 1e-9:
        raise DegenerateError('concentric')
    a = (l1.r ** 2 - l2.r ** 2 + d * d) / (2 * d)
    h2 = l1.r ** 2 - a * a
    if h2 < -1e-12:
        raise DegenerateError('circles do not meet')
    h = math.sqrt(max(h2, 0.0))
    u = (l2.c - l1.c) / d
    m = l1.c + u * a
    return [m + 1j * u * h, m - 1j * u * h]


def foot(a, b, c):
    d = unit(c - b)
    return b + d * ((a - b) * d.conjugate()).real


def reflect(a, b, c):
    f = foot(a, b, c)
    return 2 * f - a


def circumcenter(a, b, c):
    ax, ay, bx, by, cx, cy = a.real, a.imag, b.real, b.imag, c.real, c.imag
    d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    if abs(d) < 1e-12:
        raise DegenerateError('collinear circumcenter')
    ux = ((ax * ax + ay * ay) * (by - cy) + (bx * bx + by * by) * (cy - ay) + (cx * cx + cy * cy) * (ay - by)) / d
    uy = ((ax * ax + ay * ay) * (cx - bx) + (bx * bx + by * by) * (ax - cx) + (cx * cx + cy * cy) * (bx - ax)) / d
    return complex(ux, uy)


def orthocenter(a, b, c):
    o = circumcenter(a, b, c)
    return a + b + c - 2 * o


def incenter(a, b, c):
    la, lb, lc = abs(b - c), abs(c - a), abs(a - b)
    return (la * a + lb * b + lc * c) / (la + lb + lc)


def excenter(a, b, c):
    la, lb, lc = abs(b - c), abs(c - a), abs(a - b)
    s = -la + lb + lc
    if abs(s) < 1e-12:
        raise DegenerateError('excenter')
    return (-la * a + lb * b + lc * c) / s


def rand_triangle(rng):
    """Random reasonably non-degenerate triangle."""
    while True:
        pts = [complex(rng.uniform(-1, 1), rng.uniform(-1, 1)) for _ in range(3)]
        a, b, c = pts
        ok = True
        for x, y, z in ((a, b, c), (b, c, a), (c, a, b)):
            ang = abs(cmath.phase((y - x) / (z - x)))
            if ang < math.radians(20) or ang > math.radians(140):
                ok = False
        if min(abs(a - b), abs(b - c), abs(c - a)) < 0.5:
            ok = False
        if ok:
            return pts


def rand_point(rng):
    return complex(rng.uniform(-1, 1), rng.uniform(-1, 1))


def ang_dir(z):
    """Direction of vector z in units of pi, modulo 1."""
    return (cmath.phase(z) / math.pi) % 1.0


def sketch(name, args, P, rng):
    """Numeric construction.

    name: construction name; args: list of argument names (outputs first as in defs);
    P: dict name -> complex of existing points.
    Returns ('pts', [complex,...]) for explicit outputs or ('loci', [locus...]).
    For explicit outputs the list matches the construction's output order (see OUTS).
    """
    g = lambda k: P[args[k]]
    if name == 'angle_bisector':
        a, b, c = g(1), g(2), g(3)
        return 'loci', [Line(b, unit(a - b) + unit(c - b))]
    if name == 'angle_mirror':
        a, b, c = g(1), g(2), g(3)
        u = unit(c - b)
        return 'loci', [Line(b, u * u * (a - b).conjugate())]
    if name in ('circle', 'circumcenter'):
        return 'pts', [circumcenter(g(1), g(2), g(3))]
    if name == 'eq_triangle':
        b, c = g(1), g(2)
        s = rng.choice([1, -1])
        return 'pts', [b + (c - b) * cmath.exp(1j * s * math.pi / 3)]
    if name == 'eqangle2':
        a, b, c = g(1), g(2), g(3)
        for _ in range(20):
            t = rng.uniform(0.15, 0.85) * math.pi
            la = Line(a, (b - a) * cmath.exp(1j * t))
            lc = Line(c, (b - c) * cmath.exp(-1j * t))
            try:
                return 'pts', intersect(la, lc)
            except DegenerateError:
                continue
        raise DegenerateError('eqangle2')
    if name == 'eqdistance':
        return 'loci', [Circle(g(1), abs(g(2) - g(3)))]
    if name == 'foot':
        return 'pts', [foot(g(1), g(2), g(3))]
    if name == 'free':
        return 'pts', [rand_point(rng)]
    if name == 'incenter':
        return 'pts', [incenter(g(1), g(2), g(3))]
    if name == 'excenter':
        return 'pts', [excenter(g(1), g(2), g(3))]
    if name in ('incenter2', 'excenter2'):
        a, b, c = g(4), g(5), g(6)
        i = incenter(a, b, c) if name == 'incenter2' else excenter(a, b, c)
        return 'pts', [foot(i, b, c), foot(i, c, a), foot(i, a, b), i]
    if name == 'centroid':
        a, b, c = g(4), g(5), g(6)
        return 'pts', [(b + c) / 2, (c + a) / 2, (a + b) / 2, (a + b + c) / 3]
    if name == 'ninepoints':
        a, b, c = g(4), g(5), g(6)
        x, y, z = (b + c) / 2, (c + a) / 2, (a + b) / 2
        return 'pts', [x, y, z, circumcenter(x, y, z)]
    if name == 'intersection_cc':
        o, w, a = g(1), g(2), g(3)
        return 'pts', [reflect(a, o, w)]
    if name == 'intersection_lc':
        a, o, b = g(1), g(2), g(3)
        pts = intersect(line2(a, b), Circle(o, abs(b - o)))
        pts = [p for p in pts if abs(p - b) > 1e-6]
        return 'pts', pts[:1]
    if name == 'intersection_ll':
        return 'pts', intersect(line2(g(1), g(2)), line2(g(3), g(4)))
    if name == 'intersection_lp':
        return 'pts', intersect(line2(g(1), g(2)), Line(g(3), g(5) - g(4)))
    if name == 'intersection_lt':
        return 'pts', intersect(line2(g(1), g(2)), Line(g(3), 1j * (g(5) - g(4))))
    if name == 'intersection_pp':
        return 'pts', intersect(Line(g(1), g(3) - g(2)), Line(g(4), g(6) - g(5)))
    if name == 'intersection_tt':
        return 'pts', intersect(Line(g(1), 1j * (g(3) - g(2))), Line(g(4), 1j * (g(6) - g(5))))
    if name == 'iso_triangle':
        a, b = rand_point(rng), rand_point(rng)
        t = rng.uniform(0.4, 2.6)
        return 'pts', [a, b, a + (b - a) * cmath.exp(1j * t)]
    if name == 'lc_tangent':
        a, o = g(1), g(2)
        return 'loci', [Line(a, 1j * (a - o))]
    if name == 'midpoint':
        return 'pts', [(g(1) + g(2)) / 2]
    if name == 'mirror':
        return 'pts', [2 * g(2) - g(1)]
    if name == 'nsquare':
        a, b = g(1), g(2)
        return 'pts', [a + (b - a) * (-1j)]
    if name == 'psquare':
        a, b = g(1), g(2)
        return 'pts', [a + (b - a) * 1j]
    if name == 'on_aline':
        a, b, c, d, e = g(1), g(2), g(3), g(4), g(5)
        return 'loci', [Line(a, unit(b - a) * unit(c - d) / unit(e - d))]
    if name in ('on_aline2', 'eqangle3'):
        if name == 'on_aline2':
            a, b, c, d, e = g(1), g(2), g(3), g(4), g(5)
            rot = unit(e - d) / unit(c - d)  # angle from dc to de
        else:
            a, b, d, e, f = g(1), g(2), g(3), g(4), g(5)
            rot = unit(f - d) / unit(e - d)
        for _ in range(20):
            t = rng.uniform(0, 2 * math.pi)
            da = cmath.exp(1j * t)
            try:
                x0 = intersect(Line(a, da), Line(b, da * rot))[0]
                return 'loci', [Circle(circumcenter(a, b, x0), abs(circumcenter(a, b, x0) - a))]
            except DegenerateError:
                continue
        raise DegenerateError('aline2')
    if name == 'on_bline':
        a, b = g(1), g(2)
        return 'loci', [Line((a + b) / 2, 1j * (b - a))]
    if name == 'on_circle':
        return 'loci', [Circle(g(1), abs(g(2) - g(1)))]
    if name in ('on_line', 'on_opline'):
        return 'loci', [line2(g(1), g(2))]
    if name == 'on_pline':
        return 'loci', [Line(g(1), g(3) - g(2))]
    if name == 'on_tline':
        return 'loci', [Line(g(1), 1j * (g(3) - g(2)))]
    if name == 'on_dia':
        a, b = g(1), g(2)
        return 'loci', [Circle((a + b) / 2, abs(a - b) / 2)]
    if name == 'on_circum':
        a, b, c = g(1), g(2), g(3)
        o = circumcenter(a, b, c)
        return 'loci', [Circle(o, abs(a - o))]
    if name == 'orthocenter':
        return 'pts', [orthocenter(g(1), g(2), g(3))]
    if name == 'parallelogram':
        a, b, c = g(0), g(1), g(2)
        return 'pts', [a + c - b]
    if name == 'reflect':
        return 'pts', [reflect(g(1), g(2), g(3))]
    if name == 'shift':
        return 'pts', [g(1) + g(2) - g(3)]
    if name in ('triangle',):
        return 'pts', rand_triangle(rng)
    if name in ('segment',):
        a = rand_point(rng)
        return 'pts', [a, a + 1.3 * cmath.exp(1j * rng.uniform(0, 2 * math.pi))]
    if name in ('quadrangle', 'pentagon'):
        n = 4 if name == 'quadrangle' else 5
        return 'pts', [rand_point(rng) for _ in range(n)]
    if name == 'r_triangle':
        a, b = rand_point(rng), rand_point(rng)
        return 'pts', [a, b, a + 1j * (b - a) * rng.uniform(0.5, 1.6) * rng.choice([1, -1])]
    if name == 'risos':
        a, b = rand_point(rng), rand_point(rng)
        return 'pts', [a, b, a + 1j * (b - a)]
    if name in ('ieq_triangle',):
        a, b = rand_point(rng), rand_point(rng)
        return 'pts', [a, b, a + (b - a) * cmath.exp(1j * math.pi / 3)]
    if name == 'trapezoid':
        a, b, c = rand_triangle(rng)
        return 'pts', [a, b, c, c + (a - b) * rng.uniform(0.3, 1.5)]
    if name == 'r_trapezoid':
        a, b = rand_point(rng), rand_point(rng)
        d = a + 1j * (b - a) * rng.uniform(0.5, 1.5)
        return 'pts', [a, b, d + (b - a) * rng.uniform(0.3, 1.5), d]
    if name == 'eq_trapezoid':
        a, b = rand_point(rng), rand_point(rng)
        m = (a + b) / 2
        h = 1j * (b - a) * rng.uniform(0.4, 1.2)
        s = rng.uniform(0.3, 1.5)
        c = m + h + (b - a) / 2 * s
        d = m + h - (b - a) / 2 * s
        return 'pts', [a, b, c, d]
    if name == 'rectangle':
        a, b = rand_point(rng), rand_point(rng)
        h = 1j * (b - a) * rng.uniform(0.5, 1.5)
        return 'pts', [a, b, b + h, a + h]
    if name == 'isquare':
        a, b = rand_point(rng), rand_point(rng)
        h = 1j * (b - a)
        return 'pts', [a, b, b + h, a + h]
    if name == 'square':
        a, b = g(0), g(1)
        h = 1j * (b - a)
        return 'pts', [b + h, a + h]
    if name == 'cc_tangent':
        o, a, w, b = g(4), g(5), g(6), g(7)
        r1, r2 = abs(a - o), abs(b - w)
        d = abs(w - o)
        if d <= abs(r1 - r2) + 1e-9:
            raise DegenerateError('no external tangent')
        u = unit(w - o)
        res = []
        # external tangents: normal n with n.(w-o) = r1 - r2
        cosv = (r1 - r2) / d
        sinv = math.sqrt(max(0.0, 1 - cosv * cosv))
        for s in (1, -1):
            n = u * complex(cosv, s * sinv)
            res.append((o + r1 * n, w + r2 * n))
        return 'pts', [res[0][0], res[0][1], res[1][0], res[1][1]]
    if name == 'tangent':
        a, o, b = g(2), g(3), g(4)
        r = abs(b - o)
        d = abs(a - o)
        if d <= r + 1e-9:
            raise DegenerateError('tangent from inside')
        u = unit(a - o)
        cosv = r / d
        sinv = math.sqrt(1 - cosv * cosv)
        return 'pts', [o + r * u * complex(cosv, sinv), o + r * u * complex(cosv, -sinv)]
    if name == 'trisegment':
        a, b = g(2), g(3)
        return 'pts', [a + (b - a) / 3, a + 2 * (b - a) / 3]
    if name == 'triangle12':
        a, b = rand_point(rng), rand_point(rng)
        return 'pts', [a, b, a + 2 * (b - a) * cmath.exp(1j * rng.uniform(0.5, 2.6))]
    raise NotImplementedError(name)


# ---------------- numeric predicates ---------------------------------------
TOL = 1e-7


def n_coll(a, b, c, tol=TOL):
    s = max(abs(b - a), abs(c - a), 1e-12)
    return abs(((b - a).conjugate() * (c - a)).imag) / s / s < tol


def n_para(a, b, c, d, tol=TOL):
    return abs((unit(b - a).conjugate() * unit(d - c)).imag) < tol


def n_perp(a, b, c, d, tol=TOL):
    return abs((unit(b - a).conjugate() * unit(d - c)).real) < tol


def n_cong(a, b, c, d, tol=TOL):
    return abs(abs(a - b) - abs(c - d)) < tol * max(1.0, abs(a - b))


def angdiff(a, b, c, d):
    """Directed angle from line ab to line cd, units of pi, mod 1."""
    return (ang_dir(d - c) - ang_dir(b - a)) % 1.0


def mod1_close(x, y, tol=TOL):
    t = (x - y) % 1.0
    return min(t, 1 - t) < tol


def n_eqangle(a, b, c, d, p, q, r, s, tol=TOL):
    return mod1_close(angdiff(a, b, c, d), angdiff(p, q, r, s), tol)


def n_cyclic(pts, tol=TOL):
    a, b, c = pts[:3]
    try:
        o = circumcenter(a, b, c)
    except DegenerateError:
        return False
    r = abs(a - o)
    return all(abs(abs(p - o) - r) < tol * max(1, r) for p in pts[3:])


def n_eqratio(a, b, c, d, p, q, r, s, tol=TOL):
    x = abs(a - b) / abs(c - d)
    y = abs(p - q) / abs(r - s)
    return abs(x - y) < tol * max(1, x)


def check_pred(name, args, P, tol=1e-6):
    """Numerically check a defs-language predicate."""
    pts = [P[x] for x in args if x in P]
    if name == 'coll':
        return all(n_coll(pts[0], pts[1], z, tol) for z in pts[2:])
    if name == 'para':
        return n_para(*pts[:4], tol=tol)
    if name == 'perp':
        return n_perp(*pts[:4], tol=tol)
    if name == 'cong':
        return n_cong(*pts[:4], tol=tol)
    if name == 'eqangle':
        return n_eqangle(*pts[:8], tol=tol)
    if name == 'cyclic':
        return n_cyclic(pts, tol=tol)
    if name == 'eqratio':
        return n_eqratio(*pts[:8], tol=tol)
    if name == 'midp':
        return abs(pts[0] - (pts[1] + pts[2]) / 2) < tol
    if name == 'rconst':
        num, den = int(args[4]), int(args[5])
        return abs(abs(pts[0] - pts[1]) / abs(pts[2] - pts[3]) - num / den) < tol
    if name == 'ncoll':
        return not n_coll(*pts[:3], tol=1e-4)
    if name == 'diff':
        return abs(pts[0] - pts[1]) > 1e-4
    if name == 'npara':
        return not n_para(*pts[:4], tol=1e-4)
    if name == 'nperp':
        return not n_perp(*pts[:4], tol=1e-4)
    if name == 's_angle':
        return True
    raise NotImplementedError(name)
