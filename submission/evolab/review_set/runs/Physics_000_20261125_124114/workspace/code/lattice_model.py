"""E4: phenomenological lattice-level stochastic path model.
From shell (h,k), each outward nearest-neighbour step s is weighted by
   w_s = exp[-lam * N_s * (delta_actual - delta*_s)^2]
with lam = beta*K/2 (K: per-atom curvature of the LJ E(delta) scans,
beta = 1/T*); a random-step channel with probability p_r chooses uniformly.
delta*_s: LJ numerical value where available, else the analytic formula.
Outputs ../outputs/lattice_model.json"""
import json, numpy as np
import icoshell as ic

rng = np.random.default_rng(42)
SC = json.load(open('../outputs/scans_LJ.json'))


def curvature():
    """per-shell-atom curvature K from quadratic fits around minima of NN LJ scans."""
    Ks = []
    for core, v in SC.items():
        for c, r in v['results'].items():
            if not r['nn'] or r['edge_of_grid']:
                continue
            g = np.array(r['grid']); E = np.array(r['E'])
            i = int(np.argmin(E))
            sl = slice(max(i - 2, 0), min(i + 3, len(g)))
            if sl.stop - sl.start < 3:
                continue
            a = np.polyfit(g[sl], E[sl], 2)[0]
            Nsh = ic.N_of(*r['shell'])
            Ks.append(dict(core=core, shell=c, K_atom=2 * a / Nsh, Nsh=Nsh))
    return Ks


def dstar_table():
    tab = {}
    for core, v in SC.items():
        hk = tuple(v['core_path'][-1])
        for c, r in v['results'].items():
            if r['nn']:
                tab[(hk, tuple(r['shell']))] = r['delta_opt']
    return tab


NUM = dstar_table()


def canon(s):
    h, k = s
    return (0, max(h, k)) if min(h, k) == 0 else (h, k)


def dstar(a, b):
    a, b = canon(a), canon(b)
    for key in [(a, b), ((a[1], a[0]), (b[1], b[0]))]:
        if key in NUM:
            return NUM[key], 'LJ'
    return ic.delta_star(ic.T_of(*a), ic.T_of(*b)), 'formula'


def steps_from(s):
    h, k = s
    cand = [(h, k + 1), (h + 1, k), (h - 1, k + 1), (h + 1, k - 1)]
    T = ic.T_of(h, k)
    out = []
    for c in cand:
        if min(c) < 0 or ic.T_of(*c) <= T:
            continue
        cc = canon(c)
        if cc in [canon(o) for o in out]:
            continue
        out.append(c)
    return out


def stype(a, b):
    ca, cb = ic.class_index(*a), ic.class_index(*b)
    return 'conservative' if ca == cb else ('mismatch-driven' if cb > ca else 'reverse')


def run_paths(delta_seq, lam, p_r, npaths=500, start=(0, 1)):
    stats = dict(conservative=0, **{'mismatch-driven': 0}, random=0, reverse=0)
    paths, classes = [], []
    for _ in range(npaths):
        s = start
        p = [s]; cl = [ic.shell_class(*s)]
        for dlt in delta_seq:
            opts = steps_from(s)
            if rng.random() < p_r:
                j = rng.integers(len(opts)); kind = 'random'
            else:
                w = np.array([-lam * ic.N_of(*o) * (dlt - dstar(s, o)[0]) ** 2 for o in opts])
                w = np.exp(w - w.max()); w /= w.sum()
                j = rng.choice(len(opts), p=w)
                kind = stype(s, opts[j])
            stats[kind] += 1
            s = opts[j]
            p.append(s); cl.append(ic.shell_class(*s))
        paths.append(p); classes.append(cl)
    # most common path
    from collections import Counter
    cnt = Counter(tuple(map(tuple, map(canon, p))) for p in paths)
    return dict(stats=stats, top_paths=[(list(map(list, k)), v) for k, v in cnt.most_common(5)],
                class_traj=[[c for c in cl] for cl in classes[:50]],
                class_freq_by_step=[dict(Counter(cl[i] for cl in classes)) for i in range(len(delta_seq) + 1)])


if __name__ == '__main__':
    Ks = curvature()
    Kmed = float(np.median([k['K_atom'] for k in Ks]))
    Tstar = 0.22
    lam_th = Kmed / 2 / Tstar
    out = dict(K=Ks, K_median=Kmed, lam_thermo=lam_th, runs={})
    L = 5
    for lam_name, lam in [('thermo', lam_th), ('soft', lam_th / 30)]:
        for p_r in [0.0, 0.1]:
            for dl in [0.0, 0.04, 0.08, 0.14, 0.2]:
                key = '%s_pr%.1f_d%.2f' % (lam_name, p_r, dl)
                out['runs'][key] = run_paths([dl] * L, lam, p_r)
                out['runs'][key].update(lam=lam, p_r=p_r, delta=dl)
            # alkali-like decreasing sequence Na13 -> K -> Rb -> Cs, then continued small mismatch
            r = dict(Na=1.86, K=2.27, Rb=2.48, Cs=2.65)
            seq = [r['K'] / r['Na'] - 1, r['Rb'] / r['K'] - 1, r['Cs'] / r['Rb'] - 1]
            key = '%s_pr%.1f_alkali' % (lam_name, p_r)
            out['runs'][key] = run_paths(seq, lam, p_r)
            out['runs'][key].update(lam=lam, p_r=p_r, delta=seq)
            # 'mixed' scenario: Mackay-like growth then a large-mismatch species
            seq = [0.03, 0.03, 0.15, 0.03, 0.03]
            key = '%s_pr%.1f_mixed' % (lam_name, p_r)
            out['runs'][key] = run_paths(seq, lam, p_r)
            out['runs'][key].update(lam=lam, p_r=p_r, delta=seq)
    # table of delta* used for all outward NN steps up to T~40
    tab = []
    for h in range(0, 6):
        for k in range(h, 6):
            if (h, k) == (0, 0):
                continue
            for o in steps_from((h, k)):
                d, src = dstar((h, k), o)
                tab.append(dict(frm=[h, k], to=list(o), type=stype((h, k), o), dstar=float(d), src=src,
                                formula=float(ic.delta_star(ic.T_of(h, k), ic.T_of(*o))),
                                limit=float(ic.delta_star_limit(ic.T_of(h, k), ic.T_of(*o))),
                                cos=ic.step_cos((h, k), o)))
    out['dstar_table'] = tab
    json.dump(out, open('../outputs/lattice_model.json', 'w'), indent=1)
    for k, v in out['runs'].items():
        print(k, v['stats'], v['top_paths'][0])
    print('K median', Kmed, 'lam', lam_th)
