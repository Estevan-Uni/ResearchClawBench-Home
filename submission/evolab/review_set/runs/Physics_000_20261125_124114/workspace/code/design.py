"""Real-element design table: greedy 'nearest delta*' rule.
Mismatch convention: delta = r_out/r_in - 1 with the dataset's atomic radii.
From the current outer shell, the next shell is the outward NN step whose LJ
delta* (numerical where available, else formula) is closest to the actual delta.
Also audits the dataset's tabulated pairs/clusters.  -> ../outputs/design.json"""
import json, itertools, numpy as np
import icoshell as ic
from lattice_model import steps_from, dstar, canon, stype

RAD = dict([('Na', 1.86), ('K', 2.27), ('Rb', 2.48), ('Cs', 2.65), ('Ag', 1.44), ('Cu', 1.28), ('Ni', 1.24)])


def greedy(seq, start=(0, 1)):
    s = start
    path = [(0, 0), s]
    rows = []
    for a, b in zip(seq[:-1], seq[1:]):
        dl = RAD[b] / RAD[a] - 1
        opts = steps_from(s)
        ds = [dstar(s, o)[0] for o in opts]
        j = int(np.argmin([abs(dl - d) for d in ds]))
        rows.append(dict(inner=a, outer=b, delta=dl, options=[[list(canon(o)), float(d)] for o, d in zip(opts, ds)],
                         chosen=list(canon(opts[j])), dstar=float(ds[j]), err=float(abs(dl - ds[j])),
                         type=stype(s, opts[j])))
        s = opts[j]; path.append(s)
    Ns = [ic.N_of(*p) for p in path]
    name = seq[0] + '%d' % (Ns[0] + Ns[1]) + ''.join('@%s%d' % (e, n) for e, n in zip(seq[1:], Ns[2:]))
    return dict(seq=seq, path=[list(canon(p)) for p in path], N=Ns, name=name, rows=rows,
                classes=[ic.shell_class(*p) for p in path])


SC = json.load(open('../outputs/scans_LJ.json'))
CORE_OF = {(0, 1): 'c01', (1, 1): 'c11', (0, 2): 'c02', (1, 2): 'c12', (0, 3): 'c03', (2, 2): 'c22', (1, 3): 'c13'}


def window_rule(seed, dl):
    """lowest-Delta outer shell on the LJ stability map (incl. non-NN controls) at delta"""
    core = CORE_OF.get(tuple(seed))
    if core is None:
        return None
    res = dict(SC[core]['results'])
    if core == 'c03' and 'c03b' in SC:
        res.update(SC['c03b']['results'])
    best, bv = None, np.inf
    for c, r in res.items():
        g = np.array(r['grid']); D = np.array(r['Delta']); ok = np.array(r['shift']) < 3
        if dl < g.min() or dl > g.max():
            continue
        v = np.interp(dl, g[ok], D[ok])
        if v < bv:
            best, bv = tuple(r['shell']), v
    return best


if __name__ == '__main__':
    out = dict(pairs=[], sequences=[], audit={})
    els = sorted(RAD, key=RAD.get)
    # all pairs core13 @ shell
    for a, b in itertools.permutations(RAD, 2):
        if RAD[b] <= RAD[a]:
            continue
        g = greedy([a, b])
        w = window_rule((0, 1), RAD[b] / RAD[a] - 1)
        out['pairs'].append(dict(core=a, shell=b, delta=RAD[b] / RAD[a] - 1, name=g['name'],
                                 shell_hk=g['path'][-1], cls=g['classes'][-1], dstar=g['rows'][0]['dstar'],
                                 window=list(w) if w else None,
                                 window_name=('%s13@%s%d' % (a, b, ic.N_of(*w))) if w else None))
    for seq in [['Na', 'K', 'Rb', 'Cs'], ['Na', 'K', 'Cs'], ['Na', 'Rb', 'Cs'], ['Na', 'K', 'Rb'],
                ['Ni', 'Cu', 'Ag'], ['Ni', 'Ag'], ['Cu', 'Ag'], ['Ni', 'Cu'],
                ['Na', 'Na', 'K'], ['Ni', 'Ni', 'Ag'], ['Ni', 'Ni', 'Ni', 'Ag'], ['Ni', 'Ni', 'Cu', 'Ag'],
                ['Cu', 'Cu', 'Ag'], ['Na', 'K', 'K', 'Rb', 'Cs']]:
        out['sequences'].append(greedy(seq))
    # dataset audit
    ds_pairs = [('Na', 'Rb', 0.22), ('Ag', 'Cu', 0.12), ('Ag', 'Ni', 0.15), ('Cu', 'Ni', 0.032)]
    out['audit']['pairs'] = [dict(a=a, b=b, listed=v, from_radii=max(RAD[a], RAD[b]) / min(RAD[a], RAD[b]) - 1,
                                  NaK=RAD['K'] / RAD['Na'] - 1) for a, b, v in ds_pairs]
    out['audit']['magic_N_valid'] = {n: any(10 * ic.T_of(h, k) + 2 == n for h in range(8) for k in range(8))
                                     for n in [32, 42, 45, 72, 92, 122, 132, 162, 192]}
    json.dump(out, open('../outputs/design.json', 'w'), indent=1)
    for p in out['pairs']:
        print(p)
    for s in out['sequences']:
        print(s['name'], s['path'], s['classes'], [round(r['delta'], 3) for r in s['rows']],
              [round(r['dstar'], 3) for r in s['rows']])
    print(out['audit'])
