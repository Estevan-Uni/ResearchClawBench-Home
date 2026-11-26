"""Run growth experiments.  usage: python run_growth_exp.py EXPNAME [EXPNAME ...]
Each run is saved to ../outputs/growth/<exp>_s<seed>.json as soon as it finishes."""
import sys, os, json, numpy as np, time
import icoshell as ic, md_growth as md

OUT = '../outputs/growth'
os.makedirs(OUT, exist_ok=True)


def load_joint(name):
    J = json.load(open('../outputs/joint_seeds.json'))
    return J[name]


def exp_cfg(name):
    if name.startswith('E1_') or name.startswith('E3a'):
        # E1_d{delta*100}_T{T*100}_tau{tau}
        parts = dict(p.split('=') for p in name.split('_')[1:])
        delta = float(parts['d']); T = float(parts['T']); tau = float(parts['tau'])
        nseeds = int(parts.get('n', 8)); ndep = int(parts.get('N', 42))
        return dict(seed_path=[(0, 0), (0, 1)], seed_d=[1.0, 1.0], delta=delta, T=T, tau=tau,
                    nseeds=nseeds, ndep=ndep, cands=[(0, 2), (1, 1)], ident_every=6,
                    id_centre='auto', layer='radial' if delta < 0.1 else 'index')
    if name.startswith('E3b'):
        parts = dict(p.split('=') for p in name.split('_')[1:])
        js = load_joint('p_011')
        return dict(seed_path=[(0, 0), (0, 1), (1, 1)], seed_d=js['diams'], delta=float(parts['d']),
                    T=float(parts['T']), tau=float(parts['tau']), nseeds=int(parts.get('n', 3)),
                    ndep=int(parts.get('N', 72)), cands=[(1, 2), (2, 1), (0, 2)], ident_every=8)
    if name.startswith('E2_'):
        parts = dict(p.split('=') for p in name.split('_')[1:])
        js = load_joint('p_01112')
        return dict(seed_path=[(0, 0), (0, 1), (1, 1), (1, 2)], seed_d=js['diams'],
                    delta=float(parts['d']), T=float(parts['T']), tau=float(parts['tau']),
                    nseeds=int(parts.get('n', 2)), ndep=int(parts.get('N', 132)),
                    cands=[(1, 3), (3, 1), (2, 2), (0, 3)], ident_every=11)
    if name.startswith('E2b'):
        parts = dict(p.split('=') for p in name.split('_')[1:])
        js = load_joint('p_0111213')
        return dict(seed_path=[(0, 0), (0, 1), (1, 1), (1, 2), (1, 3)], seed_d=js['diams'],
                    delta=float(parts['d']), T=float(parts['T']), tau=float(parts['tau']),
                    nseeds=int(parts.get('n', 2)), ndep=int(parts.get('N', 192)),
                    cands=[(2, 3), (3, 2), (1, 4), (4, 1), (0, 4)], ident_every=16)
    if name.startswith('E3c'):
        # dataset alternating sequence on an Ag13 seed: Cu20 + Ag10 + Cu20  (radii Ag 1.44, Cu 1.28)
        parts = dict(p.split('=') for p in name.split('_')[1:])
        dAg, dCu = 1.0, 1.28 / 1.44
        seq = [dCu] * 20 + [dAg] * 10 + [dCu] * 20
        return dict(seed_path=[(0, 0), (0, 1)], seed_d=[dAg, dAg], delta=None, seq=seq,
                    T=float(parts['T']), tau=float(parts['tau']), nseeds=int(parts.get('n', 3)),
                    ndep=len(seq), cands=[(0, 2), (1, 1)], ident_every=5, id_centre='auto')
    raise ValueError(name)


def main(name, seeds=None):
    cfg = exp_cfg(name)
    sp, sd = cfg['seed_path'], list(cfg['seed_d'])
    if cfg.get('seq') is not None:
        d_new = float(np.mean(cfg['seq']))
    else:
        d_new = sd[-1] * (1 + cfg['delta'])
    E0, Xs, Ds, Is = md.relaxed_template(sp, sd)
    ident = md.Identifier(sp, sd, d_new, cfg['cands'])
    ideal = {str(c): dict(E=v['E'], N=v['N']) for c, v in ident.cands.items()}
    log = open(os.path.join(OUT, name + '.log'), 'a')
    for s in (seeds or range(cfg['nseeds'])):
        fn = os.path.join(OUT, '%s_s%d.json' % (name, s))
        if os.path.exists(fn):
            continue
        t = time.time()
        r = md.run_growth(Xs, Ds, d_new, cfg['ndep'], T=cfg['T'], tau_dep=cfg['tau'],
                          rng=1000 + s, ident=ident, ident_every=cfg['ident_every'],
                          d_dep_seq=cfg.get('seq'), log=log,
                          id_centre=cfg.get('id_centre'), layer=cfg.get('layer', 'index'))
        r.update(cfg={k: v for k, v in cfg.items()}, name=name, seed=s, d_new=d_new,
                 E_seed=E0, ideal=ideal)
        json.dump(r, open(fn, 'w'))
        print(name, s, 'done %.0fs' % (time.time() - t), 'grown', r.get('id_grown', {}).get('cov'),
              'anneal', r.get('id_anneal', {}).get('cov'), flush=True)


if __name__ == '__main__':
    for n in sys.argv[1:]:
        main(n)
