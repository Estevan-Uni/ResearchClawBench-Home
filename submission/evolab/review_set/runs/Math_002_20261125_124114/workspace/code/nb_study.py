"""Neighbourhood study: identical LNS states and identical neighbourhoods, compare repair operators.
Early state = right after the soft-PP initial solution; late state = after `late_s` seconds of LNS2."""
import sys, os, json, time, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from data_io import *; from solver import *; from marl import Policy
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'outputs')

def repair_eval(st, S, rng, op):
    I = st.inst; w = I.T + 1
    oldP = st.P[:, S].copy(); oldarr = st.arr[S].copy(); cp0 = st.cp
    st.detach(S)
    for a in S: st.remove(a)
    before = cp0 - st.cp; cprest = st.cp
    t = time.perf_counter()
    op(st, S, rng)
    st.attach(S); dt = time.perf_counter() - t
    after = st.cp - cprest
    st.detach(S)
    for a in S: st.remove(a)
    st.P[:, S] = oldP; st.arr[S] = oldarr
    for a in S: st.insert(a)
    st.attach(S); assert st.cp == cp0
    return before, after, dt

def main(configs, n_nb=30, late_s=15.0):
    pols = {k: Policy.load(os.path.join(OUT, f)) for k, f in
            [('learned', 'policy.npz'), ('bc', 'policy_bc.npz'), ('heur', 'policy_heur.npz')]}
    ops = {'PP': lambda s, S, r: repair_pp(s, S, r, s.inst.T + 1)}
    for pk in ['learned', 'bc', 'heur']:
        for W in [8, 16, -1]:
            if pk != 'learned' and W != -1: continue
            ops[f'MARL-{pk}-W{"full" if W < 0 else W}'] = (lambda pk, W: lambda s, S, r: repair_marl(s, S, r, pols[pk], W, s.inst.T + 1))(pk, W)
    cfgs = {c[1]: c for c in list_configs()}
    fn = os.path.join(OUT, 'nb_study.jsonl')
    for name in configs:
        fam, _, n, d = cfgs[name]
        f = instance_files(d)[50 if len(instance_files(d)) > 50 else 40]   # instance not used in main eval
        g, s, t = load_instance(f); inst = Instance(g, s, t)
        rng = np.random.default_rng(7)
        st = State(inst); st.init_soft_pp(rng, inst.T + 1)
        for stage in ['early', 'late']:
            if stage == 'late':
                nbw = np.ones(len(NB_FUNCS)); t0 = time.perf_counter()
                while time.perf_counter() - t0 < late_s and st.cp > 0:
                    lns_step(st, rng, ops['PP'], 8, nbw, inst.T + 1)
            if st.cp == 0: continue
            nbs = [nb_collision(st, rng, 8) if q % 2 == 0 else nb_random(st, rng, 8) for q in range(n_nb)]
            for k, op in ops.items():
                r2 = np.random.default_rng(11); res = [repair_eval(st, S, r2, op) for S in nbs]
                R = np.array(res)
                rec = dict(config=name, family=fam, stage=stage, op=k, cp_state=int(st.cp),
                           before=float(R[:, 0].mean()), after=float(R[:, 1].mean()),
                           p_improve=float((R[:, 1] < R[:, 0]).mean()), p_accept=float((R[:, 1] <= R[:, 0]).mean()),
                           gain=float(np.maximum(R[:, 0] - R[:, 1], 0).mean()), ms=float(R[:, 2].mean() * 1e3))
                rec['gain_per_s'] = rec['gain'] / (rec['ms'] / 1e3)
                print(rec, flush=True)
                with open(fn, 'a') as fh: fh.write(json.dumps(rec) + '\n')

if __name__ == '__main__':
    main(sys.argv[1].split(','), late_s=float(sys.argv[2]) if len(sys.argv) > 2 else 15.0)
