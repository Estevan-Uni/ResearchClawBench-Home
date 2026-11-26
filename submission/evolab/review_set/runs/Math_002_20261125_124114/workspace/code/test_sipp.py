# compare soft SIPP vs space-time A* on random states: objective (collisions, arrival)
import sys, os, numpy as np, time
sys.path.insert(0, os.path.dirname(__file__))
from data_io import *; from solver import *
cfgs = {c[1]: c for c in list_configs()}
for name in sys.argv[1:]:
    f = instance_files(cfgs[name][3])[0]; g, s, t = load_instance(f); inst = Instance(g, s, t)
    rng = np.random.default_rng(0); st = State(inst); st.use_sipp = False; st.planner = 'x'
    st.init_soft_pp(rng, inst.T + 1)
    # check occ2 == occ.T
    assert (st.occ2 == st.occ.T).all()
    res = []; tA = tS = 0
    for a in rng.choice(inst.N, 40, replace=False):
        st.remove(a)
        st.use_sipp = False; t1 = time.perf_counter(); st.plan(a, inst.T + 1); tA += time.perf_counter() - t1
        cA, aA = st.last_c, st.arr[a]; pA = st.P[:, a].copy()
        st.planner = 'dp'; t1 = time.perf_counter(); st.plan(a, inst.T + 1); tS += time.perf_counter() - t1; st.planner = 'x'
        cS, aS = st.last_c, st.arr[a]
        # verify SIPP path: valid moves & recount collisions
        p = st.P[:, a]; ok = all(p[t+1] in inst.nbr[p[t]] for t in range(inst.T)) and p[0] == inst.starts[a] and p[-1] == inst.goals[a]
        cnt = sum(st.occ[t, p[t]] for t in range(1, inst.T + 1))
        for t in range(inst.T):
            if p[t] != p[t+1]:
                k = move_dir(inst.nbr, p[t], p[t+1]); sl = k if k % 2 == 1 else k - 2
                cnt += st.eout[t, p[t+1], sl]
        res.append((cA, aA, cS, aS, ok, cnt))
        st.insert(a)
    r = np.array(res)
    print(name, 'A* ms %.2f DP ms %.2f' % (tA / 40 * 1e3, tS / 40 * 1e3), 'valid', r[:, 4].all(), 'recount==c', (r[:, 5] == r[:, 2]).mean(),
          'mean c A* %.2f SIPP %.2f' % (r[:, 0].mean(), r[:, 2].mean()), 'arr A* %.1f SIPP %.1f' % (r[:, 1].mean(), r[:, 3].mean()),
          'SIPP worse c frac', (r[:, 2] > r[:, 0]).mean())
