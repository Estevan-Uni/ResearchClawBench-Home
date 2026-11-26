import sys, os, time, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from train_marl import *
rng = np.random.default_rng(1)
st = make_state(rng, 0)
while st.cp < 20: st = make_state(rng, 0)
rec = alloc(300 * 8 + 8)
ph = Policy.load('../outputs/policy_heur.npz')
for mode, temp in [(1, 1.0), (0, 0.5), (0, 0.25)]:
    for Wn in [8, -1]:
        r3 = np.random.default_rng(5); L = []; t = time.perf_counter()
        for _ in range(30):
            S = nb_collision(st, r3, 8); l, nr, term = episode(st, S, r3, ph, Wn, mode, rec, temp=temp); L.append(l)
        print('HEUR mode', mode, 'temp', temp, 'W', Wn, 'local', np.mean(L), 'ms', (time.perf_counter() - t) / 30 * 1e3)
r3 = np.random.default_rng(5); L = []
for _ in range(30):
    S = nb_collision(st, r3, 8); L.append(pp_local(st, S, r3)[0])
print('PP same nbs', np.mean(L))
