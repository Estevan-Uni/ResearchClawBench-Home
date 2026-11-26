import sys, os, time, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from train_marl import *
rng = np.random.default_rng(1)
st = make_state(rng, 0)
while st.cp < 20: st = make_state(rng, 0)
print('inst', st.inst.N, st.inst.H, 'cp', st.cp)
pol = Policy(); rec = alloc(300 * 8 + 8)
for Wn in [8, -1]:
    t = time.perf_counter(); L = []
    for _ in range(20):
        S = nb_collision(st, rng, 8); l, nr, term = episode(st, S, rng, pol, Wn, 0, rec); L.append(l)
    print('MARL W', Wn, 'local', np.mean(L), 'ms', (time.perf_counter() - t) / 20 * 1e3, 'nrec', nr)
t = time.perf_counter(); L = []
for _ in range(20):
    S = nb_collision(st, rng, 8); L.append(pp_local(st, S, rng)[0])
print('PP local', np.mean(L), 'ms', (time.perf_counter() - t) / 20 * 1e3)
S = nb_collision(st, rng, 8); nr = bc_episode(st, S, rng, pol, -1, rec); print('bc', nr)
G = returns(rec[4], rec[5], nr, 8, 0, np.zeros(8)); print(G[:3])
ph = Policy.load('../outputs/policy_heur.npz')
for Wn in [8, -1]:
    t = time.perf_counter(); L = []
    r3 = np.random.default_rng(5)
    for _ in range(20):
        S = nb_collision(st, r3, 8); l, nr, term = episode(st, S, r3, ph, Wn, 0, rec); L.append(l)
    print('HEUR W', Wn, 'local', np.mean(L), 'ms', (time.perf_counter() - t) / 20 * 1e3)
r3 = np.random.default_rng(5); L = []
for _ in range(20):
    S = nb_collision(st, r3, 8); L.append(pp_local(st, S, r3)[0])
print('PP same nbs', np.mean(L))
