import sys, os, time, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from data_io import *; from solver import *; from validate import validate
from train_marl import *
f = instance_files(os.path.join(DATA, 'random_small', 'maps_50_10_10_0.175'))[1]
g, s, t = load_instance(f); inst = Instance(g, s, t)
rng = np.random.default_rng(0); st = State(inst); st.init_soft_pp(rng, inst.T + 1)
pol = Policy()
rec = alloc(100)
S = nb_collision(st, rng, 8)
print('cp', st.cp, 'S', S)
l, nr, term = episode(st, S, rng, pol, 8, 0, rec); print('marl local', l, nr, term)
print('rewards', rec[4][:16], rec[5][:16])
G = returns(rec[4], rec[5], nr, 8, 8, term); print('G', G[:8])
print('pp local', pp_local(st, S, rng)[0])
nr = bc_episode(st, S, rng, pol, 8, rec); print('bc recs', nr, rec[2][:16])
logp, V, cache = pol.forward(rec[0][:nr], rec[1][:nr]); print(logp.shape, V.shape)
# gradient check
ps = pol.params(); dl = np.random.randn(*logp.shape) * (rec[1][:nr] > 0); dV = np.random.randn(nr)
gs = pol.backward(cache, dl, dV)
def L():
    lp, v, _ = pol.forward(rec[0][:nr], rec[1][:nr]); return (np.where(rec[1][:nr] > 0, lp, 0) * dl).sum() + (v * dV).sum()
base = L(); eps = 1e-6
for pi in [0, 2, 4]:
    P = pol.params(); P[pi].flat[3] += eps; pol.set(P); num = (L() - base) / eps; P[pi].flat[3] -= eps; pol.set(P)
    print('gradcheck', pi, num, gs[pi].flat[3])
for ctrl in [Fixed(1), Patience(5), RateSwitch(), Bandit()]:
    r = run_lns(inst, 8, seed=0, policy=pol, Wn=4, controller=ctrl)
    print(type(ctrl).__name__, r['success'], r['cp'], r['iters'], r['n_pp'], r['n_marl'], r['switch_time'])
    if r['success']: print(validate(g, s, t, r['state'].paths()))
