"""Resumable experiment runner: python run_exp.py METHOD [subset] ; writes outputs/res_<METHOD>.jsonl"""
import sys, os, json, time, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from data_io import *; from solver import *; from validate import validate
from lacam import run_lacam, run_pibt
from marl import Policy
from core import arrival_time
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'outputs')

def budget(H):
    return {10: 10.0, 25: 40.0, 50: 60.0}[H]

def n_inst(fam, method):
    base = {'random_small': 16, 'random_large': 3}.get(fam, 8)          # main methods (LNS2, LNS2+RL, PIBT)
    if method == 'PP': base = {'random_small': 4, 'random_large': 1}.get(fam, 2)
    if method == 'LaCAM': base = {'random_small': 8, 'random_large': 1}.get(fam, 3)
    if method == 'MARL': base = {'random_small': 4, 'random_large': 0}.get(fam, 2)
    if method.startswith('LNS2_seed1'): base = {'random_small': 8, 'random_large': 1}.get(fam, 3)
    if method.startswith('ABL_'): base = {'random_small': 6, 'random_large': 0, 'empty': 0}.get(fam, 3)
    return base

def tasks(method, fams=None):
    out = []
    for fam, name, n, d in list_configs():
        if fams and fam not in fams: continue
        fs = instance_files(d)
        for i in range(n_inst(fam, method)):
            out.append((fam, name, n, i, fs[i]))
    # interleave configs so partial results cover all families
    out.sort(key=lambda x: (x[3], FAMILIES.index(x[0]), x[2]))
    return out

_pol = {}
def policy(path):
    if path not in _pol: _pol[path] = Policy.load(path)
    return _pol[path]

def run_method(method, inst, tl, g, s, t):
    seed = 1 if 'seed1' in method else 0
    if method in ('LNS2', 'LNS2_seed1'):
        r = run_lns(inst, tl, seed=seed); paths = r['state'].paths() if r['success'] else None
    elif method == 'PP':
        r = run_pp(inst, tl, seed=seed); paths = r['state'].paths() if r['success'] else None
    elif method == 'LaCAM':
        r = run_lacam(inst, tl, seed=seed); paths = r['paths']
    elif method == 'PIBT':
        r = run_pibt(inst, tl, seed=seed); paths = r['paths']
    elif method == 'MARL':
        r = run_pure_marl(inst, tl, policy(os.path.join(OUT, 'policy.npz'))); paths = r.pop('paths')
    else:
        # hybrid variants: name like HYB_<rule>_W<w>[_pol]
        cfg = HYB[method]
        pol = policy(os.path.join(OUT, cfg.get('policy', 'policy.npz')))
        ctrl = cfg['ctrl']()
        r = run_lns(inst, tl, seed=seed, policy=pol, Wn=cfg['W'], controller=ctrl, marl_mode=cfg.get('mode', 1))
        paths = r['state'].paths() if r['success'] else None
    rec = {k: v for k, v in r.items() if k not in ('state', 'paths', 'trace')}
    if 'trace' in r:
        tr = r['trace']; step = max(1, len(tr) // 60)
        rec['trace'] = [(round(a, 3), int(b)) + ((int(c[0]),) if c else ()) for a, b, *c in tr[::step]] + [tuple(tr[-1][:2])]
    if paths is not None:
        v = validate(g, s, t, paths); rec['valid'] = v['valid']
        if not v['valid']: rec['success'] = False; rec['val'] = v
    return rec

def run_pure_marl(inst, tl, pol, seed=0):
    """Pure decentralised MARL (SCRIMP-style proxy, no search): all agents act with the shared policy for
    T steps, no background; success iff every agent ends at its goal and there are no collisions."""
    t0 = time.process_time(); rng = np.random.default_rng(seed)
    if inst.N * (inst.T + 1) * inst.C * 4 > 1.2e9:
        return dict(success=False, cp=-1, soc=-1, at_goal=-1, time=0.0, skipped=True, paths=None)
    st = State(inst); S = np.arange(inst.N); Wn = inst.T
    outP = marl_window(st, S, pol, Wn, seed, 0, 1.0)
    st.P[:Wn + 1] = outP; st.P[Wn + 1:] = outP[-1]
    for a in S:
        st.arr[a] = arrival_time(st.P, a, inst.goals[a], inst.T); st.insert(a)
    st.attach(S)
    at = int((outP[-1] == inst.goals).sum())
    ok = bool(at == inst.N and st.cp == 0)
    return dict(success=ok, cp=int(st.cp), soc=st.soc() if ok else -1, at_goal=at, time=time.process_time() - t0,
                paths=st.paths() if ok else None)

HYB = {
    'HYB_rate_Wfull': dict(W=-1, ctrl=lambda: RateSwitch()),
    'HYB_pat_Wfull': dict(W=-1, ctrl=lambda: Patience(10)),
    'HYB_never_Wfull': dict(W=-1, ctrl=lambda: Fixed(1)),
    'HYB_bandit_Wfull': dict(W=-1, ctrl=lambda: Bandit()),
    'HYB_rate_W8': dict(W=8, ctrl=lambda: RateSwitch()),
    'HYB_rate_W16': dict(W=16, ctrl=lambda: RateSwitch()),
    'HYB_rate_Wfull_heur': dict(W=-1, ctrl=lambda: RateSwitch(), policy='policy_heur.npz'),
    'HYB_never_W8': dict(W=8, ctrl=lambda: Fixed(1)),
    'HYB_pat_W8': dict(W=8, ctrl=lambda: Patience(10)),
    'HYB_bandit_W8': dict(W=8, ctrl=lambda: Bandit()),
    'HYB_rate_W8_heur': dict(W=8, ctrl=lambda: RateSwitch(), policy='policy_heur.npz'),
    'HYB_rate_W8_bc': dict(W=8, ctrl=lambda: RateSwitch(), policy='policy_bc.npz'),
    'HYB_rate_Wfull_bc': dict(W=-1, ctrl=lambda: RateSwitch(), policy='policy_bc.npz'),
}
for k in list(HYB): HYB['ABL_' + k] = HYB[k]

if __name__ == '__main__':
    method = sys.argv[1]
    fams = sys.argv[2].split(',') if len(sys.argv) > 2 and sys.argv[2] != 'all' else None
    part = sys.argv[3] if len(sys.argv) > 3 else None    # 'k/n' shard
    fn = os.path.join(OUT, f'res_{method}.jsonl')
    done = set()
    if os.path.exists(fn):
        for line in open(fn):
            d = json.loads(line); done.add((d['config'], d['idx']))
    T = tasks(method, fams)
    if part:
        k, n = map(int, part.split('/')); T = T[k::n]
    for fam, name, n, i, f in T:
        if (name, i) in done: continue
        g, s, t = load_instance(f); inst = Instance(g, s, t)
        tl = budget(g.shape[0])
        t0 = time.time()
        rec = run_method(method, inst, tl, g, s, t)
        rec.update(method=method, family=fam, config=name, agents=n, idx=i, budget=tl, sum_sp=int(inst.sp.sum()),
                   wall=time.time() - t0)
        with open(fn, 'a') as fh: fh.write(json.dumps(rec, default=lambda o: o.item() if hasattr(o, 'item') else str(o)) + '\n')
        print(method, name, i, 'succ', rec['success'], 'cp', rec.get('cp'), 'soc', rec.get('soc'), 't', round(rec['time'], 1), flush=True)
