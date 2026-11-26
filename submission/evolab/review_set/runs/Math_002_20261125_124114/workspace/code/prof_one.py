import sys, os, time, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from data_io import *; from solver import *; from validate import validate; from lacam import run_lacam
cfgs = {c[1]: c for c in list_configs()}
name = sys.argv[1]; tl = float(sys.argv[2]); idx = int(sys.argv[3]) if len(sys.argv) > 3 else 0
f = instance_files(cfgs[name][3])[idx]
g, s, t = load_instance(f); inst = Instance(g, s, t)
print(name, 'N', inst.N, 'T', inst.T, 'maxsp', inst.sp.max())
W = float(os.environ.get("W", 0)) or None
SL = os.environ.get("SL"); r = run_lns(inst, tl, seed=0, w=int(W) if W else None, slack=int(SL) if SL else None)
tr = r['trace']
print(f"LNS2 succ {r['success']} cp {r['cp']} it {r['iters']} it/s {r['iters']/max(1e-9, r['time']-r['t_init']):.0f} init {r['t_init']:.2f}s cp0 {tr[0][1]} soc {r['soc']}")
print(' trace', [(round(a, 1), b) for a, b, c in tr[::max(1, len(tr)//10)]])
if r['success']: print(validate(g, s, t, r['state'].paths()))
if len(sys.argv) > 4:
    r2 = run_lacam(inst, tl); print('lacam', r2['success'], r2['soc'], r2.get('nodes'), round(r2['time'], 2))
    if r2['success']: print(validate(g, s, t, r2['paths']))
