import sys, os, time, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from data_io import *; from solver import *; from validate import validate; from lacam import run_lacam, run_pibt
f = instance_files(os.path.join(DATA, 'random_small', 'maps_50_10_10_0.175'))[int(sys.argv[1]) if len(sys.argv) > 1 else 0]
g, s, t = load_instance(f)
inst = Instance(g, s, t); print('N', inst.N, 'T', inst.T)
r = run_lns(inst, float(sys.argv[2]) if len(sys.argv) > 2 else 20, seed=0)
print('LNS2', r['success'], r['cp'], r['soc'], r['iters'], round(r['time'], 2), 'init', round(r['t_init'], 2), r['trace'][0])
if r['success']: print(validate(g, s, t, r['state'].paths()))
r2 = run_lacam(inst, 10); print('lacam', r2['success'], r2['soc'], r2.get('nodes'), round(r2['time'], 2))
if r2['success']: print(validate(g, s, t, r2['paths']))
r3 = run_pibt(inst, 10); print('pibt', r3['success'], r3['soc'], r3['steps'])
if r3['success']: print(validate(g, s, t, r3['paths']))
r4 = run_pp(inst, 5); print('pp', r4['success'], r4['restarts'], r4['planned'])
if r4['success']: print(validate(g, s, t, r4['state'].paths()))
