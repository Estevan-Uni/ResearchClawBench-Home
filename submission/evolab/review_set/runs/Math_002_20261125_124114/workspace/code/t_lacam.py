import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from data_io import *; from solver import Instance; from lacam import *; from validate import validate
cfgs = {c[1]: c for c in list_configs()}
f = instance_files(cfgs[sys.argv[1]][3])[0]
g, s, t = load_instance(f); inst = Instance(g, s, t)
r = run_pibt(inst, 5); print('pibt', r['success'], r['steps'], r['cp'], flush=True)
r2 = run_lacam(inst, float(sys.argv[2])); print('lacam', r2['success'], r2['soc'], r2.get('nodes'), round(r2['time'], 2), flush=True)
if r2['success']: print(validate(g, s, t, r2['paths']))
