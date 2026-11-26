import sys, os, time, numpy as np, cProfile, pstats
sys.path.insert(0, os.path.dirname(__file__))
from data_io import *; from solver import *
cfgs = {c[1]: c for c in list_configs()}
f = instance_files(cfgs[sys.argv[1]][3])[0]
g, s, t = load_instance(f); inst = Instance(g, s, t)
run_lns(inst, 1, seed=0)
cProfile.run('r = run_lns(inst, float(sys.argv[2]), seed=0)', '/tmp/prof.out')
print(r['iters'], r['t_init'])
pstats.Stats('/tmp/prof.out').sort_stats('tottime').print_stats(12)
