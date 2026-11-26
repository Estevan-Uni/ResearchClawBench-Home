import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from data_io import *; from solver import *
cfgs = {c[1]: c for c in list_configs()}
for name in ['maps_50_10_10_0.175', 'room_maps_250_25_25']:
    g, s, t = load_instance(instance_files(cfgs[name][3])[0]); inst = Instance(g, s, t)
    for pl, tie in [('x', 0), ('dp', 0), ('dp', 1)]:
        State.planner = pl; State.tie = tie; State.use_sipp = False
        st = State(inst); st.init_soft_pp(np.random.default_rng(0), inst.T + 1)
        print(name, pl, tie, 'cp0', st.cp, 'soc', st.soc(), 'sum_sp', inst.sp.sum())
