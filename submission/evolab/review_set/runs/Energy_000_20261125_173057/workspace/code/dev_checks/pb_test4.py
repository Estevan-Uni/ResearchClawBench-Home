import sys; sys.path.insert(0,'code')
from ecat_model import *
for vc in [3.0,2.7]:
    m=ECAT(V_cut=vc)
    for x in [[5e-6,3e-6,1,35],[2.5e-6,6e-6,0.2,5],[1e-5,1.5e-6,5,40]]:
        t0=time.time(); r=m.run(x,CORE); 
        print(vc, x, "%.3fs"%(time.time()-t0), None if r is None else (r['t'][-1], r['V'][0], r['V'][-1], r['T'].max()-r['T'][0], r['Q']))
