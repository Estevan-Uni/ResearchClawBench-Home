import sys,time; sys.path.insert(0,'code')
t0=time.time()
from ecat_model import *
print("import",time.time()-t0,flush=True)
m=ECAT(V_cut=3.0); print("build",time.time()-t0,flush=True)
r=m.run([5e-6,3e-6,1,35],CORE,); print("run",time.time()-t0, None if r is None else (r['t'][-1],r['V'][-1]),flush=True)
for x in [[2.5e-6,6e-6,0.2,5],[1e-5,1.5e-6,5,40],[5e-6,3e-6,1,20]]:
    t1=time.time(); r=m.run(x,CORE,); print("run",time.time()-t1, None if r is None else (r['t'][-1],r['V'][0],r['V'][-1],r['T'].max()-r['T'][0]),flush=True)
m2=ECAT(V_cut=2.7); t1=time.time(); r=m2.run([5e-6,3e-6,1,35],CORE,); print("2.7 first",time.time()-t1, None if r is None else (r['t'][-1],r['V'][-1]))
t1=time.time(); r=m2.run([5e-6,3e-6,1,35],CORE,); print("2.7 second",time.time()-t1)
