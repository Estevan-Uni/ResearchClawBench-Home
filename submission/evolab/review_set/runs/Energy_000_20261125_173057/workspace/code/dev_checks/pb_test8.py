import sys,time,pickle; sys.path.insert(0,'code')
import numpy as np
from ecat_model import *
E=pickle.load(open('outputs/exp_data.pkl','rb'))['Oxford']['dc']
t=E['t']-E['t'][0]; I=-E['I']*CAP_NOM/0.74
for dt,rt in [(5,1e-5),(10,1e-4)]:
    tk=np.arange(0,t[-1]+dt,dt); Ik=np.array([I[(t>=a)&(t<a+dt)].mean() if np.any((t>=a)&(t<a+dt)) else 0 for a in tk])
    m=ECAT(I_app=1.0,T_amb=313.15,V_cut=2.5,current=(tk,Ik),rtol=rt,atol=rt*1e-2)
    N=['R_n','R_p','k_n','h','LAM_p','R_f','LLI']
    for k in range(2):
        t0=time.time(); r=m.run([5e-6,3e-6,1,35,0.95,0.0,1.0],N,t_eval=np.arange(0,t[-1],1.0)); print(dt,rt,time.time()-t0, r['t'][-1], 1e3*np.sqrt(np.mean((np.interp(t,r['t'],r['V'])-E['V'])**2)),flush=True)
