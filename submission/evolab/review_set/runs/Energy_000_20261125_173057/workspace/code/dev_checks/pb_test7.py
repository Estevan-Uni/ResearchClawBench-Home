import sys,time,pickle; sys.path.insert(0,'code')
import numpy as np
from ecat_model import *
E=pickle.load(open('outputs/exp_data.pkl','rb'))['Oxford']['dc']
t=E['t']-E['t'][0]; I=-E['I']*CAP_NOM/0.74
print(np.unique(np.diff(t))[:5], E['V'][:3], E['T'][:3], E['q'][:3])
m=ECAT(I_app=1.0,T_amb=313.15,V_cut=2.5,current=(t,I))
N=['R_n','R_p','k_n','h','LAM_p','R_f','LLI']
for x in [[5e-6,3e-6,1,35,0.95,0.0,1.0],[5e-6,3e-6,1,35,0.95,0.02,0.98]]:
    t0=time.time(); r=m.run(x,N,t_eval=np.arange(0,t[-1],1.0)); print(time.time()-t0, None if r is None else (r['t'][-1], r['V'][0], r['V'].min(), np.sqrt(np.mean((np.interp(t,r['t'],r['V'])-E['V'])**2)), r['T'].max()-r['T'][0]))
