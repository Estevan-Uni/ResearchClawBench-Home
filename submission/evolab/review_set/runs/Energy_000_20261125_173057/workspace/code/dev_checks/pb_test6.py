import sys,time; sys.path.insert(0,'code')
from ecat_model import *
m=ECAT(I_app=2.0,T_amb=297.15,V_cut=2.7)
N=['R_n','R_p','k_n','h','LLI','LAM_p','R_f']
for x in [[5e-6,3e-6,1,10,1,1,0],[5e-6,3e-6,1,10,1,0.8,0],[5e-6,3e-6,1,10,0.9,1,0],[5e-6,3e-6,1,10,1,0.6,0.05],[5e-6,3e-6,1,10,0.85,0.6,0.08],[5e-6,3e-6,1,10,1.02,1.05,0]]:
    r=m.run(x,N); print(x, None if r is None else "Q %.3f V0 %.3f dT %.2f t %.0f"%(r['Q'],r['V'][0],r['T'].max()-r['T'][0],r['t'][-1]),flush=True)
