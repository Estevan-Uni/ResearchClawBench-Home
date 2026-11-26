import sys,time; sys.path.insert(0,'code')
import numpy as np
from ecat_model import *
N=['R_n','R_p','k_n','h','LAM_p','R_f']; x=[6.906e-06,2.074e-06,0.3965,11.22,0.4067,0.1423]
m=ECAT(I_app=2.0,T_amb=297.15,V_cut=2.7)
r=m.run(x,N); print('dt2',r['t'][-3:],r['V'][-3:])
m.dt_out=0.25; r=m.run(x,N); print('dt0.25',r['t'][-3:],r['V'][-3:])
sol=m.sim.solve([0,9000],inputs=dict(m.fixed,**dict(zip(N,x)))); print('adaptive',sol.t[-3:],sol['Voltage [V]'].entries[-3:], sol.termination)
