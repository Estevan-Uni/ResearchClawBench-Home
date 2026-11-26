import sys; sys.path.insert(0,'code')
import numpy as np
from ecat_model import *
N=['R_n','R_p','k_n','h','LAM_p','R_f']; x=[8.63018348e-06,1.80806574e-06,1.44411291e+00,2.51794262e+01,8.45792189e-01,1.11071341e-01]
m=ECAT(I_app=2.0,T_amb=297.15,V_cut=2.7)
sol=m.sim.solve([0,m.t_max],t_interp=np.arange(0,m.t_max,2.0),inputs=dict(m.fixed,**dict(zip(N,x))))
t=sol['Time [s]'].entries; V=sol['Voltage [V]'].entries; print('raw',t[-4:],V[-4:],sol.t[-1])
r=m.run(x,N); print('fixed',r['t'][-3:],r['V'][-3:]); print(features(r)[0][-3:], features(r)[2])
