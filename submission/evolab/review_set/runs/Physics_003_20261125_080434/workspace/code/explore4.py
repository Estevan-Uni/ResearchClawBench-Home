import h5py,numpy as np
from scipy.optimize import curve_fit
f=h5py.File('data/raw_trARPES_data.h5'); E=f['energy_axis'][:]; k=f['kx_axis'][:]
off=f['pump_off_spectrum'][:]; on=f['pump_on_angle_0'][:]; d=on-off
def L(x,x0,g): return g**2/((x-x0)**2+g**2)
def G(x,x0,s): return np.exp(-(x-x0)**2/(2*s**2))
for e in [-0.45,-0.35,-0.25,0.0,0.12,0.25,0.35,0.45]:
    i=np.argmin(abs(E-e)); y=d[i]
    # print MDC of diff coarse
    print(f'{E[i]:.3f}', np.round(y[45:105:3],2))
