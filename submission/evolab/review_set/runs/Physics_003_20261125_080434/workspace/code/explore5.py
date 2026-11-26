import h5py,numpy as np
from scipy.optimize import least_squares
f=h5py.File('data/raw_trARPES_data.h5'); E=f['energy_axis'][:]; k=f['kx_axis'][:]
off=f['pump_off_spectrum'][:]; on=f['pump_on_angle_0'][:]
KK,EE=np.meshgrid(k,E)
def cone(ED,v,g,A,shape='L'):
    k0=np.abs(EE-ED)/v
    if shape=='L': return A*(g**2/((KK-k0)**2+g**2)+g**2/((KK+k0)**2+g**2))
    return A*(np.exp(-(KK-k0)**2/(2*g**2))+np.exp(-(KK+k0)**2/(2*g**2)))
def model(p,shape):
    A0,g0,A1,g1,w,v,B,sk,c=p
    return cone(0,v,g0,A0)+cone(w,v,g1,A1,shape)+cone(-w,v,g1,A1,shape)+B*np.exp(-KK**2/(2*sk**2))+c
for shape in ['L','G']:
    p0=[100,1/60,9,0.02,0.248,6,1,0.1,0]
    r=least_squares(lambda p:(model(p,shape)-on).ravel(),p0)
    print(shape,np.round(r.x,5),'rms',np.std(r.fun).round(4))
    res=(on-model(r.x,shape))
    import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
    plt.figure();plt.pcolormesh(k,E,res,cmap='RdBu_r');plt.colorbar();plt.savefig(f'outputs/resid_{shape}.png',dpi=60)
