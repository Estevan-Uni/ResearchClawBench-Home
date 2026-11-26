import h5py,numpy as np
f=h5py.File('data/raw_trARPES_data.h5'); E=f['energy_axis'][:]; k=f['kx_axis'][:]
off=f['pump_off_spectrum'][:]
KK,EE=np.meshgrid(k,E)
def cone(ED,v=6.0,g=1/60,A=100):
    k0=np.abs(EE-ED)/v
    return A*(g**2/((KK-k0)**2+g**2)+g**2/((KK+k0)**2+g**2))
print('model-off max err',np.abs(cone(0)-off).max())
on=f['pump_on_angle_0'][:]
for w in [0.20,0.24,0.248,0.25,0.26,0.30]:
    X=np.stack([cone(0).ravel(),cone(w).ravel(),cone(-w).ravel(),cone(2*w).ravel(),cone(-2*w).ravel(),np.ones(EE.size)],1)
    c,res,*_=np.linalg.lstsq(X,on.ravel(),rcond=None)
    r=on.ravel()-X@c
    print(w,np.round(c,4),'rms',r.std().round(4))
w=0.248
X=np.stack([cone(0).ravel(),cone(w).ravel(),cone(-w).ravel(),np.ones(EE.size)],1)
c,*_=np.linalg.lstsq(X,on.ravel(),rcond=None); r=(on.ravel()-X@c).reshape(on.shape)
print(c, r.std()); 
import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
plt.pcolormesh(k,E,r,cmap='RdBu_r');plt.colorbar();plt.savefig('outputs/resid_try.png',dpi=60)
