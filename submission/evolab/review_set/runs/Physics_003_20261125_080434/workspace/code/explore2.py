import h5py,numpy as np
from scipy.optimize import curve_fit
f=h5py.File('data/raw_trARPES_data.h5'); E=f['energy_axis'][:]; k=f['kx_axis'][:]
off=f['pump_off_spectrum'][:]
def two(k,a,k0,g,b,c): return a*g**2/((k-k0)**2+g**2)+a*g**2/((k+k0)**2+g**2)+b+c*k**2
for e in [-0.45,-0.3,-0.2,-0.1,-0.05,0.05,0.1,0.2,0.3,0.45]:
    i=np.argmin(abs(E-e)); y=off[i]
    p,_=curve_fit(two,k,y,p0=[y.max(),abs(e)/6+0.005,0.01,1,0],maxfev=20000)
    print(f'{E[i]:.3f}',np.round(p,4), 'resid rms',np.std(y-two(k,*p)).round(3), 'ratio k0/E',round(abs(E[i])/abs(p[1]),3))
# noise check: difference between nominally identical angles
d0=f['pump_on_angle_0'][:]-f['pump_on_angle_180'][:]; print('0-180 diff std',d0.std(), 'vs sqrt(mean)',np.sqrt(f['pump_on_angle_0'][:].mean()))
d=f['pump_on_angle_30'][:]-f['pump_on_angle_60'][:]; print('30-60 std',d.std())
d=f['pump_on_angle_0'][:]-f['pump_on_angle_90'][:]; print('0-90 std',d.std(), d.mean())
# noise structure vs intensity
r=(f['pump_on_angle_0'][:]-f['pump_on_angle_180'][:]); I=f['pump_on_angle_0'][:]
for lo,hi in [(0,2),(2,5),(5,20),(20,60),(60,300)]:
    m=(I>lo)&(I<hi); print(lo,hi,r[m].std()/np.sqrt(2), np.sqrt(I[m].mean()))
