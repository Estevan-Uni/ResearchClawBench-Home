import h5py,numpy as np,matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
f=h5py.File('data/raw_trARPES_data.h5'); E=f['energy_axis'][:]; k=f['kx_axis'][:]
off=f['pump_off_spectrum'][:]
fig,ax=plt.subplots(2,4,figsize=(16,8))
ax[0,0].pcolormesh(k,E,off,shading='auto'); ax[0,0].set_title('off')
for i,a in enumerate([0,30,60,90,120,150,180]):
    on=f['pump_on_angle_%d'%a][:]
    axx=ax.flat[i+1]; m=axx.pcolormesh(k,E,on-off,shading='auto',cmap='RdBu_r'); axx.set_title(f'diff {a}'); plt.colorbar(m,ax=axx)
plt.savefig('outputs/quicklook.png',dpi=70)
print(f['time_delays'][:])
# profile
i=np.argmin(abs(E-0)); print(np.round(off[i,::5],1))
j=np.argmin(abs(k-0)); print(np.round(off[::5,j],1))
print(np.round((f['pump_on_angle_0'][:]-off)[::5,j],2))
print(np.round((f['pump_on_angle_0'][:]-off)[np.argmin(abs(E-0.205)),::3],2))
