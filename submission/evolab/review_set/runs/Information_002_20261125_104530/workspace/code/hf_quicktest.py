import sys,time; sys.path.insert(0,'/workspace/code')
from moire import *
m=Model(Vzt=-20.0)
hf=HF(m,6,15.0)
z={1:np.zeros_like(hf.h0[1]),-1:np.zeros_like(hf.h0[1])}
E,U=hf.diag(z)
rho,occ=hf.density(E,U,2.0,(1,1))
print('bare gap TRS',hf.gap(E,occ))
print('bare band1 range', E[1][:,0].min(),E[1][:,0].max(),' band2',E[1][:,1].min(),E[1][:,1].max())
t=time.time()
r=hf.run(2.0,rho,maxit=150,valley_fix=(1,1),mix=0.3)
for i,(e,d) in enumerate(r['hist']):
    if i%5==0: print(i,e,d)
print('gap',hf.gap(r['E'],r['occ']),'niter',r['niter'],time.time()-t)
E=r['E']; print('band1',E[1][:,0].min(),E[1][:,0].max(),' band2',E[1][:,1].min(),E[1][:,1].max())
