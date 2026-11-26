import sys,time,json; sys.path.insert(0,'/workspace/code')
from moire import *
N=int(sys.argv[1]); mix=float(sys.argv[2]); Vzt=float(sys.argv[3]); eps=float(sys.argv[4]); nu=float(sys.argv[5])
vf=tuple(float(x) for x in sys.argv[6].split(',')) if sys.argv[6]!='free' else None
m=Model(Vzt=Vzt); hf=HF(m,N,eps)
z={1:np.zeros_like(hf.h0[1]),-1:np.zeros_like(hf.h0[1])}
E,U=hf.diag(z); rho,occ=hf.density(E,U,nu,vf)
trs=len(sys.argv)>8 and sys.argv[8]=='trs'
r=hf.run(nu,rho,maxit=int(sys.argv[7]),valley_fix=vf,mix=mix,tol=1e-6,trs=trs)
for i,(e,d) in enumerate(r['hist']):
    if i%10==0: print(i,e,d)
E=r['E']
print('T-asym of bands', max(np.abs(np.sort(E[1][k])-np.sort(E[-1][hf.Tm[0][k]] if hasattr(hf,'Tm') else E[-1][k])).max() for k in range(hf.Nk)))
print('gap',hf.gap(E,r['occ']),'niter',r['niter'],'Etot',r['Etot'])
for t in (1,-1): print(t,'occ',r['occ'][t].sum(),'band occ', r['occ'][t].sum(0)[:3], 'b1',E[t][:,0].min(),E[t][:,0].max(),'b2',E[t][:,1].min(),E[t][:,1].max())
