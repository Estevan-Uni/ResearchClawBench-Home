import sys; sys.path.insert(0,'/workspace/code')
from moire import *
m=Model(Vzt=-20.0); hf=HF(m,6,15.0)
Tm=build_T_map(m,hf)
hT=apply_T(hf.h0[1],Tm)   # should equal h0[-1] where valid (h~ = -h^T, T commutes)
v=Tm[2]
err=max(np.abs((hT[k]-hf.h0[-1][k])[np.outer(v[k],v[k])]).max() for k in range(hf.Nk))
print('T map check on h~:',err, 'valid fraction', v.mean())
