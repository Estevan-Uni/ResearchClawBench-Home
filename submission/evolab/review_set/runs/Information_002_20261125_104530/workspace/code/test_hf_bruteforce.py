"""Brute-force check of the HF self-energy implementation: <H_int> evaluated directly from the
unreduced 4-momentum sum (step 10/13 forms) vs 1/2 Tr(Sigma rho) from the reduced (step 14-16) code."""
import sys; sys.path.insert(0,'/workspace/code')
from moire import *
rng=np.random.default_rng(1)
m=Model(Vzt=-20.0,nshell=1)
N=3; hf=HF(m,N,15.0)
nG=m.nG; dim=2*nG
# random valley-diagonal, k-diagonal density matrices (Hermitian PSD)
rho={}
for t in (1,-1):
    X=rng.normal(size=(hf.Nk,dim,2))+1j*rng.normal(size=(hf.Nk,dim,2))
    Q,_=np.linalg.qr(X)
    rho[t]=np.einsum('kin,kjn->kij',Q.conj(),Q)
S=hf.sigma(rho)
Eint_code=0.5*sum(np.einsum('kij,kij->',S[t],rho[t]) for t in (1,-1))
# brute force: list all single-particle labels (tau, k index, layer, G)
labels=[]
for t in (1,-1):
    for ik in range(hf.Nk):
        for i in range(dim):
            labels.append((t,ik,i))
L=len(labels)
P=np.array([hf.k[ik]+m.Gc[i%nG] for (t,ik,i) in labels])
lay=np.array([i//nG for (t,ik,i) in labels]); tau=np.array([t for (t,ik,i) in labels])
R=np.zeros((L,L),complex)
for a,(t,ik,i) in enumerate(labels):
    for b,(t2,ik2,j) in enumerate(labels):
        if t==t2 and ik==ik2: R[a,b]=rho[t][ik,i,j]
A=hf.A
# Hartree: sum V(|p1-p4|) rho[1,4] rho[2,3] delta(p1+p2=p3+p4), flavor(1)=flavor(4), flavor(2)=flavor(3)
fl=list(zip(tau,lay))
same=np.array([[fl[a]==fl[b] for b in range(L)] for a in range(L)])
EH=0; EF=0
# Hartree with p1-p4 = q != 0
d14=P[:,None,:]-P[None,:,:]
for a in range(L):
    for d in range(L):
        if not same[a,d] or R[a,d]==0: continue
        q=P[a]-P[d]
        if np.linalg.norm(q)<1e-9: continue
        # p2 - p3 = p4 - p1 = -q
        mask=(np.linalg.norm(d14+q,axis=-1)<1e-9)&same
        EH+=Vq(np.array([np.linalg.norm(q)]),15.0)[0]*R[a,d]*np.sum(R[mask])
# Fock: - V(|p1-p4|) rho[1,3] rho[2,4], flavor1=flavor4, flavor2=flavor3
for a in range(L):
    for c in range(L):
        if R[a,c]==0: continue
        for b in range(L):
            if not same[b,c]: continue
            # p4 = p1+p2-p3
            p4=P[a]+P[b]-P[c]
            dd=np.where((np.linalg.norm(P-p4,axis=1)<1e-9))[0]
            for d in dd:
                if same[a,d]:
                    EF-=Vq(np.array([np.linalg.norm(P[a]-P[d])]),15.0)[0]*R[a,c]*R[b,d]
Eint_bf=(EH+EF)/(2*A)
print('code',Eint_code,' brute',Eint_bf,' diff',abs(Eint_code-Eint_bf))
