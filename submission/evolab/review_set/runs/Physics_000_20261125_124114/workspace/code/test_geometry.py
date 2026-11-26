import numpy as np, json
from icoshell import *
from scipy.optimize import minimize
res = {}
for h in range(0,6):
    for k in range(0,6):
        if (h,k)==(0,0): continue
        X = shell(h,k)
        r = np.linalg.norm(X,axis=1)
        assert len(X)==10*T_of(h,k)+2
        assert abs(r.max()-S5*np.sqrt(T_of(h,k)))<1e-9
        # min NN distance ~1 for Mackay (flat faces)
        d = np.linalg.norm(X[:,None]-X[None],axis=-1); np.fill_diagonal(d,9)
        res[str((h,k))] = dict(T=T_of(h,k), N=len(X), Rv=float(r.max()), dmin=float(d.min()), cls=shell_class(h,k))
# (h,0)==(0,h)
for h in range(1,5):
    A=shell(h,0); B=shell(0,h)
    dd=np.linalg.norm(A[:,None]-B[None],axis=-1).min(1).max(); assert dd<1e-9
# mirror
A=shell(1,2); B=shell(2,1)
print('(1,2) vs (2,1) max mismatch of same set:', np.linalg.norm(A[:,None]-B[None],axis=-1).min(1).max())
Bm=B*np.array([-1,1,1])
# mirror then best rotation? test congruence via sorted distance-from-centre & pair distance spectra
def spec(X): d=np.linalg.norm(X[:,None]-X[None],axis=-1); return np.sort(d.ravel())
print('spectra equal (mirror images congruent up to reflection):', np.allclose(spec(A),spec(B)))
# LJ checks
for path,ref in [([(0,0),(0,1)],-44.326801),([(0,0),(0,1),(0,2)],-279.248470),([(0,0),(0,1),(0,2),(0,3)],-876.461207)]:
    d=[1.0]*len(path)
    X,D,I=build_cluster(path,d)
    pot=PairPot(D,'LJ')
    # LJ in reduced units: sigma=1 -> r0 = 2^(1/6); energy depth 1
    Xs = X*2**(1/6)*0.97
    pot=PairPot(np.full(len(X),2**(1/6)),'LJ')
    E,X1=relax(Xs,pot,gtol=1e-8)
    print(len(X),E,ref); res['LJ%d'%len(X)]=dict(E=E,ref=ref)
# central difference grad check
X,D,I=build_cluster([(0,0),(0,1),(1,1)],[1,1.1,1.3]); pot=PairPot(D,'LJ')
x=X.ravel()+0.01*np.random.default_rng(0).normal(size=X.size)
E,G=pot.eg(x); errs=[]
for idx in [0,5,17,44,100]:
    e=np.zeros_like(x); e[idx]=1e-6
    num=(pot.eg(x+e)[0]-pot.eg(x-e)[0])/2e-6
    errs.append(abs(num-G[idx])/max(abs(G[idx]),1e-8))
for kind in ['Morse','harm']:
    pot2=PairPot(D,kind,rho=10.) if kind=='Morse' else PairPot(D,'harm',pairs=contact_pairs(X,D))
    E,G=pot2.eg(x)
    for idx in [0,5,17,44,100]:
        e=np.zeros_like(x); e[idx]=1e-6
        num=(pot2.eg(x+e)[0]-pot2.eg(x-e)[0])/2e-6
        errs.append(abs(num-G[idx])/max(abs(G[idx]),1e-8))
print('grad rel err', errs); res['grad_rel_err_max']=max(errs)
json.dump(res,open('../outputs/geometry_checks.json','w'),indent=1)
