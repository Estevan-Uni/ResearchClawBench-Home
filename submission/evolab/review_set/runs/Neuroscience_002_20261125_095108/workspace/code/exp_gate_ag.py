"""Experiment A(c) (autograd backend): SE-style degradation-gated fusion network.
Per-modality encoders -> modality feature channels; SE gate from [squeezed per-modality descriptors, degradation one-hot]
-> bottleneck(8) -> sigmoid gates (3) that re-weight modality channels -> head. Variants: gate w/ degradation, gate w/o, no gate."""
import sys, time; sys.path.insert(0,'/workspace/code')
from common import *
from ag_nn import *
import autograd.numpy as anp
from autograd import grad
trn,val,te=load()
mu=trn[FEAT].values.mean(0); sd=trn[FEAT].values.std(0)
def T(df): return (df[FEAT].values-mu)/sd, deg_onehot(df), df.label.values.astype(float)
H=16
def init(rng,cond):
    p=[]
    for m in 'MIE': p+=init_mlp([len(BLOCKS[m]),H,H],rng)
    p+=init_mlp([3+(4 if cond else 0),8,3],rng); p+=init_mlp([3*H+4,32,1],rng); return p
def forward(p,X,O,mode):
    F=[]
    for k,m in enumerate('MIE'):
        h=anp.maximum(mlp(p[2*k:2*k+2],X[:,BLOCKS[m]]),0); F.append(h)
    if mode=='nogate': g=anp.ones((X.shape[0],3))
    else:
        sq=anp.stack([f.mean(1) for f in F],1)
        z=anp.concatenate([sq,O],1) if mode=='deg' else sq
        g=1/(1+anp.exp(-mlp(p[6:8],z)))
    Z=anp.concatenate([F[k]*g[:,k:k+1] for k in range(3)]+[O],1)
    return mlp(p[8:10],Z)[:,0],g
def bce(p,X,O,y,mode):
    s,_=forward(p,X,O,mode); return anp.mean(anp.logaddexp(0,s)-y*s)
gb=grad(bce)
X,O,Y=T(trn); Xv,Ov,Yv=T(val); Xt,Ot,Yt=T(te)
out={}; gates=[]; t0=time.time()
for name,mode in [('SE-gated-deg','deg'),('SE-gated-nodeg','sq'),('Ungated-enc','nogate')]:
    rng=np.random.default_rng(0); p=init(rng,mode=='deg'); opt=Adam(p,2e-3); best=(1e9,None)
    for ep in range(10):
        perm=rng.permutation(len(X))
        for i in range(0,len(X),512):
            b=perm[i:i+512]; p=opt.step(p,gb(p,X[b],O[b],Y[b],mode))
        lv=bce(p,Xv,Ov,Yv,mode)
        if lv<best[0]: best=(lv,p)
        print(name,ep,round(float(lv),5),f'{time.time()-t0:.0f}s',flush=True)
    p=best[1]
    for split,(A,B,df) in [('val',(Xv,Ov,val)),('test',(Xt,Ot,te))]:
        s,g=forward(p,A,B,mode); out[f'{name}/MIE|{split}']=1/(1+np.exp(-s))
        if split=='test' and mode!='nogate':
            G=pd.DataFrame(g,columns=['gate_M','gate_I','gate_E']); G['degradation']=df.degradation.values; G['label']=df.label.values; G['model']=name; gates.append(G)
np.savez('/workspace/outputs/scores_gate.npz',**out)
G=pd.concat(gates); G.groupby(['model','degradation','label']).mean().to_csv('/workspace/outputs/se_gate_means.csv'); print(G.groupby(['model','degradation']).mean())
