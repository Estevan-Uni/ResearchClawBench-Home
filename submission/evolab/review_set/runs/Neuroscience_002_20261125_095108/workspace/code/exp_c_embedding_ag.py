"""Experiment C (autograd backend): learned pair embedding trained with
lambda_seg * discriminative clustering loss (De Brabandere et al.) + lambda_pair(t) * contrastive loss (Hadsell et al.).
Ablation of the lambda_pair schedule. Proxy ranking metric: mean rank of the true merge candidate among K
same-degradation negatives, ranked by embedding distance to the degradation-specific merge prototype."""
import sys; sys.path.insert(0,'/workspace/code')
from common import *
from ag_nn import *
import autograd.numpy as anp
from autograd import grad
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_recall_curve, average_precision_score
trn,val,te=load()
mu=trn[FEAT].values.mean(0); sd=trn[FEAT].values.std(0)
def T(df): return np.c_[(df[FEAT].values-mu)/sd,deg_onehot(df)]
Xtr,Xv,Xte=T(trn),T(val),T(te); ytr=trn.label.values.astype(int); dtr=trn.degradation.map(DEGS.index).values
DV,DD,MARGIN=0.5,1.5,2.0
def disc_loss(e,G):
    """G: (n, 8) one-hot of cluster id = 2*deg+label. pull (variance), push (between labels within deg), reg."""
    cnt=G.sum(0)+1e-9; mus=anp.dot(G.T,e)/cnt[:,None]          # (8,16) cluster means
    dist=anp.sqrt(anp.sum((e-anp.dot(G,mus))**2,1)+1e-9)
    Lv=anp.sum(anp.dot(G.T,anp.maximum(dist-DV,0)**2)/cnt)/8
    Ld=0.
    for k in range(4):
        dd=anp.sqrt(anp.sum((mus[2*k]-mus[2*k+1])**2)+1e-9); Ld=Ld+anp.maximum(2*DD-dd,0)**2
    Lr=anp.mean(anp.sqrt(anp.sum(mus**2,1)+1e-9))
    return Lv+Ld/4+1e-3*Lr
def pair_loss(e,i,j,s):
    D=anp.sqrt(anp.sum((e[i]-e[j])**2,1)+1e-9)
    return anp.mean(s*D**2+(1-s)*anp.maximum(MARGIN-D,0)**2)/2
def total(params,X,G,i,j,s,ws,wp):
    e=mlp(params,X); L=0.
    if ws>0: L=L+ws*disc_loss(e,G)
    if wp>0: L=L+wp*pair_loss(e,i,j,s)
    return L
g_total=grad(total)
SCHED={'cluster-only':lambda t:(1.,0.),'fixed-high':lambda t:(1.,1.),'fixed-low':lambda t:(1.,.1),
       'anneal-high-to-low':lambda t:(1.,1.-.9*t),'anneal-low-to-high':lambda t:(1.,.1+.9*t),'pair-only':lambda t:(0.,1.)}
EPOCHS=int(sys.argv[1]) if len(sys.argv)>1 else 8; BS=1024
def train(sched,seed):
    rng=np.random.default_rng(seed); params=init_mlp([24,64,64,16],rng); opt=Adam(params)
    pos=np.where(ytr==1)[0]; neg=np.where(ytr==0)[0]; steps=len(ytr)//BS; tot=EPOCHS*steps; it=0; hist=[]
    for ep in range(EPOCHS):
        for _ in range(steps):
            b=np.r_[rng.choice(pos,BS//4),rng.choice(neg,3*BS//4)]
            X=Xtr[b]; y=ytr[b]; d=dtr[b]; G=np.eye(8)[2*d+y]
            i=rng.integers(0,BS,2048); j=rng.integers(0,BS,2048); keep=d[i]==d[j]; i,j=i[keep],j[keep]; s=(y[i]==y[j]).astype(float)
            ws,wp=SCHED[sched](it/tot)
            params=opt.step(params,g_total(params,X,G,i,j,s,ws,wp)); it+=1
        hist.append(float(total(params,X,G,i,j,s,ws,wp)))
    return params,hist
def evaluate(params,K_list=(9,19),n_queries=3000,seed=123):
    Etr,Ev,Et=mlp(params,Xtr),mlp(params,Xv),mlp(params,Xte)
    P=np.stack([Etr[(dtr==k)&(ytr==1)].mean(0) for k in range(4)])
    def score(E,df): dd=df.degradation.map(DEGS.index).values; return -np.linalg.norm(E-P[dd],axis=1)
    st=score(Et,te); yt=te.label.values; dte=te.degradation.map(DEGS.index).values; rng=np.random.default_rng(seed); res={}
    for K in K_list:
        ranks=[]; rr=[]
        for q in rng.choice(np.where(yt==1)[0],n_queries):
            negs=rng.choice(np.where((yt==0)&(dte==dte[q]))[0],K,replace=False)
            ranks.append(1+(st[negs]>st[q]).sum()+0.5*(st[negs]==st[q]).sum()); rs=rng.random(K+1); rr.append(1+(rs[1:]>rs[0]).sum())
        ranks=np.array(ranks); res[f'mean_rank_K{K}']=ranks.mean(); res[f'top1_K{K}']=(ranks==1).mean(); res[f'random_mean_rank_K{K}']=np.mean(rr)
    for k,dg in enumerate(DEGS):   # per-degradation mean rank (K=19)
        rr=[]
        for q in rng.choice(np.where((yt==1)&(dte==k))[0],1000):
            negs=rng.choice(np.where((yt==0)&(dte==k))[0],19,replace=False); rr.append(1+(st[negs]>st[q]).sum())
        res[f'mean_rank_K19_{dg}']=np.mean(rr)
    res['AP_distance']=average_precision_score(yt,st)
    lr=LogisticRegression(max_iter=2000).fit(np.c_[Etr,deg_onehot(trn)],ytr)
    pv=lr.predict_proba(np.c_[Ev,deg_onehot(val)])[:,1]; pt=lr.predict_proba(np.c_[Et,deg_onehot(te)])[:,1]
    p,r,t=precision_recall_curve(val.label.values,pv); fs=2*p*r/np.maximum(p+r,1e-12); th=t[np.nanargmax(fs[:-1])]
    res['merge_F1']=f1_score(yt,pt>=th); res['merge_AP']=average_precision_score(yt,pt)
    for k,dg in enumerate(DEGS): res[f'merge_F1_{dg}']=f1_score(yt[dte==k],pt[dte==k]>=th)
    return res,Et
if __name__=='__main__':
    import time; t0=time.time(); rows=[]
    for sched in SCHED:
        for seed in range(3):
            params,h=train(sched,seed); r,Et=evaluate(params); r.update(schedule=sched,seed=seed,final_loss=h[-1]); rows.append(r)
            print(f'{time.time()-t0:.0f}s',sched,seed,{k:round(v,4) for k,v in r.items() if isinstance(v,float)},flush=True)
            if sched=='anneal-high-to-low' and seed==0: np.save('/workspace/outputs/learned_embedding_test.npy',Et)
            pd.DataFrame(rows).to_csv('/workspace/outputs/embedding_ablation_raw.csv',index=False)
    R=pd.DataFrame(rows); agg=R.drop(columns='seed').groupby('schedule').agg(['mean','std']); agg.to_csv('/workspace/outputs/embedding_ablation_summary.csv'); print(agg.round(4).T.to_string())
