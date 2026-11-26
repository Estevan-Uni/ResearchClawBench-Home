"""Experiment D: PCA visualisation of the embedding block (raw feats 10-19) and the learned embedding (exp C)."""
import sys, os; sys.path.insert(0,'/workspace/code')
from common import *
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import silhouette_score
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
trn,val,te=load(); rng=np.random.default_rng(0)
spaces={'raw embedding block (features 10–19)':(raw(trn,'E'),raw(te,'E'))}
if os.path.exists('/workspace/outputs/learned_embedding_test.npy'):
    spaces['learned 16-d embedding (exp. C, anneal schedule)']=(None,np.load('/workspace/outputs/learned_embedding_test.npy'))
rows=[]
fig,ax=plt.subplots(len(spaces),4,figsize=(18,4.4*len(spaces)),squeeze=False)
for r,(nm,(Xa,Xt)) in enumerate(spaces.items()):
    pca=PCA(3).fit(Xa if Xa is not None else Xt); Zt=pca.transform(Xt)
    for c,d in enumerate(DEGS):
        idx=(te.degradation==d).values; Z=Zt[idx]; y=te.label.values[idx]
        sub=rng.choice(len(Z),min(len(Z),4000),replace=False)
        a=ax[r,c]; a.scatter(Z[sub][y[sub]==0,0],Z[sub][y[sub]==0,1],s=3,c='#9e9e9e',alpha=.35,label='split (neg)')
        a.scatter(Z[sub][y[sub]==1,0],Z[sub][y[sub]==1,1],s=4,c='#d62728',alpha=.6,label='merge (pos)')
        auc=cross_val_score(LogisticRegression(max_iter=1000),Xt[idx],y,cv=5,scoring='roc_auc').mean()
        auc2=cross_val_score(LogisticRegression(max_iter=1000),Z[:,:2],y,cv=5,scoring='roc_auc').mean()
        ss=sub[:3000]; sil=silhouette_score(Xt[idx][ss],y[ss])
        rows.append(dict(space=nm,degradation=d,linear_probe_AUC_full=auc,linear_probe_AUC_PC12=auc2,silhouette=sil,expl_var_PC12=pca.explained_variance_ratio_[:2].sum()))
        a.set_title(f'{d}\nprobe AUC={auc:.3f}, silhouette={sil:.3f}',fontsize=10); a.set_xlabel('PC1'); a.set_ylabel('PC2')
        if c==0: a.legend(markerscale=4,fontsize=8,title=nm,title_fontsize=8)
plt.tight_layout(); plt.savefig('/workspace/report/images/fig6_embedding_pca.png',dpi=120)
pd.DataFrame(rows).to_csv('/workspace/outputs/embedding_separability.csv',index=False); print(pd.DataFrame(rows).round(3).to_string())
# RGB strip: PC1-3 -> RGB for 40 example pairs per class per degradation (learned or raw)
nm=list(spaces)[-1]; Xa,Xt=spaces[nm]; Zt=PCA(3).fit_transform(Xt); lo,hi=np.percentile(Zt,2,0),np.percentile(Zt,98,0)
C=np.clip((Zt-lo)/(hi-lo),0,1)
fig,ax=plt.subplots(4,1,figsize=(12,4.2))
for i,d in enumerate(DEGS):
    idx=np.where((te.degradation==d).values)[0]; y=te.label.values[idx]
    p=rng.choice(idx[y==1],40,replace=False); n=rng.choice(idx[y==0],40,replace=False)
    ax[i].imshow(np.r_[C[p][None],C[n][None]],aspect='auto'); ax[i].set_yticks([0,1]); ax[i].set_yticklabels(['merge','split']); ax[i].set_xticks([]); ax[i].set_ylabel(d,rotation=0,ha='right',va='center')
ax[0].set_title(f'PC1–3 of {nm} mapped to RGB, 40 random test pairs per class')
plt.tight_layout(); plt.savefig('/workspace/report/images/fig7_embedding_rgb.png',dpi=120)
