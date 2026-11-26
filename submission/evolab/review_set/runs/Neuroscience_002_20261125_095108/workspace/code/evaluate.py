"""Compute P/R/F1 (val-optimal global threshold), AP, AUC per degradation; bootstrap CIs; fusion table; PR curves."""
import sys; sys.path.insert(0,'/workspace/code')
from common import *
from sklearn.metrics import precision_recall_curve, average_precision_score, roc_auc_score, f1_score, precision_score, recall_score
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
trn,val,te=load(); yv=val.label.values; yt=te.label.values
Z=dict(np.load('/workspace/outputs/scores_A.npz'))
try: Z.update(dict(np.load('/workspace/outputs/scores_gate.npz')))
except FileNotFoundError: pass
names=sorted({k.split('|')[0] for k in Z})
def best_thr(y,s):
    p,r,t=precision_recall_curve(y,s); f=2*p*r/np.maximum(p+r,1e-12); i=np.nanargmax(f[:-1]); return t[i]
rows=[]; thr={}
for n in names:
    sv,st=Z[n+'|val'],Z[n+'|test']; th=best_thr(yv,sv); thr[n]=th
    for d in DEGS+['All']:
        idx=np.ones(len(te),bool) if d=='All' else (te.degradation==d).values
        y,s=yt[idx],st[idx]; yh=s>=th
        rows.append(dict(model=n,scheme=n.split('/')[0],subset=n.split('/')[1],degradation=d,
            P=precision_score(y,yh,zero_division=0),R=recall_score(y,yh),F1=f1_score(y,yh),AP=average_precision_score(y,s),AUC=roc_auc_score(y,s)))
R=pd.DataFrame(rows); R.to_csv('/workspace/outputs/metrics_long.csv',index=False)
W=R.pivot_table(index=['scheme','subset'],columns='degradation',values=['P','R','F1'])
W=W.reindex(columns=pd.MultiIndex.from_product([['P','R','F1'],DEGS+['All']])).swaplevel(0,1,axis=1)
W=W[[ (d,m) for d in DEGS+['All'] for m in ['P','R','F1']]]
order=['M','I','E','MI','ME','IE','MIE']
W=W.reset_index(); W['o']=W.subset.map(order.index); W=W.sort_values(['scheme','o']).drop(columns='o')
W.to_csv('/workspace/outputs/fusion_table.csv',index=False)
print(R[R.degradation=='All'].sort_values('F1',ascending=False).round(4).to_string())
# bootstrap CIs for M+I+E models (stratified by degradation within test)
rng=np.random.default_rng(0); B=200; ci=[]
mie=[n for n in names if n.endswith('/MIE')]
groups=[np.where((te.degradation==d).values)[0] for d in DEGS]
boots=[np.concatenate([rng.choice(g,len(g)) for g in groups]) for _ in range(B)]
for n in mie:
    st=Z[n+'|test']; yh=st>=thr[n]
    for d in DEGS+['All']:
        vals=[]; vap=[]
        for b in boots:
            bb=b if d=='All' else b[(te.degradation.values[b]==d)]
            vals.append(f1_score(yt[bb],yh[bb])); vap.append(average_precision_score(yt[bb],st[bb]))
        ci.append(dict(model=n,degradation=d,F1_lo=np.percentile(vals,2.5),F1_hi=np.percentile(vals,97.5),AP_lo=np.percentile(vap,2.5),AP_hi=np.percentile(vap,97.5)))
CI=pd.DataFrame(ci); CI.to_csv('/workspace/outputs/bootstrap_ci_MIE.csv',index=False); print(CI.round(4).to_string())
# modality gain table (per degradation F1 differences) for HGB-early, QDA, MLP
gain=[]
for sch in ['QDA-blockmean','HGB-early','MLP-early','LR-perdeg']:
    for base,add_,lab in [('MI','MIE','+E to M+I'),('ME','MIE','+I to M+E'),('IE','MIE','+M to I+E')]:
        for d in DEGS+['All']:
            g=lambda s: R[(R.model==f'{sch}/{s}')&(R.degradation==d)].F1.values[0]
            gain.append(dict(scheme=sch,added=lab,degradation=d,F1_base=g(base),F1_full=g(add_),gain=g(add_)-g(base)))
G=pd.DataFrame(gain); G.to_csv('/workspace/outputs/modality_gain.csv',index=False)
# PR curves: see fig_pr.py
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
# F1 heatmap subset x degradation for each scheme
schemes=['QDA-blockmean','LR-perdeg','HGB-early','MLP-early']
fig,ax=plt.subplots(1,4,figsize=(20,4.6))
import seaborn as sns
for a,sch in zip(ax,schemes):
    M=R[(R.scheme==sch)].pivot(index='subset',columns='degradation',values='F1').reindex(index=order,columns=DEGS+['All'])
    sns.heatmap(M,annot=True,fmt='.3f',vmin=0,vmax=1,cmap='magma',ax=a,cbar=False); a.set_title(sch)
plt.tight_layout(); plt.savefig('/workspace/report/images/fig4_modality_subset_f1.png',dpi=130)
print(G[G.scheme=='HGB-early'].round(4).to_string())
