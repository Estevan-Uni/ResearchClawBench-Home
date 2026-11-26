import sys; sys.path.insert(0,'/workspace/code')
from common import *
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
R=pd.read_csv('/workspace/outputs/embedding_ablation_raw.csv')
order=['pair-only','cluster-only','fixed-low','fixed-high','anneal-high-to-low','anneal-low-to-high']
fig,ax=plt.subplots(1,4,figsize=(20,4.6))
mets=[('mean_rank_K19','Mean rank of true candidate (K=19 negatives)\nlower is better; random = 10.5'),
      ('top1_K19','Top-1 retrieval (K=19)'),('merge_F1','Downstream merge F1 (LR on embedding)')]
for a,(m,t) in zip(ax[:3],mets):
    g=R.groupby('schedule')[m]; mu=g.mean().reindex(order); sd=g.std().reindex(order)
    a.bar(range(6),mu,yerr=sd,capsize=4,color=plt.cm.Set2.colors[:6])
    for s in range(3): a.scatter(range(6),R[R.seed==s].set_index('schedule')[m].reindex(order),c='k',s=10,zorder=3)
    a.set_xticks(range(6)); a.set_xticklabels(order,rotation=30,ha='right'); a.set_title(t,fontsize=10)
    lo=(mu-sd).min(); hi=(mu+sd).max(); a.set_ylim(lo-0.3*(hi-lo),hi+0.2*(hi-lo))
cols=[c for c in R.columns if c.startswith('mean_rank_K19_')]
if cols:
    M=R.groupby('schedule')[cols].mean().reindex(order); M.columns=[c.replace('mean_rank_K19_','') for c in cols]
    for k,s in enumerate(order): ax[3].plot(range(4),M.loc[s],'o-',label=s,color=plt.cm.Set2.colors[k])
    ax[3].set_xticks(range(4)); ax[3].set_xticklabels(M.columns,rotation=20); ax[3].set_title('Mean rank (K=19) per degradation',fontsize=10); ax[3].legend(fontsize=7)
plt.tight_layout(); plt.savefig('/workspace/report/images/fig5_embedding_loss_schedule_ablation.png',dpi=120); plt.close()
# schedule illustration inset as separate small fig
t=np.linspace(0,1,100)
fig,a=plt.subplots(figsize=(5,3))
for k,(s,f) in enumerate({'cluster-only':0*t,'fixed-low':0*t+.1,'fixed-high':0*t+1,'anneal-high-to-low':1-.9*t,'anneal-low-to-high':.1+.9*t}.items()): a.plot(t,f,label=s,color=plt.cm.Set2.colors[k+1])
a.set_xlabel('training progress t'); a.set_ylabel('λ_pair(t)  (λ_seg = 1)'); a.legend(fontsize=7); a.set_title('Loss-weight schedules (pair-only: λ_seg=0, λ_pair=1)',fontsize=9)
plt.tight_layout(); plt.savefig('/workspace/report/images/fig5a_schedules.png',dpi=120); plt.close()
# SE gate figure
G=pd.read_csv('/workspace/outputs/se_gate_means.csv')
fig,ax=plt.subplots(1,2,figsize=(12,4))
for a,mdl in zip(ax,['SE-gated-deg','SE-gated-nodeg']):
    S=pd.read_csv('/workspace/outputs/dataset_summary.csv'); pr=S[S.split=='test'].set_index('degradation').pos_rate
    q=G[G.model==mdl].copy(); q['w']=np.where(q.label==1,q.degradation.map(pr),1-q.degradation.map(pr))   # prevalence-weighted pooled mean
    g=q.assign(**{c:q[c]*q.w for c in ['gate_M','gate_I','gate_E']}).groupby('degradation')[['gate_M','gate_I','gate_E']].sum().reindex(DEGS)
    x=np.arange(4)
    for k,(c,lab) in enumerate(zip(g.columns,['Morphology','Intensity','Embedding'])): a.bar(x+(k-1)*.27,g[c],.27,label=lab)
    a.set_xticks(x); a.set_xticklabels(DEGS); a.set_ylim(0,1.05); a.set_ylabel('mean SE gate value (test)'); a.set_title(mdl+(' (gate input: descriptors + degradation)' if mdl.endswith('deg') and 'no' not in mdl else ' (gate input: descriptors only)'),fontsize=10)
ax[0].legend()
plt.tight_layout(); plt.savefig('/workspace/report/images/fig8_se_gates.png',dpi=120); plt.close()
