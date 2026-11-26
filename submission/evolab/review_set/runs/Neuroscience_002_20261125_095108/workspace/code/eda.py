import sys; sys.path.insert(0,'/workspace/code')
from common import *
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt, seaborn as sns
tr=pd.read_csv(D+'train_simulated.csv'); te=pd.read_csv(D+'test_simulated.csv')
rows=[]
for name,df in [('train',tr),('test',te)]:
    print(name,df.shape,'nan',df.isna().sum().sum(),'dup',df.duplicated().sum())
    for d in DEGS:
        s=df[df.degradation==d]; rows.append(dict(split=name,degradation=d,n=len(s),pos=int(s.label.sum()),pos_rate=s.label.mean()))
    rows.append(dict(split=name,degradation='All',n=len(df),pos=int(df.label.sum()),pos_rate=df.label.mean()))
summ=pd.DataFrame(rows); print(summ); summ.to_csv('/workspace/outputs/dataset_summary.csv',index=False)
# Cohen d per feature per degradation
cd=[]
for d in DEGS:
    s=tr[tr.degradation==d]; p=s[s.label==1][FEAT]; n=s[s.label==0][FEAT]
    sp=np.sqrt((p.var()+n.var())/2); cd.append(((p.mean()-n.mean())/sp).rename(d))
cd=pd.DataFrame(cd); cd.to_csv('/workspace/outputs/cohens_d.csv'); print(cd.round(2))
fig,ax=plt.subplots(1,3,figsize=(18,5.2))
for a,(lab,t) in zip(ax[:2],[(0,'negatives (label 0)'),(1,'positives (label 1)')]):
    C=tr[tr.label==lab][FEAT].corr()
    sns.heatmap(C,ax=a,cmap='RdBu_r',vmin=-0.3,vmax=0.3,square=True,cbar_kws={'shrink':.7})
    for b in [5,10]: a.axhline(b,c='k',lw=1); a.axvline(b,c='k',lw=1)
    a.set_title(f'Feature correlation, {t}')
sns.heatmap(cd,ax=ax[2],cmap='viridis',annot=False,cbar_kws={'label':"Cohen's d (pos vs neg)"})
for b in [5,10]: ax[2].axvline(b,c='w',lw=1.5)
ax[2].set_title("Effect size per feature × degradation\n(M: 0–4 | I: 5–9 | E: 10–19)")
plt.tight_layout(); plt.savefig('/workspace/report/images/fig1_feature_structure.png',dpi=130)
# block-mean distributions
fig,ax=plt.subplots(3,4,figsize=(16,9),sharex='row')
for i,m in enumerate('MIE'):
    for j,d in enumerate(DEGS):
        s=tr[tr.degradation==d]; bm=s[[str(k) for k in BLOCKS[m]]].mean(1)
        for lab,c in [(0,'#888'),(1,'#d62728')]:
            ax[i,j].hist(bm[s.label==lab],bins=60,density=True,alpha=.6,color=c,label=['neg','pos'][lab])
        if i==0: ax[i,j].set_title(d)
        if j==0: ax[i,j].set_ylabel({'M':'Morphology','I':'Intensity','E':'Embedding'}[m]+' block mean\ndensity')
ax[0,0].legend(); plt.tight_layout(); plt.savefig('/workspace/report/images/fig2_block_means.png',dpi=120)
# counts figure
fig,ax=plt.subplots(figsize=(7,3.8))
s2=summ[summ.degradation!='All']
x=np.arange(4); w=.35
for k,(sp,c) in enumerate([('train','#1f77b4'),('test','#ff7f0e')]):
    q=s2[s2.split==sp]; ax.bar(x+k*w-w/2,q.n,w,color=c,label=f'{sp} (all)'); ax.bar(x+k*w-w/2,q.pos,w,color='k',alpha=.5)
ax.set_xticks(x); ax.set_xticklabels(DEGS); ax.set_ylabel('# candidate pairs'); ax.legend(); ax.set_title('Pairs per degradation (dark = positive/merge pairs)')
plt.tight_layout(); plt.savefig('/workspace/report/images/fig0_dataset_counts.png',dpi=130)
