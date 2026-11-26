"""Fig 3: per-degradation PR curves, all fusion schemes (M+I+E, solid) + modality subsets (MLP-early, dotted)."""
import sys; sys.path.insert(0,'/workspace/code')
from common import *
from sklearn.metrics import precision_recall_curve, average_precision_score
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
trn,val,te=load(); yt=te.label.values
Z=dict(np.load('/workspace/outputs/scores_A.npz')); Z.update(dict(np.load('/workspace/outputs/scores_gate.npz')))
fus=['MLP-early/MIE','SE-gated-deg/MIE','SE-gated-nodeg/MIE','Ungated-enc/MIE','QDA-blockmean/MIE','Gated-LR/MIE','Late-LR/MIE','HGB-experts/MIE','HGB-early/MIE','LR-perdeg/MIE']
sub=['MLP-early/'+s for s in ['M','I','E','MI','ME','IE']]
c20=plt.cm.tab20.colors
for zoom in [False,True]:
    fig,ax=plt.subplots(2,2,figsize=(14,11))
    for a,d in zip(ax.flat,DEGS):
        idx=(te.degradation==d).values
        for k,n in enumerate(fus+sub):
            p,r,_=precision_recall_curve(yt[idx],Z[n+'|test'][idx]); ap=average_precision_score(yt[idx],Z[n+'|test'][idx])
            a.plot(r,p,'-' if n in fus else ':',color=c20[(2*k)%20 if k<10 else (2*(k-10)+1)%20],lw=1.8 if n in fus else 2.2,label=f'{n} ({ap:.3f})')
        a.axhline(yt[idx].mean(),color='grey',lw=.8,ls='--')
        a.set_title(f'{d}  (test n={idx.sum()}, merge pairs={int(yt[idx].sum())})'); a.set_xlabel('Recall'); a.set_ylabel('Precision')
        if zoom: a.set_xlim(0.7,1.0); a.set_ylim(0.7,1.005)
        else: a.set_xlim(0,1); a.set_ylim(0,1.02)
        a.legend(fontsize=7,loc='lower left',title='model (AP)',title_fontsize=7,ncol=2 if not zoom else 1)
    plt.suptitle('Precision–recall per degradation — solid: fusion schemes on M+I+E; dotted: MLP early fusion on modality subsets'+(' (zoom)' if zoom else ''))
    plt.tight_layout(); plt.savefig('/workspace/report/images/fig3%s_pr_curves_per_degradation.png'%('b' if zoom else ''),dpi=120); plt.close()
