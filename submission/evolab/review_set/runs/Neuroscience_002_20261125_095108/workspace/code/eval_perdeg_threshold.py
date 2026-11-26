"""Sensitivity: F1 when the decision threshold is tuned on validation separately per degradation (tag known at inference)."""
import sys; sys.path.insert(0,'/workspace/code')
from common import *
from sklearn.metrics import precision_recall_curve, f1_score
trn,val,te=load()
Z=dict(np.load('/workspace/outputs/scores_A.npz')); Z.update(dict(np.load('/workspace/outputs/scores_gate.npz')))
names=sorted({k.split('|')[0] for k in Z}); rows=[]
def bt(y,s):
    p,r,t=precision_recall_curve(y,s); f=2*p*r/np.maximum(p+r,1e-12); return t[np.nanargmax(f[:-1])]
for n in names:
    yh=np.zeros(len(te),bool)
    for d in DEGS:
        iv=(val.degradation==d).values; it=(te.degradation==d).values
        th=bt(val.label.values[iv],Z[n+'|val'][iv]); yh[it]=Z[n+'|test'][it]>=th
        rows.append(dict(model=n,degradation=d,F1_perdeg_thr=f1_score(te.label.values[it],yh[it])))
    rows.append(dict(model=n,degradation='All',F1_perdeg_thr=f1_score(te.label.values,yh)))
R=pd.DataFrame(rows).pivot(index='model',columns='degradation',values='F1_perdeg_thr')[DEGS+['All']]
R.to_csv('/workspace/outputs/f1_perdeg_threshold.csv'); print(R.round(3).to_string())
