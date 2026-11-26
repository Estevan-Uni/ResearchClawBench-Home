"""Experiment A: modality-subset x model-family fusion comparison.
Saves validation/test scores for every model to outputs/scores_*.npz for later experiments."""
import sys, time; sys.path.insert(0,'/workspace/code')
from common import *
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
trn,val,te=load()
SUBSETS=['M','I','E','MI','ME','IE','MIE']
def perdeg(make, Xf):
    """fit one model per degradation; Xf(df)->features. returns scorer"""
    models={}
    for d in DEGS:
        s=trn[trn.degradation==d]; m=make(); m.fit(Xf(s),s.label.values); models[d]=m
    def score(df):
        out=np.zeros(len(df))
        for d in DEGS:
            idx=(df.degradation==d).values
            p=models[d].predict_proba(Xf(df[idx]))[:,1]; out[idx]=p
        return out
    return score
scores={}
def add(name,fv,ft): scores[name]=(fv,ft); print(f'{name:35s} done {time.time()-t0:.0f}s',flush=True)
t0=time.time()
for S in SUBSETS:
    # (d) interpretable per-degradation QDA on block means
    f=perdeg(lambda: QuadraticDiscriminantAnalysis(reg_param=1e-4), lambda df: block_means(df,S))
    add(f'QDA-blockmean/{S}',f(val),f(te))
    # per-degradation logistic regression on raw features
    f=perdeg(lambda: LogisticRegression(max_iter=2000,C=1.0), lambda df: raw(df,S))
    add(f'LR-perdeg/{S}',f(val),f(te))
    # (a) early fusion: HistGB on concatenated raw features + degradation tag
    Xtr=np.c_[raw(trn,S),deg_onehot(trn)]
    m=HistGradientBoostingClassifier(max_iter=300,learning_rate=0.1,early_stopping=True,random_state=0).fit(Xtr,trn.label)
    add(f'HGB-early/{S}',m.predict_proba(np.c_[raw(val,S),deg_onehot(val)])[:,1],m.predict_proba(np.c_[raw(te,S),deg_onehot(te)])[:,1])
    # (a') early fusion: MLP on raw + tag
    sc=StandardScaler().fit(raw(trn,S))
    m=MLPClassifier((64,32),max_iter=60,early_stopping=True,random_state=0,batch_size=512).fit(np.c_[sc.transform(raw(trn,S)),deg_onehot(trn)],trn.label)
    add(f'MLP-early/{S}',m.predict_proba(np.c_[sc.transform(raw(val,S)),deg_onehot(val)])[:,1],m.predict_proba(np.c_[sc.transform(raw(te,S)),deg_onehot(te)])[:,1])

# (b) late fusion: per-modality encoders (HistGB on block + degradation tag) -> logits -> combiner
def logit(p): p=np.clip(p,1e-6,1-1e-6); return np.log(p/(1-p))
oof=np.zeros((len(trn),3)); Lv=np.zeros((len(val),3)); Lt=np.zeros((len(te),3))
skf=StratifiedKFold(3,shuffle=True,random_state=0)
for j,mo in enumerate('MIE'):
    X=np.c_[raw(trn,mo),deg_onehot(trn)]
    for a,b in skf.split(X,trn.label):
        m=HistGradientBoostingClassifier(max_iter=200,random_state=0).fit(X[a],trn.label.values[a]); oof[b,j]=logit(m.predict_proba(X[b])[:,1])
    m=HistGradientBoostingClassifier(max_iter=200,random_state=0).fit(X,trn.label)
    Lv[:,j]=logit(m.predict_proba(np.c_[raw(val,mo),deg_onehot(val)])[:,1]); Lt[:,j]=logit(m.predict_proba(np.c_[raw(te,mo),deg_onehot(te)])[:,1])
np.savez('/workspace/outputs/modality_logits.npz',oof=oof,val=Lv,test=Lt)
c=LogisticRegression(max_iter=1000).fit(oof,trn.label)
add('Late-LR/MIE',c.predict_proba(Lv)[:,1],c.predict_proba(Lt)[:,1])
print('late-fusion weights',c.coef_)
# (c1) degradation-gated late fusion (linear gate: modality weight depends on degradation one-hot)
def gfeat(L,df): O=deg_onehot(df); return np.c_[L,(L[:,:,None]*O[:,None,:]).reshape(len(L),-1),O]
c=LogisticRegression(max_iter=3000).fit(gfeat(oof,trn),trn.label)
add('Gated-LR/MIE',c.predict_proba(gfeat(Lv,val))[:,1],c.predict_proba(gfeat(Lt,te))[:,1])
# (c2) per-degradation experts (mixture with hard routing on the degradation tag)
f=perdeg(lambda: HistGradientBoostingClassifier(max_iter=300,early_stopping=True,random_state=0), lambda df: raw(df,'MIE'))
add('HGB-experts/MIE',f(val),f(te))
np.savez('/workspace/outputs/scores_A.npz',**{k+'|val':v[0] for k,v in scores.items()},**{k+'|test':v[1] for k,v in scores.items()})
print('ALL DONE',time.time()-t0)
