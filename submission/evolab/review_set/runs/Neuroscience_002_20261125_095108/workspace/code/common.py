import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
D='/workspace/data/'
DEGS=['Average','Misalignment','Missing Sections','Mixed']
BLOCKS={'M':list(range(0,5)),'I':list(range(5,10)),'E':list(range(10,20))}
FEAT=[str(i) for i in range(20)]
def load():
    tr=pd.read_csv(D+'train_simulated.csv'); te=pd.read_csv(D+'test_simulated.csv')
    strat=tr['degradation']+'_'+tr['label'].astype(int).astype(str)
    trn,val=train_test_split(tr,test_size=0.2,random_state=0,stratify=strat)
    return trn.reset_index(drop=True),val.reset_index(drop=True),te.reset_index(drop=True)
def deg_onehot(df): return np.stack([(df['degradation']==d).values.astype(float) for d in DEGS],1)
def block_means(df,mods='MIE'):
    X=df[FEAT].values
    return np.stack([X[:,BLOCKS[m]].mean(1) for m in mods],1)
def raw(df,mods='MIE'):
    X=df[FEAT].values; idx=sum([BLOCKS[m] for m in mods],[])
    return X[:,idx]
