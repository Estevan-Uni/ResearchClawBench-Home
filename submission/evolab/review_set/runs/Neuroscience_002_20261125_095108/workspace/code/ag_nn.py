"""Tiny autograd-based MLP + Adam (fallback when torch is unavailable)."""
import autograd.numpy as anp
from autograd import grad
import numpy as np
def init_mlp(sizes,rng):
    return [(rng.normal(0,np.sqrt(2/a),(a,b)),np.zeros(b)) for a,b in zip(sizes[:-1],sizes[1:])]
def mlp(params,X):
    h=X
    for i,(W,b) in enumerate(params):
        h=anp.dot(h,W)+b
        if i<len(params)-1: h=anp.maximum(h,0)
    return h
class Adam:
    def __init__(s,params,lr=1e-3,b1=.9,b2=.999,eps=1e-8):
        s.lr,s.b1,s.b2,s.eps=lr,b1,b2,eps; s.t=0
        s.m=[(np.zeros_like(W),np.zeros_like(b)) for W,b in params]; s.v=[(np.zeros_like(W),np.zeros_like(b)) for W,b in params]
    def step(s,params,grads):
        s.t+=1; out=[]
        for i,((W,b),(gW,gb)) in enumerate(zip(params,grads)):
            new=[]
            for j,(p,g) in enumerate([(W,gW),(b,gb)]):
                m=s.m[i][j]; v=s.v[i][j]; m*=s.b1; m+=(1-s.b1)*g; v*=s.b2; v+=(1-s.b2)*g*g
                mh=m/(1-s.b1**s.t); vh=v/(1-s.b2**s.t); new.append(p-s.lr*mh/(np.sqrt(vh)+s.eps))
            out.append(tuple(new))
        return out
