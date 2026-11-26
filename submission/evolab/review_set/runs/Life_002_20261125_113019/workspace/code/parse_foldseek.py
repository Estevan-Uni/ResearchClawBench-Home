"""Parse Foldseek-Multimer output -> per-target best assignment, chain table, U/t consistency."""
import numpy as np, json, sys, os
sys.path.insert(0, os.path.dirname(__file__)); import pdbutil as P
FMT=open('outputs/pair/fmt.txt').read().strip().split(',')
def split_name(n):
    return (n.split('_',1)[0], n.split('_',1)[1]) if '_' in n else (n, None)
def load_m8(fn):
    rows=[]
    for l in open(fn):
        f=l.rstrip('\n').split('\t'); d=dict(zip(FMT,f))
        for k in ['fident','evalue','alntmscore','qtmscore','ttmscore','lddt','rmsd','prob','complexqtmscore','complexttmscore','interfacelddt']:
            d[k]=float(d[k])
        for k in ['alnlen','qstart','qend','qlen','tstart','tend','tlen','bits','complexassignid']: d[k]=int(d[k])
        d['U']=np.array([float(x) for x in d['u'].split(',')]).reshape(3,3)
        d['T']=np.array([float(x) for x in d['t'].split(',')])
        d['cU']=np.array([float(x) for x in d['complexu'].split(',')]).reshape(3,3)
        d['cT']=np.array([float(x) for x in d['complext'].split(',')])
        d['qc'],d['qch']=split_name(d['query']); d['tc'],d['tch']=split_name(d['target'])
        if d['tch'] is None: d['tch']='A'
        if d['qch'] is None: d['qch']='A'
        rows.append(d)
    return rows
def rot_angle(A,B):
    c=(np.trace(A.T@B)-1)/2; return float(np.degrees(np.arccos(np.clip(c,-1,1))))
def best_assignments(rows):
    """per target complex: assignment with highest complex qTM -> list of rows"""
    by={}
    for r in rows: by.setdefault((r['tc'],r['complexassignid']),[]).append(r)
    best={}
    for (tc,aid),rs in by.items():
        s=rs[0]['complexqtmscore']
        if tc not in best or s>best[tc][0]['complexqtmscore']: best[tc]=rs
    return best
