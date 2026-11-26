"""Chain-correspondence tables + per-chain-pair superposition (U,t) consistency vs complex (U,t)."""
import numpy as np, json, sys, os, itertools
sys.path.insert(0, os.path.dirname(__file__)); import pdbutil as P
from parse_foldseek import load_m8, best_assignments, rot_angle
rows=load_m8(sys.argv[1] if len(sys.argv)>1 else 'outputs/search/exh.m8')
best=best_assignments(rows)
Q=P.ca_chains('outputs/structs/7XG4.pdb')
meta=json.load(open('outputs/db_meta.json'))
def transform(X,U,T):
    # Foldseek u,t map TARGET->QUERY (x_q = U x_t + t); invert to move query onto target
    return (X-T)@U
def nn_med(X,Y):
    d=np.sqrt(((X[:,None,:]-Y[None,:,:])**2).sum(-1)); return float(np.median(d.min(1)))
out={}
# --- convention check: x' = U x + t ?
r=best['9N3Y'][0]; Tc=P.ca_chains('outputs/structs/9N3Y.pdb')
conv={'Ux+t':nn_med(transform(Q[r['qch']][1],r['U'],r['T']),Tc[r['tch']][1]),
      'naive x_t=Ux_q+t':nn_med(Q[r['qch']][1]@r['U'].T+r['T'],Tc[r['tch']][1])}
print('convention check (median NN distance, A):',conv)
out['convention']=conv
chain_tables={}
for tc,rs in sorted(best.items(), key=lambda kv:-kv[1][0]['complexqtmscore']):
    T=P.ca_chains(f'outputs/structs/{tc}.pdb')
    cU,cT=rs[0]['cU'],rs[0]['cT']
    # reference points: all query CA of assigned chains
    tab=[]
    for r in sorted(rs,key=lambda r:r['qch']):
        Xq=Q[r['qch']][1]
        rmsd_disp=float(np.sqrt(((transform(Xq,r['U'],r['T'])-transform(Xq,cU,cT))**2).sum(1).mean()))
        tab.append(dict(qchain=r['qch'],tchain=r['tch'],qlen=r['qlen'],tlen=r['tlen'],alnlen=r['alnlen'],fident=r['fident'],
                        chainTM_q=r['qtmscore'],chainTM_t=r['ttmscore'],alnTM=r['alntmscore'],lddt=r['lddt'],evalue=r['evalue'],
                        U=r['U'].round(3).tolist(),t=r['T'].round(2).tolist(),U_q2t=r['U'].T.round(3).tolist(),t_q2t=(-r['U'].T@r['T']).round(2).tolist(),
                        dAngle_deg=rot_angle(r['U'],cU),dT=float(np.linalg.norm(r['T']-cT)),dispRMS=rmsd_disp,
                        complexNN_med=nn_med(transform(Xq,cU,cT),T[r['tch']][1])))
    fid=[x['fident'] for x in tab]
    chain_tables[tc]=dict(tier=meta.get(tc,{}).get('tier','self'),complexqTM=rs[0]['complexqtmscore'],complexTM_t=rs[0]['complexttmscore'],
        complexU=cU.round(5).tolist(),complext=cT.round(3).tolist(),complexU_q2t=cU.T.round(5).tolist(),complext_q2t=(-cU.T@cT).round(3).tolist(),interfacelddt=rs[0]['interfacelddt'],nchain_pairs=len(tab),
        fident_mean=float(np.mean(fid)),fident_min=float(np.min(fid)),fident_max=float(np.max(fid)),pairs=tab)
out['tables']=chain_tables
# --- wrong-pair control for 9N3Y and 8S35: realign Csf2 chain C onto other Csf2 copies using per-chain alignments of ALL assignments
def load_cp(tc):
    rs=[]
    for l in open(f'outputs/chainpairs/7XG4_vs_{tc}.m8'):
        f=l.split('\t'); qn,tn=f[0],f[1]
        rs.append(dict(qch=qn.split('_')[-1],tch=tn.split('_')[-1],fident=float(f[2]),qtmscore=float(f[5]),
            U=np.array([float(x) for x in f[7].split(',')]).reshape(3,3),T=np.array([float(x) for x in f[8].split(',')])))
    return rs
ctrl={}
for tc in ['9N3Y','8RFJ','8S35']:
    rs=best[tc]; cU,cT=rs[0]['cU'],rs[0]['cT']
    assigned={r['qch']:r['tch'] for r in rs}
    allr=load_cp(tc)
    seen=set(); lst=[]
    for r in allr:
        key=(r['qch'],r['tch'])
        if key in seen: continue
        seen.add(key)
        Xq=Q[r['qch']][1]
        lst.append(dict(qchain=r['qch'],tchain=r['tch'],correct=assigned.get(r['qch'])==r['tch'],chainTM_q=r['qtmscore'],fident=r['fident'],
                        dAngle_deg=rot_angle(r['U'],cU),dispRMS=float(np.sqrt(((transform(Xq,r['U'],r['T'])-transform(Xq,cU,cT))**2).sum(1).mean()))))
    ctrl[tc]=lst
out['all_chain_pairs']=ctrl
json.dump(out,open('outputs/chain_tables.json','w'),indent=1)
for tc in ['9N3Y','8RFJ','8S35','11HC','6N40']:
    d=chain_tables[tc]; print(f"\n== {tc} {d['tier']} cqTM={d['complexqTM']:.3f} ctTM={d['complexTM_t']:.3f} fident mean/min/max={d['fident_mean']:.3f}/{d['fident_min']:.3f}/{d['fident_max']:.3f}")
    for x in d['pairs']: print(x['qchain'],x['tchain'],x['alnlen'],round(x['fident'],3),round(x['chainTM_q'],3),round(x['dAngle_deg'],2),round(x['dispRMS'],2),round(x['complexNN_med'],2))
for tc,l in ctrl.items():
    c=[x for x in l if x['correct']]; w=[x for x in l if not x['correct']]
    print(tc,'correct n=%d median angle %.2f disp %.2f | wrong n=%d median angle %.1f disp %.1f'%(len(c),np.median([x['dAngle_deg'] for x in c]),np.median([x['dispRMS'] for x in c]),len(w),np.median([x['dAngle_deg'] for x in w]),np.median([x['dispRMS'] for x in w])))
