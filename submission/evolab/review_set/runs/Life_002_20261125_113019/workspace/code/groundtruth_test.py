"""Ground-truth checks: (1) rigid transform of 7XG4 with known R,t -> does Foldseek-Multimer recover them?
(2) Csf2 chain labels permuted (C..G shuffled) + rigid transform -> is the chain correspondence recovered from geometry?
(3) a sub-complex (Csf2 filament only, chains C-G) as query."""
import numpy as np, subprocess, json, sys, os
sys.path.insert(0,'code'); import pdbutil as P
from parse_foldseek import load_m8, best_assignments, rot_angle
from scipy.spatial.transform import Rotation as Rot
F='outputs/tools/foldseek'; od='outputs/groundtruth'; os.makedirs(od,exist_ok=True)
FMT=open('outputs/pair/fmt.txt').read().strip()
atoms=P.read_pdb_atoms('outputs/structs/7XG4.pdb')
rng=np.random.default_rng(7); res=[]
def write(atoms,R,t,relabel,keep,fn):
    out=[]
    for a in atoms:
        if a['chain'] not in keep: continue
        b=dict(a); x=R@np.array([a['x'],a['y'],a['z']])+t; b['x'],b['y'],b['z']=x; b['chain']=relabel.get(a['chain'],a['chain']); out.append(b)
    # order chains alphabetically by new label to avoid order leaking
    out.sort(key=lambda b:b['chain']); P.write_clean_pdb(out,fn)
cases=[]
for k in range(6):
    R=Rot.random(random_state=k).as_matrix(); t=rng.uniform(-200,200,3)
    perm=dict(zip('CDEFG',rng.permutation(list('CDEFG')))) if k>=2 else {}
    keep=set('ABCDEFGHL') if k<5 else set('CDEFG')
    cases.append((f'case{k}',R,t,perm,keep))
for name,R,t,perm,keep in cases:
    fn=f'{od}/{name}.pdb'; write(atoms,R,t,perm,keep,fn)
    subprocess.run([F,'easy-multimersearch','outputs/structs/7XG4.pdb' if len(keep)==9 else fn, fn if len(keep)==9 else 'outputs/structs/7XG4.pdb',f'{od}/{name}.m8',f'/tmp/gt_{name}','--exhaustive-search','1','-e','inf','--min-aligned-chains','1','--format-output',FMT,'--threads','1'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    b=best_assignments(load_m8(f'{od}/{name}.m8')); rs=list(b.values())[0]
    cU,cT=rs[0]['cU'],rs[0]['cT']
    if len(keep)==9:
        # query = 7XG4 (orig), target = transformed; Foldseek U maps target->query, so expected U = R^T, t = -R^T t
        expU,expT=R.T,-R.T@t; mapping={r['qch']:r['tch'] for r in rs}; truth={c:perm.get(c,c) for c in keep}
    else:
        expU,expT=R,t; mapping={r['qch']:r['tch'] for r in rs}; inv={v:k for k,v in perm.items()}; truth={perm.get(c,c):c for c in keep}
    correct=sum(mapping.get(c)==truth[c] for c in truth)
    d=dict(case=name,nchains=len(keep),permuted=bool(perm),perm=perm,complexqTM=rs[0]['complexqtmscore'],n_pairs=len(rs),correct_pairs=correct,
           angle_err_deg=rot_angle(cU,expU),t_err_A=float(np.linalg.norm(cT-expT)))
    res.append(d); print(d,flush=True)
json.dump(res,open(f'{od}/groundtruth.json','w'),indent=1)
