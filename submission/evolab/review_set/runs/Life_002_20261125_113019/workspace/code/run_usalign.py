"""Run US-align -mm 1 -ter 0 for query vs every target; parse chain mapping, TM, matrix, timing."""
import subprocess, json, re, time, sys, os, numpy as np
q=sys.argv[1]; targets=sys.argv[2:]
U='outputs/tools/USalign'; sd='outputs/structs'; od='outputs/usalign'; os.makedirs(od,exist_ok=True)
for t in targets:
    jf=f'{od}/{q}_vs_{t}.json'
    if os.path.exists(jf): continue
    t0=time.time()
    r=subprocess.run([U,f'{sd}/{q}.pdb',f'{sd}/{t}.pdb','-mm','1','-ter','0','-m',f'{od}/{q}_vs_{t}.mat'],capture_output=True,text=True)
    dt=time.time()-t0
    txt=r.stdout; open(f'{od}/{q}_vs_{t}.txt','w').write(txt)
    n1=re.search(r'Name of Structure_1: \S+?\.pdb(\S*)',txt).group(1)
    n2=re.search(r'Name of Structure_2: \S+?\.pdb(\S*)',txt).group(1)
    c1=[x.split(',')[-1] if x else '' for x in n1.split(':')[1:]]
    c2=[x.split(',')[-1] if x else '' for x in n2.split(':')[1:]]
    pairs=[(a,b) for a,b in zip(c1,c2) if a and b]
    tm=[float(x) for x in re.findall(r'TM-score= ([\d.]+)',txt)]
    m=re.search(r'Aligned length=\s*(\d+), RMSD=\s*([\d.]+), Seq_ID=n_identical/n_aligned=\s*([\d.]+)',txt)
    mat=np.loadtxt(f'{od}/{q}_vs_{t}.mat',skiprows=2,max_rows=3)
    res=dict(query=q,target=t,qtm=tm[0],ttm=tm[1],alnlen=int(m.group(1)),rmsd=float(m.group(2)),seqid=float(m.group(3)),
             pairs=pairs,t=mat[:,1].tolist(),u=mat[:,2:].tolist(),time=dt)
    json.dump(res,open(jf,'w'))
    print(t,res['qtm'],res['ttm'],len(pairs),round(dt,1),flush=True)
