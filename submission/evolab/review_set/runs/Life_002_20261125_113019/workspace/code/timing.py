"""Isolated pairwise timing: Foldseek-Multimer (end-to-end easy-multimersearch, and with pre-built DBs)
vs US-align -mm 1 -ter 0, same pair, single thread, >=3 repeats. Also database-amortised Foldseek cost."""
import subprocess, time, json, os, shutil, statistics as st
F='outputs/tools/foldseek'; U='outputs/tools/USalign'; S='outputs/structs'; od='outputs/timing'; os.makedirs(od,exist_ok=True)
FMT=open('outputs/pair/fmt.txt').read().strip()
def load(): return open('/proc/loadavg').read().split()[0]
def run(cmd):
    t=time.perf_counter(); r=subprocess.run(cmd,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL); return time.perf_counter()-t, r.returncode
res={'pairs':{}, 'loadavg_start':open('/proc/loadavg').read().strip(), 'nproc':os.cpu_count()}
pairs=[('7XG4','9N3Y'),('7XG4','8S35'),('7XG4','6N40')]
for q,t in pairs:
    # prebuilt dbs
    for x in (q,t):
        if not os.path.exists(f'{od}/{x}_db'): subprocess.run([F,'createdb',f'{S}/{x}.pdb',f'{od}/{x}_db','--threads','1'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    R={'fs_easy':[],'fs_db':[],'us':[],'load':[]}
    for rep in range(5):
        tmp=f'/tmp/fst_{q}{t}{rep}'; shutil.rmtree(tmp,ignore_errors=True)
        dt,rc=run([F,'easy-multimersearch',f'{S}/{q}.pdb',f'{S}/{t}.pdb',f'{od}/o.m8',tmp,'--exhaustive-search','1','-e','inf','--min-aligned-chains','1','--format-output',FMT,'--threads','1'])
        R['fs_easy'].append(dt)
        tmp2=tmp+'b'; shutil.rmtree(tmp2,ignore_errors=True); 
        for f in os.listdir(od):
            if f.startswith('res_'): os.remove(f'{od}/{f}')
        dt,rc2=run([F,'multimersearch',f'{od}/{q}_db',f'{od}/{t}_db',f'{od}/res_aln',tmp2,'--exhaustive-search','1','-e','inf','--min-aligned-chains','1','--threads','1'])
        R['fs_db'].append(dt); R['load'].append(load())
    for rep in range(3):
        dt,rc=run([U,f'{S}/{q}.pdb',f'{S}/{t}.pdb','-mm','1','-ter','0']); R['us'].append(dt); R['load'].append(load())
    for k in ['fs_easy','fs_db','us']: R[k+'_median']=st.median(R[k])
    R['speedup_easy']=R['us_median']/R['fs_easy_median']; R['speedup_db']=R['us_median']/R['fs_db_median']
    res['pairs'][f'{q}-{t}']=R; print(q,t,{k:(round(v,3) if isinstance(v,float) else v) for k,v in R.items() if 'median' in k or 'speed' in k},flush=True)
# database-amortised: 7XG4 vs 25-entry db replicated x8 (200 entries), exhaustive, createdb excluded
rep_dir=f'{od}/repdb'; os.makedirs(rep_dir,exist_ok=True)
ids=open('outputs/db/ids.txt').read().split()
for k in range(8):
    for i in ids:
        dst=f'{rep_dir}/{i}r{k}.pdb'
        if not os.path.exists(dst): shutil.copy(f'{S}/{i}.pdb',dst)
subprocess.run([F,'createdb',rep_dir,f'{od}/rep_db','--threads','1'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
subprocess.run([F,'createdb',f'{S}/7XG4.pdb',f'{od}/q_db','--threads','1'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
am={}
for mode,extra in [('exhaustive',['--exhaustive-search','1','-e','inf']),('prefilter_default',[]),('prefilter_maxseqs3000',['--max-seqs','3000'])]:
    ts=[]
    for rep in range(3):
        tmp=f'/tmp/fsrep_{mode}{rep}'; shutil.rmtree(tmp,ignore_errors=True)
        for f in os.listdir(od):
            if f.startswith('rep_res'): os.remove(f'{od}/{f}')
        dt,rc=run([F,'multimersearch',f'{od}/q_db',f'{od}/rep_db',f'{od}/rep_res',tmp,'--min-aligned-chains','1','--threads','1']+extra); ts.append(dt)
    subprocess.run([F,'createtsv',f'{od}/q_db',f'{od}/rep_db',f'{od}/rep_res',f'{od}/rep_{mode}.tsv'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    hits=set(l.split('\t')[1].split('_')[0] for l in open(f'{od}/rep_{mode}.tsv'))
    am[mode]=dict(times=ts,median=st.median(ts),n_entries=len(ids)*8,per_entry=st.median(ts)/(len(ids)*8),n_hit_complexes=len(hits))
    print(mode,am[mode],flush=True)
res['amortised']=am; res['loadavg_end']=open('/proc/loadavg').read().strip()
json.dump(res,open(f'{od}/timing.json','w'),indent=1)
