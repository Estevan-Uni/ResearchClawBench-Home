import pickle, numpy as np, collections
recs = pickle.load(open('outputs/data_cache.pkl','rb'))
idx = {(r['family'],r['type'],r['N'],r['d'],r['r']): r for r in recs}
def get(fam,typ,N,d,r): return idx.get((fam,typ,N,d,r))
# d24 vs d12 MB
for r in [1,2,3]:
    a=get('N56_depths','MB',56,24,r); b=get('N56_depths','MB',56,12,r); x=get('N56_depths','XEB',56,12,r)
    print(r, 'd24 target', ''.join(map(str,a['ideal_bitstring'])))
    print(r, 'd12 target', ''.join(map(str,b['ideal_bitstring'])))
    print('  d24 hits',a['counts'].get(a['ideal_bitstring'],0),'/100; top counts',sorted(a['counts'].values())[-3:])
    print('  d24 target in d12 XEB samples?', a['ideal_bitstring'] in x['counts'], ' in d12 MB samples?', a['ideal_bitstring'] in b['counts'])
# hamming distances between d24 and d12 MB targets for all r
hd=[sum(i!=j for i,j in zip(get('N56_depths','MB',56,24,r)['ideal_bitstring'],get('N56_depths','MB',56,12,r)['ideal_bitstring'])) for r in range(1,51)]
print('hamming d24 vs d12 MB targets', hd)
# all-zero targets?
for typ in ['MB','Transport_1QRB']:
    w=[sum(r['ideal_bitstring'])/r['N'] for r in recs if r['type']==typ]
    print(typ,'target weight frac mean',np.mean(w), 'min', np.min(w), 'max',np.max(w))
# are MB targets shared across depths for same r?
for d in [8,10,12,14]:
    print('N56 d',d,'r1 target', ''.join(map(str,get('N56_depths','MB',56,d,1)['ideal_bitstring'])))
# Does N56 XEB r samples relate to MB? check: d24 MB counts frequent bitstrings
a=get('N56_depths','MB',56,24,1); print(sorted(a['counts'].items(), key=lambda t:-t[1])[:2])
# TRB example
t=get('N40_verification','Transport_1QRB',40,96,1); print('TRB target', t['ideal_bitstring']); print(list(t['counts'].items())[:2])
# compare N40 d12 two datasets
a=get('N40_verification','MB',40,12,1); b=get('N_scan_depth12','MB',40,12,1); print('N40 d12 MB targets equal?', a['ideal_bitstring']==b['ideal_bitstring'])
a=get('N40_verification','XEB',40,12,1); b=get('N_scan_depth12','XEB',40,12,1); print('N40 d12 XEB sample overlap', len(set(a['counts'])&set(b['counts'])), 'amp key overlap', len(set(a['ideal_probs'])&set(b['ideal_probs'])))
