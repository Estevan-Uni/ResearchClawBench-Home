"""5x5 Csf2 (Cas7) homomer swap matrix: chainTM, rotation angle and displacement of each pair's own
superposition relative to the complex superposition. Shows why per-chain superpositions disambiguate homomers."""
import json, numpy as np
d=json.load(open('outputs/chain_tables.json'))
res={}
for tc in ['9N3Y','8RFJ','8S35']:
    tab=d['tables'][tc]; amap={p['qchain']:p['tchain'] for p in tab['pairs']}
    q=['C','D','E','F','G']; tt=[amap[x] for x in q]
    allp={(x['qchain'],x['tchain']):x for x in d['all_chain_pairs'][tc]}
    M={k:np.full((5,5),np.nan) for k in ['chainTM','angle','disp']}
    for i,a in enumerate(q):
        for j,b in enumerate(tt):
            x=allp.get((a,b))
            if x: M['chainTM'][i,j]=x['chainTM_q']; M['angle'][i,j]=x['dAngle_deg']; M['disp'][i,j]=x['dispRMS']
    res[tc]=dict(qchains=q,tchains=tt,**{k:v.round(3).tolist() for k,v in M.items()})
    print(tc,'target Csf2 order',tt); 
    for k in M: print(k); print(np.round(M[k],2))
json.dump(res,open('outputs/csf2_swap.json','w'),indent=1)
