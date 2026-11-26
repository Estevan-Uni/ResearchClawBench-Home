import json, numpy as np, pandas as pd
from scipy import stats
OUT = '/workspace/outputs'
e1 = pd.DataFrame(json.load(open(f'{OUT}/E1_main.json'))).drop(columns='HOTA_alpha')
cols = ['HOTA', 'DetA', 'AssA', 'MOTA', 'IDF1', 'IDSW', 'FP', 'FN', 'Frag', 'MT', 'ML', 'num_pred_ids']
t1 = e1.groupby(['protocol', 'method'])[cols].mean()
t1.round(3).to_csv(f'{OUT}/table_main.csv'); print(t1.round(3).to_string())
e4 = pd.DataFrame(json.load(open(f'{OUT}/E4_density.json'))).drop(columns='HOTA_alpha')
rows = []
for p in ['single', 'occ2stage']:
    for N in [10, 20, 50, 100, 150]:
        s = e4[(e4.protocol == p) & (e4.N == N)].pivot(index='subset', columns='method', values=['HOTA', 'IDF1'])
        for m in ['SparseTrack(DCM,L=2)', 'SparseTrack(DCM,L=3)', 'Depth-gated(g=0.2)']:
            for met in ['HOTA', 'IDF1']:
                d = s[(met, m)] - s[(met, 'ByteTrack')]
                pval = stats.wilcoxon(d).pvalue if np.any(d != 0) else 1.0
                rows.append(dict(protocol=p, N=N, method=m, metric=met, mean_diff=d.mean(),
                                 ci95=1.96 * d.std(ddof=1) / np.sqrt(len(d)), wins=int((d > 0).sum()), losses=int((d < 0).sum()), p_wilcoxon=pval))
t = pd.DataFrame(rows); t.to_csv(f'{OUT}/table_density_paired.csv', index=False)
print(t[t.metric == 'HOTA'].round(4).to_string(index=False))
e2 = pd.DataFrame(json.load(open(f'{OUT}/E2_grid.json'))).drop(columns='HOTA_alpha')
for p in ['single', 'occ2stage']:
    s = e2[e2.protocol == p]
    b = s[s.levels == 1].sort_values('HOTA').iloc[-1]; d = s[s.levels > 1].sort_values('HOTA').iloc[-1]; g = s[s.gate.notna()].sort_values('HOTA').iloc[-1]
    print(p, 'best ByteTrack', b[['match_thresh', 'HOTA', 'IDF1', 'MOTA', 'IDSW']].to_dict())
    print(p, 'best DCM', d[['match_thresh', 'levels', 'HOTA', 'IDF1', 'MOTA', 'IDSW']].to_dict())
    print(p, 'best gate', g[['match_thresh', 'gate', 'HOTA', 'IDF1', 'MOTA', 'IDSW']].to_dict())
