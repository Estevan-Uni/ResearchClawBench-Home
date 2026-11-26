import pandas as pd
DEGS=['Average','Misalignment','Missing Sections','Mixed','All']
R=pd.read_csv('/workspace/outputs/metrics_long.csv')
order=['M','I','E','MI','ME','IE','MIE']; lab={'M':'M','I':'I','E':'E','MI':'M+I','ME':'M+E','IE':'I+E','MIE':'M+I+E'}
sch=['QDA-blockmean','LR-perdeg','HGB-early','MLP-early','Late-LR','Gated-LR','HGB-experts','Ungated-enc','SE-gated-nodeg','SE-gated-deg']
out=['| Scheme | Modalities | '+' | '.join(f'{d} P / R / F1' for d in DEGS)+' |','|'+'---|'*(2+len(DEGS))]
for s in sch:
    for sub in order:
        q=R[(R.scheme==s)&(R.subset==sub)]
        if len(q)==0: continue
        cells=[]
        for d in DEGS:
            r=q[q.degradation==d].iloc[0]; cells.append(f'{r.P:.3f} / {r.R:.3f} / **{r.F1:.3f}**' if d=='All' else f'{r.P:.3f} / {r.R:.3f} / {r.F1:.3f}')
        out.append(f'| {s} | {lab[sub]} | '+' | '.join(cells)+' |')
open('/workspace/outputs/fusion_table.md','w').write('\n'.join(out)); print('\n'.join(out))
