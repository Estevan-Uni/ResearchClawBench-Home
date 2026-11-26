import glob, os, pandas as pd, numpy as np
rows=[]
for f in sorted(glob.glob('data/glambie/input/*/*.csv')):
    reg=os.path.basename(os.path.dirname(f)); rid=int(reg.split('_')[0])
    d=pd.read_csv(f); name=os.path.basename(f)[:-4]
    src=[s for s in ['glaciological','demdiff','altimetry','gravimetry','combined'] if f'_{s}_' in name][0]
    step=(d.end_dates-d.start_dates)
    rows.append(dict(file=name,region=reg,rid=rid,source=src,unit=d.unit.iloc[0],n=len(d),t0=d.start_dates.min(),t1=d.end_dates.max(),
        median_step=step.median(),author=d.author.iloc[0],nan_err=int(d.errors.isna().sum()),zero_err=int((d.errors==0).sum()),contiguous=bool(np.allclose(d.start_dates.values[1:],d.end_dates.values[:-1],atol=1e-3)) if len(d)>1 else True))
inv=pd.DataFrame(rows).sort_values(['rid','source'])
os.makedirs('outputs',exist_ok=True); inv.to_csv('outputs/inventory.csv',index=False)
print(inv.source.value_counts()); print(len(inv))
pd.set_option('display.width',250); pd.set_option('display.max_rows',300); pd.set_option('display.max_colwidth',60)
print(inv[['file','unit','n','t0','t1','median_step','nan_err','zero_err','contiguous']].to_string())
