import pandas as pd, glob
for f in sorted(glob.glob('data/*.xlsx')):
    x = pd.ExcelFile(f)
    for s in x.sheet_names:
        d = x.parse(s)
        print('=====', f, s, d.shape)
        print(d.columns.tolist())
        print(d.head(4).to_string())
