"""Load, verify and harmonise all six workbooks. Writes tidy CSVs to outputs/."""
import pandas as pd, numpy as np, re, json
pd.set_option('display.width', 220)
D = 'data/'
MON = ['HEA', 'BA', 'CBEA', 'ATAC', 'PEA', 'AAm']
RAW = ['Nucleophilic-HEA', 'Hydrophobic-BA', 'Acidic-CBEA', 'Cationic-ATAC', 'Aromatic-PEA', 'Amide-AAm']
ren = dict(zip(RAW, MON))
log = {}

# ---------------- initial library ----------------
v = pd.read_excel(D + '184_verified_Original Data_ML_20230926.xlsx').rename(columns=ren)
v = v.rename(columns={'No.': 'ID', 'Glass (kPa)_10s': 'Fa_glass', 'Glass (kPa)_60s': 'Fa_glass60',
                      'Steel (kPa)_10s': 'Fa_steel', 'Steel (kPa)_60s': 'Fa_steel60',
                      'Phase Seperation': 'PhaseSep', 'Modulus (kPa)': 'Modulus', 'Tanδ': 'tand'})
for c in ['Modulus', 'tand', 'Slope', 'Log_Slope', "G''", 'XlogP3', 'Q', 'PhaseSep']:
    v[c] = pd.to_numeric(v[c], errors='coerce')
v['sumx'] = v[MON].sum(1)
v['set'] = np.where(v.ID.str.startswith('G-'), 'DM', 'GPRFR_pilot')
log['n_verified'] = len(v); log['n_DM'] = int((v.set == 'DM').sum())
log['nonnull_60s_glass'] = int(v.Fa_glass60.notna().sum()); log['nonnull_steel10'] = int(v.Fa_steel.notna().sum())
log['sum_composition_range'] = [float(v.sumx.min()), float(v.sumx.max())]

# historical snapshots
ver = []
snaps = {'20220829': ('Original Data_ML_20220829.xlsx', 'Glass (kPa)'),
         '20221031': ('Original Data_ML_20221031.xlsx', 'Glass (kPa)_10s'),
         '20221129': ('Original Data_ML_20221129.xlsx', 'Glass (kPa)_10s')}
pilots = []
for k, (f, yc) in snaps.items():
    s = pd.read_excel(D + f).rename(columns=ren)
    g = s[s['No.'].astype(str).str.startswith('G-')].set_index('No.')
    vv = v[v.set == 'DM'].set_index('ID')
    common = vv.index.intersection(g.index)
    dx = float(np.abs(g.loc[common, MON].values - vv.loc[common, MON].values).max())
    dy = float(np.nanmax(np.abs(g.loc[common, yc].values - vv.loc[common, 'Fa_glass'].values)))
    others = s[~s['No.'].astype(str).str.startswith('G-')]
    ver.append(dict(snapshot=k, n_rows=len(s), n_G=len(g), n_common=len(common), max_abs_dx=dx, max_abs_dy=dy,
                    other_ids=';'.join(others['No.'].astype(str))))
    if len(others):
        o = others.rename(columns={'No.': 'ID', 'Glass (kPa)_10s': 'Fa_glass', 'Glass (kPa)_60s': 'Fa_glass60'})
        o['snapshot'] = k
        pilots.append(o[['ID', 'snapshot'] + MON + ['Fa_glass', 'Fa_glass60']])
ver = pd.DataFrame(ver); ver.to_csv('outputs/verification_snapshots.csv', index=False)
print(ver.to_string())
pil = pd.concat(pilots)
# check the two P-snapshots agree
p1 = pil[pil.snapshot == '20221031'].set_index('ID'); p2 = pil[pil.snapshot == '20221129'].set_index('ID')
log['P_snapshots_max_abs_diff'] = float(np.nanmax(np.abs(p1[MON + ['Fa_glass', 'Fa_glass60']].values - p2[MON + ['Fa_glass', 'Fa_glass60']].values)))
pil = p2.reset_index(); pil.to_csv('outputs/pilot_P_series.csv', index=False)
v.to_csv('outputs/initial_184.csv', index=False)
v[v.set == 'DM'].to_csv('outputs/initial_180.csv', index=False)

# ---------------- optimisation rounds ----------------
rows = []
for sheet in ['EI', 'PRED']:
    d = pd.read_excel(D + 'ML_ei&pred (1&2&3rounds)_20240408.xlsx', sheet_name=sheet).rename(columns=ren)
    d['ML'] = d['ML'].ffill()
    d = d.rename(columns={'NO.': 'pos', 'Glass (kPa)_max': 'Fa_raw'})
    for i, r in d.iterrows():
        lab = r.ML
        m = re.match(r'(.*)-(\d)rd-(ei|pred)$', lab)
        if m: method, rnd, crit = m.group(1), int(m.group(2)), m.group(3).upper()
        else: method, rnd, crit = lab, 1, sheet
        rec = dict(sheet=sheet, label=lab, method=method, round=rnd, criterion=crit, pos=int(r.pos), note='')
        h = str(r.HEA)
        dup = re.match(r'与(\d+)号重复', h)
        if dup:  # "duplicate of #k" within same list
            k = int(dup.group(1)); src = d[(d.ML == lab) & (d.pos == k)].iloc[0]
            for mm in MON: rec[mm] = float(src[mm])
            rec['note'] = f'duplicate_of_pos{k}'
        elif 'NO GELATION' in h:
            for mm in MON: rec[mm] = np.nan
            rec['note'] = 'NO_GELATION_composition_not_recorded'
        else:
            for mm in MON: rec[mm] = float(r[mm])
        fa = r.Fa_raw
        rec['gelled'] = not ('NO GELATION' in str(fa) or 'NO GELATION' in h)
        rec['Fa_glass'] = pd.to_numeric(fa, errors='coerce') if rec['gelled'] else np.nan
        if not rec['gelled'] and rec['note'] == '': rec['note'] = 'NO_GELATION'
        rows.append(rec)
ml = pd.DataFrame(rows)
ml['sumx'] = ml[MON].sum(1)
ml.loc[ml.sumx > 0, MON] = ml.loc[ml.sumx > 0, MON].div(ml.loc[ml.sumx > 0, 'sumx'], axis=0)
ml[MON] = ml[MON].astype(float)
keyfun = lambda df: df[MON].astype(float).round(3).apply(lambda r: '|'.join(f'{x:.3f}' for x in r), axis=1)
ml['key'] = keyfun(ml)
ml.to_csv('outputs/ml_designs_all_lists.csv', index=False)
log['labels_EI'] = ml[ml.sheet == 'EI'].label.unique().tolist()
log['labels_PRED'] = ml[ml.sheet == 'PRED'].label.unique().tolist()
log['list_entries_per_round'] = ml.groupby('round').size().to_dict()

# subset file check
sub = []
for sheet in ['EI', 'PRED']:
    a = pd.read_excel(D + 'ML_ei&pred_20240213.xlsx', sheet_name=sheet).rename(columns=ren)
    b = pd.read_excel(D + 'ML_ei&pred (1&2&3rounds)_20240408.xlsx', sheet_name=sheet).rename(columns=ren).iloc[:len(a)]
    aa = a[MON + ['Glass (kPa)_max']].apply(pd.to_numeric, errors='coerce').values
    bb = b[MON + ['Glass (kPa)_max']].apply(pd.to_numeric, errors='coerce').values
    sub.append(dict(sheet=sheet, n_0213=len(a), max_abs_diff_vs_0408=float(np.nanmax(np.abs(aa - bb)))))
log['subset_file_check'] = sub

# unique formulations (tested, gelled or not, with known composition)
mk = ml[ml.HEA.notna()]
uni = mk.groupby('key').agg(round=('round', 'min'), n_lists=('label', 'nunique'), n_entries=('label', 'size'),
                             labels=('label', lambda s: ';'.join(sorted(set(s)))),
                             Fa_glass=('Fa_glass', 'max'), Fa_min=('Fa_glass', 'min'),
                             gelled=('gelled', 'all'), **{m: (m, 'first') for m in MON}).reset_index()
log['unique_value_conflicts'] = int((uni.Fa_glass - uni.Fa_min).abs().gt(1e-6).sum())
# remove formulations already present in initial set
v['key'] = keyfun(v)
uni['in_initial'] = uni.key.isin(v.key)
# keep only first round in which composition appears
uni.to_csv('outputs/ml_unique_formulations.csv', index=False)
cum = [len(v)]
for r in [1, 2, 3]:
    newk = set(uni[(uni['round'] == r) & (~uni.in_initial)].key)
    cum.append(cum[-1] + len(newk))
log['unique_new_per_round'] = uni[~uni.in_initial].groupby('round').size().to_dict()
log['cumulative_unique_dataset_size'] = cum
log['no_gelation_entries'] = ml[~ml.gelled][['label', 'pos', 'note']].to_dict('records')

# Training table for modelling: 184 initial + unique ML formulations (gelled, known comp)
tr = pd.concat([v[['ID'] + MON + ['Fa_glass']].assign(round=0, source=v.set),
                uni[(uni.gelled) & (~uni.in_initial)].assign(ID=lambda x: 'R' + x['round'].astype(str) + '_' + x.index.astype(str), source='ML')[['ID'] + MON + ['Fa_glass', 'round', 'source', 'labels']]],
               ignore_index=True)
tr.to_csv('outputs/all_measured_formulations.csv', index=False)
log['n_all_measured'] = len(tr)

# Key formulations
dm = v[v.set == 'DM']
g = dm.loc[dm.Fa_glass.idxmax()]
keyf = [dict(name='G-max (best DM/protein-derived gel)', ID=g.ID, source='184_verified', **{m: round(g[m], 4) for m in MON}, Fa_glass_kPa=round(g.Fa_glass, 1))]
pp = v.loc[v[v.set != 'DM'].Fa_glass.idxmax()]
keyf.append(dict(name='Best GPRFR pilot (in 184 training set)', ID=pp.ID, source='184_verified', **{m: round(pp[m], 4) for m in MON}, Fa_glass_kPa=round(pp.Fa_glass, 1)))
for r in [1, 2, 3]:
    s = ml[(ml['round'] == r) & ml.gelled]
    b = s.loc[s.Fa_glass.idxmax()]
    same = s[np.isclose(s.Fa_glass, b.Fa_glass)]
    keyf.append(dict(name=f'R{r}-max (best round-{r} ML design)', ID=f'{b.sheet}:{b.label}#{b.pos}',
                     source='lists with same value: ' + ';'.join(same.sheet + ':' + same.label + '#' + same.pos.astype(str)),
                     **{m: round(b[m], 4) for m in MON}, Fa_glass_kPa=round(b.Fa_glass, 1)))
keyf = pd.DataFrame(keyf); keyf.to_csv('outputs/key_formulations.csv', index=False)
print(keyf.to_string())
log['DM_Fa_range'] = [float(dm.Fa_glass.min()), float(dm.Fa_glass.max())]; log['DM_Fa_median'] = float(dm.Fa_glass.median())
log['ML_gelled_entries_median'] = float(ml[ml.gelled].Fa_glass.median())
log['ML_unique_median'] = float(uni[uni.gelled & ~uni.in_initial].Fa_glass.median())
json.dump(log, open('outputs/data_log.json', 'w'), indent=1, default=str)
print(json.dumps(log, indent=1, default=str))
print(pil.to_string())
