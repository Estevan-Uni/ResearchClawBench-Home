"""Substrate specificity on the 28 gels with steel data: does cationic ATAC favour glass (anionic silanol surface)?"""
import pandas as pd, numpy as np
from scipy import stats
d = pd.read_csv('outputs/initial_180.csv'); s = d.dropna(subset=['Fa_steel']).copy()
s['log_ratio_glass_over_steel'] = np.log(s.Fa_glass / s.Fa_steel)
out = []
for v in ['HEA', 'BA', 'CBEA', 'ATAC', 'PEA', 'AAm', 'PhaseSep', 'Q']:
    r, p = stats.spearmanr(s[v], s.log_ratio_glass_over_steel); out.append(dict(variable=v, rho_vs_log_glass_over_steel=r, p=p,
        rho_vs_steel=stats.spearmanr(s[v], s.Fa_steel)[0], p_steel=stats.spearmanr(s[v], s.Fa_steel)[1]))
o = pd.DataFrame(out); o.to_csv('outputs/steel_substrate_specificity.csv', index=False); print(o.round(3))
s['hasATAC'] = s.ATAC > 0.005
g = s.groupby('hasATAC').agg(n=('ID', 'size'), med_glass=('Fa_glass', 'median'), med_steel=('Fa_steel', 'median'),
                            med_ratio=('log_ratio_glass_over_steel', lambda x: float(np.exp(x.median()))))
g.to_csv('outputs/steel_by_ATAC.csv'); print(g)
print('MWU log ratio ATAC vs no ATAC p =', stats.mannwhitneyu(s[s.hasATAC].log_ratio_glass_over_steel, s[~s.hasATAC].log_ratio_glass_over_steel).pvalue)
print(d.loc[d.ID == 'G-042', ['Q', 'PhaseSep', 'Modulus', 'tand']], 'tand rank', d.tand.rank(ascending=False)[d.ID == 'G-042'].values)
