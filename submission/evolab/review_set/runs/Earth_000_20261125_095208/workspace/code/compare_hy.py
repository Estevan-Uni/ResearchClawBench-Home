"""Compare HY regional series of a configuration with official GlaMBIE results."""
import sys, json, numpy as np, pandas as pd
sys.path.insert(0, 'code'); import glambie_lib as L

def compare(cfg, cache=None, verbose=False):
    rows = []
    for reg in L.REGIONS:
        ds = cache[reg] if cache else None
        t, _ = L.region_table(reg, cfg, ds)
        t = t.set_index('hy'); o = L.official_hy(reg)
        for col in ['combined', 'altimetry', 'gravimetry', 'demdiff_and_glaciological']:
            if col + '_mwe' not in t or col + '_mwe' not in o: continue
            j = pd.concat([t[col + '_mwe'].rename('m'), o[col + '_mwe'].rename('o'),
                           t[col + '_mwe_errors'].rename('me'), o[col + '_mwe_errors'].rename('oe')], axis=1).dropna(subset=['o'])
            jj = j.dropna()
            if len(j) == 0: continue
            rows.append(dict(region=reg, group=col, n_off=len(j), n_both=len(jj), n_mine_extra=int(t[col+'_mwe'].loc[2000:2023].notna().sum()-len(jj)),
                             bias=(jj.m - jj.o).mean(), rmsd=np.sqrt(((jj.m - jj.o) ** 2).mean()),
                             r=np.corrcoef(jj.m, jj.o)[0, 1] if len(jj) > 2 else np.nan,
                             err_ratio=(jj.me / jj.oe).median()))
    return pd.DataFrame(rows)

if __name__ == '__main__':
    cfg = json.loads(sys.argv[1]) if len(sys.argv) > 1 else {}
    cache = {r: L.load_region(r) for r in L.REGIONS}
    c = compare(cfg, cache)
    pd.set_option('display.width', 200)
    print(c.round(3).to_string())
