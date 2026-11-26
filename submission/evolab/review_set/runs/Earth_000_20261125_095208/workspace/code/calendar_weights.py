"""Diagnose calendar-conversion weights.
(1) weights implied by the official hydrological/calendar pairs (least squares, per region)
(2) independent seasonal weights from the monthly glaciological sine-wave product (Zemp et al.):
    regress calendar-year sums of the monthly series on HY sums of the two overlapping HYs."""
import sys, numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, 'code'); import glambie_lib as L

def implied_official(reg):
    rid = L.rid_of(reg)
    if rid == 16: return 0.0
    h = L.official_hy(reg).combined_gt; c = L.official_cy(reg).combined_gt
    Ys = [y for y in range(2001, 2023) if y in c.index]
    nb = (lambda y: y + 1) if rid <= 15 else (lambda y: y - 1)
    Ys = [y for y in Ys if y in h.index and nb(y) in h.index]
    x = np.array([h[nb(y)] - h[y] for y in Ys]); r = np.array([c[y] - h[y] for y in Ys])
    return float(np.sum(x * r) / np.sum(x * x))

def sine_wave(reg):
    """Calendar vs HY weight from the monthly glaciological product. Because a fixed seasonal
    cycle cancels between consecutive years, the implied weight reflects how the inter-annual
    anomaly is distributed over months in that product."""
    rid = L.rid_of(reg)
    if rid == 16: return 0.0, np.nan
    ds = [d for d in L.load_region(reg) if d['kind'] == 'monthly' and d['src'] == 'glaciological']
    if not ds:
        return np.nan, np.nan
    d = ds[0]['data']; t0 = np.round(d.t0.values * 12) / 12; v = d.v.values
    off = L.hy_offset(rid)
    H = {i: v[(t0 >= i + off - 1e-6) & (t0 < i + 1 + off - 1e-6)].sum() for i in range(2000, 2024)}
    C = {y: v[(t0 >= y - 1e-6) & (t0 < y + 1 - 1e-6)].sum() for y in range(2001, 2023)}
    nb = (lambda y: y + 1) if rid <= 15 else (lambda y: y - 1)
    Ys = [y for y in C if y in H and nb(y) in H]
    x = np.array([H[nb(y)] - H[y] for y in Ys]); r = np.array([C[y] - H[y] for y in Ys])
    w = float(np.sum(x * r) / np.sum(x * x))
    # monthly distribution of the anomaly: fraction of the variance in the neighbour-year months
    return w, np.corrcoef(x, r)[0, 1]

if __name__ == '__main__':
    rows = []
    for reg in L.REGIONS:
        rid = L.rid_of(reg)
        wo = implied_official(reg); ws, r = sine_wave(reg)
        rows.append(dict(region=reg, hy_start={True: '1 Oct', False: ('1 Jan' if rid == 16 else '1 Apr')}[rid <= 15],
                         w_official_implied=wo, w_sinewave=ws, w_time=0.0 if rid == 16 else 0.25))
    df = pd.DataFrame(rows); df.to_csv('outputs/calendar_weights.csv', index=False); print(df.round(3).to_string())
