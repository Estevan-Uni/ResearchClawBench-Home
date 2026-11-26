"""Pre-compute neighbour matrices (sites x TC points) for R=50/100/200 km and SLR metrics per site."""
import numpy as np, json, time, shapely
from scipy import sparse
from shapely import STRtree
import model as M
W = M.W
t0 = time.time()
S, cw = M.load_sites()
# snap remaining unassigned points to nearest country (no distance cap) for service weights
geoms = [shapely.from_wkt(g) for g in json.load(open(W + 'outputs/cwon_geom_wkt.json'))]
cidx = S['cidx'].copy(); miss = np.where(cidx < 0)[0]
tree = STRtree(geoms)
ni, d = tree.query_nearest(shapely.points(S['lon'][miss], S['lat'][miss]), return_distance=True)
cidx[miss[ni[0]]] = np.where(d <= 3.0, ni[1], -1)
np.save(W + 'outputs/site_snapdist.npy', np.r_[d])
print('snapped far points', len(miss), 'max dist deg', d.max())
np.save(W + 'outputs/site_cidx.npy', cidx)
tlon, tlat, tw = M.load_tc()
np.save(W + 'outputs/tc_basin.npy', M.basin_of(tlon, tlat))
np.save(W + 'outputs/site_basin.npy', M.basin_of(S['lon'], S['lat']))
import sys
if 'slr_only' in sys.argv: RS=[]
else: RS=[50,100,200]
for R in RS:
    A = M.neighbour_matrix(S['lon'], S['lat'], tlon, tlat, R)
    sparse.save_npz(W + f'outputs/nbr_R{R}.npz', A)
    print('R', R, 'nnz', A.nnz, 'mean nbrs', A.nnz / A.shape[0], time.time() - t0, flush=True)
out = {}
for ssp in M.SSPS:
    rates, q, yrs, dist = M.load_slr(ssp, S['lon'], S['lat'])
    yi = (yrs >= 2020) & (yrs <= 2100); yl = (yrs >= 2080) & (yrs <= 2100)
    qi = {k: int(np.argmin(np.abs(q - v))) for k, v in [('q05', .05), ('q17', .17), ('q50', .5), ('q83', .83), ('q95', .95)]}
    for k, i in qi.items():
        out[f'{ssp}_{k}'] = np.nanmean(rates[:, i, yi], axis=1)
    out[f'{ssp}_q50_late'] = np.nanmean(rates[:, qi['q50'], yl], axis=1)
    # full quantile curve of 2020-2100 mean (for Monte Carlo)
    out[f'{ssp}_qcurve'] = np.nanmean(rates[:, :, yi], axis=2).astype(np.float32)
    out['quantiles'] = q; out['nn_dist_km'] = dist
    print(ssp, 'years used', yrs[yi], 'median rate pctl', np.nanpercentile(out[f'{ssp}_q50'], [5, 50, 95]), time.time() - t0, flush=True)
np.savez(W + 'outputs/slr_sites.npz', **out)
print('done', time.time() - t0)
