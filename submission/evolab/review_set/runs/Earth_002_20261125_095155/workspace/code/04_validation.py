"""Validation of the baseline TC hazard: return periods at reference sites, basin statistics, wind distribution."""
import numpy as np, json
from scipy.spatial import cKDTree
import model as M
W = M.W
tlon, tlat, wind = M.load_tc(); tb = M.basin_of(tlon, tlat)
# approximate observed return periods (yr) of hurricane-strength (>=33 m/s) storm centres passing within ~100 km
# Sources (indicative): NOAA NHC return periods (within 50 nmi) for Miami / Key West; JTWC/PAGASA, HKO, BoM, IMD/BMD, Meteo-Madagascar climatologies.
REF = {'Miami': (-80.19, 25.76, 7.0), 'Key West': (-81.78, 24.56, 5.5), 'New Orleans': (-90.07, 29.95, 8.0),
       'Belize City': (-88.2, 17.5, 9.0), 'Manila': (120.98, 14.6, 2.5), 'Hong Kong': (114.17, 22.32, 3.0),
       'Okinawa': (127.7, 26.3, 1.5), 'Darwin': (130.84, -12.46, 15.0), 'Cairns': (145.77, -16.92, 8.0),
       'Sundarbans': (89.5, 21.9, 6.0), 'Mahajanga': (46.32, -15.72, 8.0), 'Beira': (34.84, -19.84, 12.0),
       'Nouméa': (166.45, -22.27, 6.0), 'Suva': (178.44, -18.14, 6.0), 'Ho Chi Minh City': (106.7, 10.8, 50.0)}
out = {}
t = cKDTree(M.xyz(tlon, tlat))
for R in [50, 100, 200]:
    w0 = M.point_base_weight(tb, R)
    wu = np.full(len(tb), 47.0 * M.L_STORM / (len(tb) * np.pi * R / 2))   # no basin correction
    for n, (lo, la, obs) in REF.items():
        idx = t.query_ball_point(M.xyz([lo], [la])[0], 2 * np.sin(R / M.RE / 2))
        f = w0[idx].sum(); fu = wu[idx].sum()
        out.setdefault(n, {'obs_rp': obs})
        out[n][f'rp_R{R}'] = 1 / f if f > 0 else np.inf
        out[n][f'rp_R{R}_nocorr'] = 1 / fu if fu > 0 else np.inf
        out[n][f'npts_R{R}'] = len(idx)
for n, d in out.items():
    print(f"{n:18s} obs {d['obs_rp']:5.1f}  R100 {d['rp_R100']:7.1f}  (no-corr {d['rp_R100_nocorr']:7.1f}) R50 {d['rp_R50']:7.1f} R200 {d['rp_R200']:7.1f} n100={d['npts_R100']}")
bas = {b: int((tb == b).sum()) for b in M.BASINS}
print('points per basin', bas)
print('class counts', [int(((wind >= a) & (wind < b)).sum()) for a, b in [(33, 49), (49, 58), (58, 999)]])
json.dump(dict(ref_coords={n: (v[0], v[1]) for n, v in REF.items()}, ref=out, basin_points=bas, obs=M.OBS_HURR), open(W + 'outputs/validation.json', 'w'), indent=1, default=float)
