"""Core model: mangrove TC regime-shift (Kropf et al.) + SLR (Saintilan et al.) composite risk."""
import json, numpy as np, h5py
from scipy.spatial import cKDTree
from scipy import sparse
W = '/workspace/'
RE = 6371.0
# ----------------------------------------------------------------- constants
BASINS = ['NA', 'EP', 'WP', 'NI', 'SI', 'SP', 'SA']
# Observed mean annual number of storms reaching hurricane strength (>=33 m/s), IBTrACS 1980-2018 (approx.)
OBS_HURR = dict(NA=6.4, EP=9.0, WP=16.5, NI=1.5, SI=8.0, SP=4.5, SA=0.05)
L_STORM = 1200.0          # km of track at >=33 m/s per storm (approx. 2.5 d x 20 km/h)
CLASS_EDGES = [33.0, 49.0, 58.0]   # Kropf L/M/H (m/s)
F_REG = 1 / 20.0          # regular disturbance frequency
F_MIN = 1 / 250.0         # affected threshold
# warming of 2081-2100 relative to 1850-1900 (AR6 WG1 SPM best estimate, likely range)
GSAT = {'ssp245': (2.7, 2.1, 3.5), 'ssp370': (3.6, 2.8, 4.6), 'ssp585': (4.4, 3.3, 5.7)}
BASE_WARM = 0.3           # mean warming of the 1850-2014 track period vs 1850-1900
SSPS = ['ssp245', 'ssp370', 'ssp585']
# delta-change variants, values per 2 degC of warming (fractional)
VARIANTS = {
  # A: MIT/Emanuel-consistent (Emanuel 2013 PNAS; Emanuel 2021 J Clim): frequency increases, LMI +
  'A_MIT': dict(df=dict(NA=0.10, EP=0.10, WP=0.10, NI=0.10, SI=0.10, SP=0.10, SA=0.0), di=0.05),
  # B: Knutson et al. 2020 consensus (median): frequency decreases, LMI +5%
  'B_Knutson': dict(df=dict(NA=-0.10, EP=-0.10, WP=-0.20, NI=-0.10, SI=-0.20, SP=-0.25, SA=0.0), di=0.05),
}
EP_LON = np.array([-100, -97, -90, -85, -80, -77, -75.0]); EP_LAT = np.array([22, 19, 15.5, 13.5, 8.5, 8, 8.0])

def basin_of(lon, lat):
    lon = ((np.asarray(lon) + 180) % 360) - 180; lat = np.asarray(lat)
    b = np.full(lon.shape, 'NA', dtype='<U2')
    nh = lat >= 0
    ep = nh & ((lon < -100) | ((lon < -75) & (lat < np.interp(lon, EP_LON, EP_LAT))))
    b[ep] = 'EP'
    b[nh & (lon >= 100)] = 'WP'
    b[nh & (lon >= 30) & (lon < 100)] = 'NI'
    b[nh & ~ep & (lon < 30) & (lon >= -75)] = 'NA'
    b[nh & ~ep & (lon < -75) & (lon >= -100)] = 'NA'
    sh = ~nh
    b[sh & (lon >= 20) & (lon < 135)] = 'SI'
    b[sh & ((lon >= 135) | (lon < -70))] = 'SP'
    b[sh & (lon >= -70) & (lon < 20)] = 'SA'
    return b

def xyz(lon, lat):
    lo, la = np.radians(lon), np.radians(lat)
    return np.c_[np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)]

# ----------------------------------------------------------------- loaders
def load_tc():
    f = h5py.File(W + 'data/tc/tracks_mit_mpi-esm1-2-hr_historical_reduced.nc', 'r')
    lat, lon, w = f['lat'][:].astype(float), f['lon'][:].astype(float), f['wind'][:].astype(float)
    return lon, lat, w

def load_sites():
    s = np.load(W + 'outputs/sites.npz')
    cw = json.load(open(W + 'outputs/cwon.json'))
    return dict(lon=s['lon'], lat=s['lat'], cls=s['cls'], cidx=s['cidx']), cw

def site_weights(cidx, cw, mask):
    """Spread CWON 2020 country totals equally over the selected points of each country."""
    out = {}
    for key, col in [('area', 'Mang_Ha_2020'), ('pop', 'Ben_Pop_2020'), ('stock', 'Ben_Stock_2020')]:
        tot = np.array([float(c[col] or 0) for c in cw])
        ok = mask & (cidx >= 0)
        n = np.bincount(cidx[ok], minlength=len(cw)).astype(float)
        w = np.zeros(len(cidx))
        w[ok] = tot[cidx[ok]] / n[cidx[ok]]
        out[key] = w
    return out

def load_slr(ssp, site_lon, site_lat):
    """Return rates (nsite x nquantile x nyears), quantiles, years, nn distance (km)."""
    f = h5py.File(W + f'data/slr/total_{ssp}_medium_confidence_rates.nc', 'r')
    lat, lon = f['lat'][:], f['lon'][:]
    q, yrs = f['quantiles'][:], f['years'][:]
    v = f['sea_level_change_rate']
    # valid locations: median rate at 2020 not fill
    med_i = int(np.argmin(np.abs(q - 0.5)))
    test = v[med_i, 0, :]
    valid = np.where((test != -32768) & np.isfinite(lat) & np.isfinite(lon))[0]
    tree = cKDTree(xyz(lon[valid], lat[valid]))
    d, j = tree.query(xyz(site_lon, site_lat))
    loc = valid[j]
    uloc, inv = np.unique(loc, return_inverse=True)
    raw = v[:, :, :][:, :, uloc].astype(float)            # (q, yr, uloc)
    raw[raw == -32768] = np.nan
    raw *= float(v.attrs['scale_factor'][0])
    rates = np.transpose(raw, (2, 0, 1))[inv]              # (site, q, yr)
    dist = 2 * RE * np.arcsin(np.clip(d / 2, 0, 1))
    return rates, q, yrs, dist

def slr_level(rate):
    return np.where(rate >= 7.0, 2, np.where(rate >= 4.0, 1, 0))

# ----------------------------------------------------------------- TC hazard
def neighbour_matrix(site_lon, site_lat, tc_lon, tc_lat, R):
    """Sparse (nsite x ntc) indicator of TC points within R km of each site."""
    chord = 2 * np.sin(R / RE / 2)
    t = cKDTree(xyz(tc_lon, tc_lat))
    lists = t.query_ball_point(xyz(site_lon, site_lat), chord, workers=2)
    ind = np.concatenate([np.asarray(l, dtype=np.int32) for l in lists]) if len(lists) else np.zeros(0, int)
    ptr = np.r_[0, np.cumsum([len(l) for l in lists])]
    return sparse.csr_matrix((np.ones(len(ind), np.float32), ind, ptr), shape=(len(site_lon), len(tc_lon)))

def point_base_weight(tc_basin, R, kappa_mult=1.0):
    """Per-point annual passage weight: kappa_b = N_b L / (NP_b * pi R / 2), basin bias-corrected."""
    w = np.zeros(len(tc_basin))
    for b in BASINS:
        m = tc_basin == b
        if m.sum():
            w[m] = OBS_HURR[b] * L_STORM / (m.sum() * np.pi * R / 2)
    if isinstance(kappa_mult, dict):
        km = np.ones(len(tc_basin))
        for b, v in kappa_mult.items():
            km[tc_basin == b] = v
        return w * km
    return w * kappa_mult

def class_weights(wind, w0, tc_basin, df=None, di=0.0, rho=1.0):
    """Return (3 x ntc) weights for classes L,M,H after delta change.
    df: dict basin->fractional freq change; di: fractional wind (LMI) change.
    Points that newly exceed 33 m/s are represented by pseudo-points (re-using points with
    33 <= w < 33(1+di) as proxies for the unobserved sub-threshold segments; density ratio rho)."""
    fm = np.ones(len(wind))
    if df is not None:
        for b, v in df.items():
            fm[tc_basin == b] = 1 + v
    w = w0 * fm
    wf = wind * (1 + di)
    W3 = np.zeros((3, len(wind)))
    e = CLASS_EDGES
    W3[0] = w * ((wf >= e[0]) & (wf < e[1]))
    W3[1] = w * ((wf >= e[1]) & (wf < e[2]))
    W3[2] = w * (wf >= e[2])
    if di > 0:
        pseudo = (wind >= e[0]) & (wind < e[0] * (1 + di))
        W3[0] += rho * w * pseudo
    return W3

def site_freq(A, W3):
    return np.vstack([A @ W3[c] for c in range(3)]).T     # (nsite x 3)

# ----------------------------------------------------------------- Kropf states / regime shift
def unit_ids(lon, lat, size):
    if isinstance(size, str):
        raise ValueError
    ix = np.floor((lon + 180) / size).astype(int); iy = np.floor((lat + 90) / size).astype(int)
    return ix * 10000 + iy

def unit_percentiles(f, uid):
    """p20, p80 per unit (for each class). Returns arrays mapped to sites."""
    u, inv = np.unique(uid, return_inverse=True)
    order = np.argsort(inv, kind='stable'); inv_s = inv[order]
    starts = np.r_[0, np.flatnonzero(np.diff(inv_s)) + 1, len(inv_s)]
    p20 = np.zeros((len(u), f.shape[1])); p80 = np.zeros_like(p20)
    fs = f[order]
    for k in range(len(u)):
        blk = fs[starts[k]:starts[k + 1]]
        p20[k] = np.percentile(blk, 20, axis=0); p80[k] = np.percentile(blk, 80, axis=0)
    return u, inv, p20, p80

STATE_NAMES = {0: 'unaffected', 1: 'vulnerable', 2: 'resilient', 3: 'dependent'}

def kropf_states(p20, p80):
    st = np.where(p80 >= F_MIN, 1, 0)
    st = np.where(p80 >= F_REG, 2, st)
    st = np.where(p20 >= F_REG, 3, st)
    return st

def regime_shift(f0, f1, uid, newly_rule=True):
    """Unit-level Kropf test. Returns per-site: shift flags (n x 3), states (n x 3), r (n x 3), tc level."""
    u, inv, p20, p80 = unit_percentiles(f0, uid)
    _, _, q20, q80 = unit_percentiles(f1, uid)
    st = kropf_states(p20, p80)
    with np.errstate(divide='ignore', invalid='ignore'):
        # r = (RP0 - RP1)/RP0 = 1 - f0/f1 ; p = p80 for vulnerable/resilient, p20 for dependent
        r80 = np.where(q80 > 0, 1 - p80 / q80, 0.0)
        r20 = np.where(q20 > 0, 1 - p20 / q20, -1.0)
    shift = np.zeros_like(st, dtype=bool)
    shift |= (st == 1) & (r80 > 0.10)
    shift |= (st == 2) & (r80 > 0.20)
    shift |= (st == 3) & (np.abs(r20) > 0.05)
    newly = (st == 0) & (q80 >= F_MIN)
    if newly_rule: shift |= newly
    r = np.where(st == 3, r20, r80)
    lvl = np.zeros(len(u), int)
    for c in range(3):
        lvl = np.where(shift[:, c], c + 1, lvl)
    return dict(shift=shift[inv], state=st[inv], r=r[inv], newly=newly[inv], tclevel=lvl[inv],
                unit=u, unit_inv=inv, p20=p20, p80=p80, q20=q20, q80=q80, ustate=st, ushift=shift, ulevel=lvl)

def site_regime_shift(f0, f1, state):
    """Per-site variant (sensitivity): site's own r with unit state."""
    with np.errstate(divide='ignore', invalid='ignore'):
        r = np.where(f1 > 0, 1 - f0 / f1, 0.0)
    shift = ((state == 1) & (r > 0.10)) | ((state == 2) & (r > 0.20)) | ((state == 3) & (np.abs(r) > 0.05)) | ((f0 < F_MIN) & (f1 >= F_MIN))
    lvl = np.zeros(len(f0), int)
    for c in range(3):
        lvl = np.where(shift[:, c], c + 1, lvl)
    return lvl

SCORE_NAMES = ['None', 'Low', 'Moderate', 'High', 'Very high', 'Severe']
