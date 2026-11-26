"""Shared data utilities: loading ERA5 years, normalisation, forcings, verification-variable extraction."""
import numpy as np, os, json

ROOT = '/workspace/outputs'
ERA = f'{ROOT}/era5'
LEVELS = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
NAMES = [f'{v}{l}' for v in 'ZTUVR' for l in LEVELS] + ['T2M', 'U10', 'V10', 'MSL', 'TP']
NCH = 70
TP = 69
TP_EPS = 1e-4          # m per 6 h (0.1 mm) for log1p transform
VER = ['MSL', 'T2M', 'U10', 'V10', 'Z500', 'T500', 'U500', 'V500', 'Z850', 'T850', 'TP6']
VER_IDX = [NAMES.index(n) for n in ['MSL', 'T2M', 'U10', 'V10', 'Z500', 'T500', 'U500', 'V500', 'Z850', 'T850', 'TP']]
LAT = np.linspace(-87.1875, 87.1875, 32)
LON = np.arange(64) * 5.625
WLAT = np.cos(np.deg2rad(LAT)); WLAT = WLAT / WLAT.mean()


def raw_stats():
    st = np.load(f'{ERA}/stats.npz'); return st['mean'].astype(np.float64), st['std'].astype(np.float64)


def load_year_physical(y):
    """Return physical float32 array (T,70,32,64) and dates (datetime64[h])."""
    X = np.load(f'{ERA}/{y}.npy')
    d = np.load(f'{ERA}/{y}_dates.npy').astype('datetime64[h]')
    if X.dtype == np.float16:
        mu, sd = raw_stats()
        X = X.astype(np.float32) * sd[None, :, None, None].astype(np.float32) + mu[None, :, None, None].astype(np.float32)
    X[:, TP] = np.maximum(X[:, TP], 0)
    return X, d


class Normaliser:
    """z-score per channel; TP is transformed with log1p(TP/eps) before the z-score."""
    def __init__(self, path=f'{ROOT}/norm.npz'):
        if not os.path.exists(path):
            X, _ = load_year_physical(2017)
            X = X.astype(np.float64); X[:, TP] = np.log1p(X[:, TP] / TP_EPS)
            mu = X.mean((0, 2, 3)); sd = X.std((0, 2, 3))
            Z = (X - mu[None, :, None, None]) / sd[None, :, None, None]
            dsd = np.sqrt(((Z[1:] - Z[:-1]) ** 2).mean((0, 2, 3)))
            np.savez(path, mean=mu, std=sd, dstd=dsd)
        f = np.load(path)
        self.mu, self.sd, self.dsd = f['mean'].astype(np.float32), f['std'].astype(np.float32), f['dstd'].astype(np.float32)

    def norm(self, X):
        X = X.astype(np.float32).copy()
        X[:, TP] = np.log1p(np.maximum(X[:, TP], 0) / TP_EPS)
        return (X - self.mu[None, :, None, None]) / self.sd[None, :, None, None]

    def denorm(self, Z):
        """Z: (..., 70, H, W) normalised -> physical."""
        sh = (1,) * (Z.ndim - 3) + (NCH, 1, 1)
        X = Z * self.sd.reshape(sh) + self.mu.reshape(sh)
        X[..., TP, :, :] = TP_EPS * np.expm1(np.clip(X[..., TP, :, :], 0, None))
        return X


def coszen(dates):
    """Cosine of the solar zenith angle (clipped at 0 -> TOA insolation proxy) at given UTC times. (T,32,64)."""
    d = dates.astype('datetime64[h]')
    doy = (d - d.astype('datetime64[Y]')).astype('timedelta64[h]').astype(np.float64) / 24.0
    hour = (d.astype(np.int64) % 24).astype(np.float64)
    decl = -23.44 * np.cos(np.deg2rad(360.0 / 365.0 * (doy + 10)))
    lat = np.deg2rad(LAT)[None, :, None]; dec = np.deg2rad(decl)[:, None, None]
    ha = np.deg2rad((hour[:, None, None] * 15.0 + LON[None, None, :]) - 180.0)
    cz = np.sin(lat) * np.sin(dec) + np.cos(lat) * np.cos(dec) * np.cos(ha)
    return np.clip(cz, 0, None).astype(np.float32)


def static_fields():
    st = np.load(f'{ROOT}/bench/static.npz')
    lsm = st['land_sea_mask'].astype(np.float32)
    oro = st['geopotential_at_surface'].astype(np.float32); oro = (oro - oro.mean()) / oro.std()
    lat = np.repeat(np.sin(np.deg2rad(LAT))[:, None], 64, 1).astype(np.float32)
    return np.stack([lsm, oro, lat])          # (3,32,64)


def forcing(dates_t):
    """Forcing channels for a step whose *current* time is dates_t (target = t+6h): (T,7,32,64)."""
    dates_t = np.asarray(dates_t).astype('datetime64[h]')
    st = static_fields()
    c0 = coszen(dates_t); c1 = coszen(dates_t + np.timedelta64(6, 'h'))
    doy = (dates_t - dates_t.astype('datetime64[Y]')).astype('timedelta64[h]').astype(np.float64) / (24 * 365.25)
    s = np.sin(2 * np.pi * doy).astype(np.float32)[:, None, None] * np.ones((1, 32, 64), np.float32)
    c = np.cos(2 * np.pi * doy).astype(np.float32)[:, None, None] * np.ones((1, 32, 64), np.float32)
    return np.concatenate([np.broadcast_to(st, (len(dates_t),) + st.shape), c0[:, None], c1[:, None], s[:, None], c[:, None]], 1)
