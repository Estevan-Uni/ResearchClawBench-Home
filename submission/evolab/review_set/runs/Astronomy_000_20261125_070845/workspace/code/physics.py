"""Superradiance physics for scalar ultralight bosons around Kerr BHs.

Conventions: G=c=hbar=1 inside, M in Msun, mu in eV, ages in yr, f in GeV.
alpha = G M mu / (hbar c) = 7.4828e9 * M[Msun] * mu[eV].
"""
import numpy as np
from math import factorial

ALPHA_PER_MSUN_EV = 7.4828e9       # alpha per (Msun * eV)
TSUN = 4.925491e-6                 # G Msun / c^3 [s]
YR = 3.15576e7                     # s
MSUN_GEV = 1.11572e57              # Msun in GeV
MPL = 1.22091e19                   # non-reduced Planck mass [GeV]
MPL_RED = MPL / np.sqrt(8 * np.pi)
LMAX = 6

# Dolan (2007) numerical peak of M*omega_I for l=m=1 (n=2) at a=0.99: ~1.5e-7 near M mu ~ 0.42
DOLAN_PEAK = 1.5e-7


def C_nl(n, l):
    return (2 ** (4 * l + 1) * factorial(n + l) / (n ** (2 * l + 4) * factorial(n - l - 1))
            * (factorial(l) / (factorial(2 * l) * factorial(2 * l + 1))) ** 2)


def omega_R(alpha, n):
    """M*omega_R, hydrogenic spectrum to O(alpha^3)."""
    return alpha * (1.0 - alpha ** 2 / (2.0 * n ** 2))


def a_sat(alpha, l, n=None):
    """Spin at which omega = m Omega_H (m=l)."""
    n = l + 1 if n is None else n
    x = omega_R(alpha, n) / l
    return np.where(x < 0.5, 4 * x / (1 + 4 * x ** 2), 1.0)


def MomegaI(alpha, a, l, n=None, guard=True):
    """Detweiler / Brito-Cardoso-Pani small-alpha growth rate of the field amplitude, M*omega_I.
    Positive => superradiant growth."""
    n = l + 1 if n is None else n
    m = l
    a = np.asarray(a, float)
    rp = 1.0 + np.sqrt(np.clip(1 - a ** 2, 0, None))
    OmH = a / (2 * rp)
    w = omega_R(alpha, n)
    g = np.ones_like(a * alpha)
    for k in range(1, l + 1):
        g = g * (k ** 2 * (1 - a ** 2) + (a * m - 2 * rp * w) ** 2)
    wI = 2 * rp * C_nl(n, l) * g * (m * OmH - w) * alpha ** (4 * l + 5)
    if guard:
        wI = np.where(alpha > 0.6 * m, 0.0, wI)
    return wI


# Calibration of the analytic l=1 rate against the numerical (Dolan 2007) peak at a=0.99
def _calib():
    al = np.linspace(0.05, 0.5, 2000)
    v = MomegaI(al, 0.99, 1)
    return DOLAN_PEAK / v.max(), al[np.argmax(v)], v.max()
RATE_CALIB, ALPHA_PEAK_ANALYTIC, PEAK_ANALYTIC = _calib()


def calib_factor(alpha, l):
    """Smooth relativistic enhancement: ->1 for alpha<<l, -> RATE_CALIB at the l=1 analytic peak.
    K_l(alpha) = RATE_CALIB ** ((alpha / (ALPHA_PEAK_ANALYTIC * l))**2)."""
    return RATE_CALIB ** (np.clip(alpha / (ALPHA_PEAK_ANALYTIC * l), 0, 3) ** 2)


def Gamma_sr(alpha, a, M, l, rate_model="detweiler", rate_scale=1.0):
    """Occupation-number growth rate 2*omega_I in 1/yr (M in Msun)."""
    wI = MomegaI(alpha, a, l)
    if rate_model == "calibrated":
        wI = wI * calib_factor(alpha, l)
    return rate_scale * 2.0 * wI / (M * TSUN) * YR


def N_need(M, da, m):
    return (M * MSUN_GEV / MPL) ** 2 * np.clip(da, 0, None) / m


def N_sat(alpha, a, M, l, f, model, c=1.0, rate_model="detweiler"):
    """Occupation number at which self-interactions halt exponential growth.
    model 'bosenova': N_BN = c0 n^4/alpha^3 (f/MPL)^2 (M/MPL)^2 (Arvanitaki+2015 normalisation, c0=c)
    model 'mixing'  : quasi-equilibrium of the 211-322 self-interaction network (Baryakhtar+2021
                      structure; coefficients kappa_a = 4.3e-7 alpha^7, kappa_b = 1.1e-8 alpha^8 in units
                      (MPL/f)^4 (MPL/M)^4 / r_g), N_eq = (2/sqrt3) sqrt(G kb)/ka, times bracket factor c.
                      Applied to the l=1 cloud only; l>=2 levels use the bosenova cap with c0=5."""
    n = l + 1
    MM = (M * MSUN_GEV / MPL) ** 2
    FF = (f / MPL) ** 2
    if model == "bosenova":
        return c * n ** 4 / alpha ** 3 * FF * MM
    elif model == "mixing":
        if l >= 2:
            # equilibrium network only derived for the 211-322 system; use bosenova cap (c0=5) for l>=2
            return 5.0 * n ** 4 / alpha ** 3 * FF * MM
        G = 2.0 * MomegaI(alpha, a, l)
        if rate_model == "calibrated":
            G = G * calib_factor(alpha, l)
        G = np.clip(G, 1e-300, None)
        ka = 4.3e-7 * alpha ** 7
        kb = 1.1e-8 * alpha ** 8
        return c * (2 / np.sqrt(3)) * np.sqrt(G * kb) / ka * FF * MM
    raise ValueError(model)


def t_spindown(M, a, mu, l, f=None, model=None, c=1.0, rate_model="detweiler", rate_scale=1.0):
    """Time [yr] for level l=m (n=l+1) to spin BH from a to a_sat; inf if not superradiant."""
    alpha = ALPHA_PER_MSUN_EV * M * mu
    asat = a_sat(alpha, l)
    G = Gamma_sr(alpha, a, M, l, rate_model, rate_scale)
    Nn = N_need(M, a - asat, l)
    ok = (a > asat) & (G > 0) & (Nn > 1)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        if f is None or model is None:
            t = np.log(np.clip(Nn, 1.0, None)) / G
        else:
            Ns = N_sat(alpha, a, M, l, f, model, c, rate_model)
            Ns = np.clip(Ns, 1.0, None)
            Ne = np.minimum(Nn, Ns)
            t = (np.log(Ne) + np.clip(Nn - Ns, 0, None) / Ns) / G
    return np.where(ok, t, np.inf)


def unstable(M, a, mu, tage, lmax=LMAX, **kw):
    """Boolean: (M,a) would have been spun down within tage by any level l<=lmax."""
    out = np.zeros(np.broadcast(M, a, mu).shape, bool)
    for l in range(1, lmax + 1):
        out |= t_spindown(M, a, mu, l, **kw) < tage
    return out


def regge_spin(M, a0, mu, tage, lmax=LMAX, **kw):
    """Evolve initial spin a0 along successive levels (fastest first): spin after tage."""
    a = np.array(a0, float, copy=True)
    M = np.broadcast_to(M, a.shape)
    t_used = np.zeros_like(a)
    for l in range(1, lmax + 1):
        alpha = ALPHA_PER_MSUN_EV * M * mu
        ts = t_spindown(M, a, mu, l, **kw)
        done = (t_used + ts) < tage
        a = np.where(done, np.minimum(a, a_sat(alpha, l)), a)
        t_used = np.where(done, t_used + ts, t_used)
    return a


def f_qcd(mu):
    """QCD axion decay constant [GeV] for mass mu [eV]: m_a = 5.7 ueV (1e12 GeV/f)."""
    return 5.7e6 / mu
