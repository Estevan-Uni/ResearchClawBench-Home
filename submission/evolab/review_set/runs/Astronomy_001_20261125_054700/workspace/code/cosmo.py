"""CAMB wrappers for LCDM, EDE (EarlyQuintessence, n=3) and w0wa (PPF)."""
import numpy as np, camb
from camb.dark_energy import EarlyQuintessence
C_KMS = 299792.458
OMNUH2 = 0.0006442  # m_nu = 0.06 eV, one massive state (CAMB default convention)

def make_params(p, model='lcdm', transfer=False):
    h = p['H0'] / 100.
    ombh2 = p['ombh2']
    omch2 = p['omega_m'] * h**2 - ombh2 - OMNUH2
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=p['H0'], ombh2=ombh2, omch2=omch2, mnu=0.06, omk=0,
                       tau=p.get('tau', 0.055), nnu=3.044)
    if model == 'ede':
        de = EarlyQuintessence(n=3, theta_i=2.83, use_zc=True,
                               zc=10**(-p['log10_ac']) - 1, fde_zc=p['f_EDE'],
                               frac_lambda0=1.0)
        pars.DarkEnergy = de
    elif model == 'w0wa':
        pars.set_dark_energy(w=p['w0'], wa=p['wa'], dark_energy_model='ppf')
    pars.InitPower.set_params(As=np.exp(p.get('ln10As', 3.044)) * 1e-10, ns=p.get('ns', 0.965))
    if transfer:
        pars.set_matter_power(redshifts=[0.], kmax=2.0)
        pars.WantTransfer = True
    else:
        pars.WantTransfer = False
    pars.WantCls = False
    return pars

def background(p, model='lcdm'):
    pars = make_params(p, model)
    res = camb.get_background(pars)
    der = res.get_derived_params()
    return res, der

def distances(res, der, z):
    z = np.atleast_1d(z).astype(float)
    DM = res.comoving_radial_distance(z)
    H = res.hubble_parameter(z)
    DH = C_KMS / H
    DV = (z * DM**2 * DH)**(1/3.)
    rd = der['rdrag']
    return dict(DM=DM, DH=DH, DV=DV, rd=rd, DVrd=DV / rd, DMrd=DM / rd, DHrd=DH / rd,
                FAP=DM / DH, DL=(1 + z) * DM, mu=5 * np.log10((1 + z) * DM) + 25)

def sigma8(p, model='lcdm'):
    pars = make_params(p, model, transfer=True)
    res = camb.get_results(pars)
    return float(res.get_sigma8_0()), res.get_derived_params()
