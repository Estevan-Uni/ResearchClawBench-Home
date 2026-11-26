"""Electrochemical-aging-thermal (ECAT) coupled model built on PyBaMM.

Electrochemistry : SPMe (single particle model with electrolyte), PyBaMM 26.x
Thermal          : lumped (x-averaged) energy balance, ohmic + reaction + reversible (entropic) heat,
                   Newton cooling with a total heat-transfer coefficient h; Arrhenius coupling of the
                   kinetics/diffusion via the Ai2020 parameter functions.
Aging            : state-of-health parameters enter as (i) loss of lithium inventory (LLI: scaling of the
                   initial negative-electrode stoichiometry), (ii) loss of positive active material
                   (LAM_p: scaling of the positive active-material volume fraction), and (iii) a film /
                   contact resistance R_f (SEI-like ohmic growth).
Base parameter set: Ai2020 (LiCoO2 / graphite, 2.28 Ah) -- an LCO/graphite cell close to the 2 Ah 18650
                   cells of the NASA PCoE data set.
All identified parameters are passed as PyBaMM InputParameters so that the model is discretised once and
re-solved for every LHS sample (particle radii use a symbolic radial mesh).
"""
import numpy as np, pybamm, time, warnings
warnings.filterwarnings('ignore')
pybamm.set_logging_level('ERROR')

BASE = 'Ai2020'
_p0 = pybamm.ParameterValues(BASE)
NOMINAL = dict(R_n=_p0['Negative particle radius [m]'], R_p=_p0['Positive particle radius [m]'],
               k_n=1.0, h=_p0['Total heat transfer coefficient [W.m-2.K-1]'],
               LLI=1.0, LAM_p=1.0, R_f=0.0, SOCp=1.0)
CAP_NOM = _p0['Nominal cell capacity [A.h]']

# Parameter names, bounds, scaling for the core identification problem (paper Step 1-3)
CORE = ['R_n', 'R_p', 'k_n', 'h']
BOUNDS = dict(R_n=(2.5e-6, 10e-6), R_p=(1.5e-6, 6e-6), k_n=(0.2, 5.0), h=(5.0, 40.0),
              LLI=(0.85, 1.02), LAM_p=(0.4, 1.0), R_f=(0.0, 0.15), SOCp=(0.8, 1.2))
# wider box for real-data identification (selected with environment variable ECAT_BOUNDS=real); the synthetic
# benchmark (Steps 1-3) uses the default box above
import os as _os
if _os.environ.get('ECAT_BOUNDS', '') == 'real':
    BOUNDS.update(R_n=(2.5e-6, 20e-6), k_n=(0.05, 5.0), LLI=(0.6, 1.02), SOCp=(0.8, 1.2))
LOGSCALE = dict(R_n=True, R_p=True, k_n=True, h=True, LLI=False, LAM_p=False, R_f=False, SOCp=False)
UNITS = dict(SOCp='-', R_n='m', R_p='m', k_n='x j0_ref', h='W/m2/K', LLI='-', LAM_p='-', R_f='Ohm')

# normalised-time grid on which V and T are represented (denser at both ends)
TAU = np.unique(np.concatenate([np.linspace(0, 0.05, 11), np.linspace(0.05, 0.95, 61), np.linspace(0.95, 1.0, 11)]))


def unit_to_phys(u, names):
    """map unit-cube samples (n, d) to physical values (log scaling where flagged)"""
    u = np.atleast_2d(u); x = np.empty_like(u, dtype=float)
    for j, n in enumerate(names):
        lo, hi = BOUNDS[n]
        x[:, j] = np.exp(np.log(lo) + u[:, j] * (np.log(hi) - np.log(lo))) if LOGSCALE[n] else lo + u[:, j] * (hi - lo)
    return x


def phys_to_unit(x, names):
    x = np.atleast_2d(x); u = np.empty_like(x, dtype=float)
    for j, n in enumerate(names):
        lo, hi = BOUNDS[n]
        u[:, j] = (np.log(x[:, j]) - np.log(lo)) / (np.log(hi) - np.log(lo)) if LOGSCALE[n] else (x[:, j] - lo) / (hi - lo)
    return u


class ECAT:
    def __init__(self, I_app=None, T_amb=298.15, V_cut=3.0, var_pts=None, rtol=1e-6, atol=1e-8,
                 fixed=None, model='SPMe', dt_out=2.0, current=None):
        self.dt_out = dt_out
        p = pybamm.ParameterValues(BASE)
        self.I_app = CAP_NOM if I_app is None else I_app   # 1C by default
        self.V_cut = V_cut
        j0n = p['Negative electrode exchange-current density [A.m-2]']
        c_n0 = p['Initial concentration in negative electrode [mol.m-3]']
        eps_p0 = p['Positive electrode active material volume fraction']
        c_p0 = p['Initial concentration in positive electrode [mol.m-3]']
        p.update({
            'Current function [A]': self.I_app if current is None else
                pybamm.Interpolant(current[0], current[1], pybamm.t, interpolator='linear'),
            'Ambient temperature [K]': T_amb, 'Initial temperature [K]': T_amb,
            'Lower voltage cut-off [V]': V_cut,
            'Upper voltage cut-off [V]': 4.2 if current is None else 4.5,
            'Negative particle radius [m]': pybamm.InputParameter('R_n'),
            'Positive particle radius [m]': pybamm.InputParameter('R_p'),
            'Total heat transfer coefficient [W.m-2.K-1]': pybamm.InputParameter('h'),
            'Negative electrode exchange-current density [A.m-2]':
                lambda ce, cs, cmax, T: pybamm.InputParameter('k_n') * j0n(ce, cs, cmax, T),
            'Initial concentration in negative electrode [mol.m-3]': c_n0 * pybamm.InputParameter('LLI'),
            'Positive electrode active material volume fraction': eps_p0 * pybamm.InputParameter('LAM_p'),
            'Contact resistance [Ohm]': pybamm.InputParameter('R_f'),
            # initial positive-electrode stoichiometry factor (electrode balancing / SOC window)
            'Initial concentration in positive electrode [mol.m-3]': c_p0 * pybamm.InputParameter('SOCp'),
        }, check_already_exists=False)
        self.fixed = dict(NOMINAL); self.fixed.update(fixed or {})
        opts = {'thermal': 'lumped', 'contact resistance': 'true'}
        m = pybamm.lithium_ion.SPMe(options=opts) if model == 'SPMe' else pybamm.lithium_ion.DFN(options=opts)
        sm = dict(m.default_submesh_types)
        sm['negative particle'] = pybamm.SymbolicUniform1DSubMesh
        sm['positive particle'] = pybamm.SymbolicUniform1DSubMesh
        vp = dict(m.default_var_pts); vp.update(var_pts or {})
        solver = pybamm.IDAKLUSolver(rtol=rtol, atol=atol)
        self.sim = pybamm.Simulation(m, parameter_values=p, submesh_types=sm, var_pts=vp, solver=solver)
        self.t_max = 1.6 * 3600 * CAP_NOM / self.I_app

    def run(self, x, names, t_eval=None):
        """simulate one parameter vector; returns dict(t,V,T,Q) or None on failure"""
        inp = dict(self.fixed); inp.update({n: float(v) for n, v in zip(names, x)})
        if t_eval is None:
            t_eval = np.arange(0, self.t_max, self.dt_out)
        try:
            sol = self.sim.solve([0, t_eval[-1]], t_interp=t_eval, inputs=inp)
            t = sol['Time [s]'].entries; V = sol['Voltage [V]'].entries
            T = sol['Volume-averaged cell temperature [K]'].entries
            # keep a strictly increasing time axis ending at the termination event (t_interp points that lie
            # beyond the event time are discarded -- they otherwise corrupt interpolation at tau = 1)
            te = sol.t[-1]
            keep = np.concatenate([np.diff(t) > 0, [True]]) & (t <= te + 1e-9)
            t, V, T = t[keep], V[keep], T[keep]
            o = np.argsort(t, kind='stable'); t, V, T = t[o], V[o], T[o]
            # if the cut-off event is reached through a steep collapse (positive surface saturation) the solver
            # reports a voltage below V_cut at the event; the physical end of discharge is the cut-off itself
            if t[-1] < self.t_max - 1 and V[-1] < self.V_cut:
                V = V.copy(); V[-1] = self.V_cut
            if len(t) < 20 or not np.all(np.isfinite(V)):
                return None
            return dict(t=t, V=V, T=T, Q=t[-1] * self.I_app / 3600.0)
        except Exception as e:
            return None


def features(res, tau=TAU):
    """fixed-length representation: V(tau), T(tau)-T0 on normalised time grid, plus discharge time"""
    t, V, T = res['t'], res['V'], res['T']
    tt = tau * t[-1]
    return np.interp(tt, t, V), np.interp(tt, t, T) - T[0], t[-1]
