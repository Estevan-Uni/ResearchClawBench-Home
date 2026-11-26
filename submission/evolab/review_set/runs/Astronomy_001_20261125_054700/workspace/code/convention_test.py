"""Sensitivity of model chi^2 to the sign convention of the digitised Δμ and ΔF_AP points."""
import os, json, numpy as np
from common import load_data, MODELS, OUT
d = load_data()
A = np.loadtxt(os.path.join(OUT, 'distance_curves.txt')); hdr = open(os.path.join(OUT, 'distance_curves.txt')).readline()[2:].split()
z = A[:, 0]; col = {h: A[:, i] for i, h in enumerate(hdr)}
out = {}
for key, k in [('sne_mu_points', 'mu'), ('desi_fap_points', 'fap'), ('desi_dvrd_points', 'dvrd')]:
    pt = np.array(d[key])
    for sgn in [+1, -1]:
        for m in MODELS:
            mod = np.interp(pt[:, 0], z, col[f'{m}_{k}']); y = sgn * pt[:, 1]; w = 1 / pt[:, 2]**2
            c = np.sum((y - mod)**2 * w); off = np.sum(w * (y - mod)) / w.sum(); co = np.sum((y - mod - off)**2 * w)
            out[f'{k}_sign{sgn:+d}_{m}'] = dict(chi2=float(c), chi2_offmarg=float(co))
for kk, v in out.items(): print(f'{kk:22s} chi2={v["chi2"]:6.2f}  offset-marg={v["chi2_offmarg"]:6.2f}')
json.dump(out, open(os.path.join(OUT, 'convention_test.json'), 'w'), indent=1)
