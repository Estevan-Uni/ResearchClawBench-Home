"""KDE normalisation check (integral over ln M and 0<a<1 should be ~1) and limit checks."""
import numpy as np, json, os, stats as st, physics as ph
from data_overview import load, ROOT
out = {}
for name in ["M33 X-7", "IRAS 09149-6206"]:
    M, a = load(name); k = st.KDE2(M, a)
    lx = np.log(M)
    gx = np.linspace(lx.min() - 1, lx.max() + 1, 160); gy = np.linspace(0, 1, 160)
    X, Y = np.meshgrid(gx, gy)
    Z = k(X.ravel(), Y.ravel()).reshape(X.shape)
    I = np.trapezoid(np.trapezoid(Z, gy, axis=0), gx)
    lo, hi = st.MU_RANGE[name]
    far = [lo / 10, hi * 10]
    out[name] = dict(kde_norm=float(I), psurv_far=[float(st.p_surv(M, a, m, 1e10, rate_model="calibrated")) for m in far])
print(out); json.dump(out, open(os.path.join(ROOT, "outputs/validation_kde.json"), "w"), indent=1)
