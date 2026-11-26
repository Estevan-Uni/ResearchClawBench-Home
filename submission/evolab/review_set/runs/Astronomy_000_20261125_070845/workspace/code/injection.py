"""Injection / calibration study: how often does each decision rule exclude the TRUE boson mass?"""
import numpy as np, os, json, time, sys
import physics as ph, stats as st
from data_overview import load, ROOT

FID = dict(rate_model="calibrated")
name = sys.argv[1] if len(sys.argv) > 1 else "M33 X-7"
NMOCK = int(sys.argv[2]) if len(sys.argv) > 2 else 300
M, a = load(name)
X = np.c_[np.log(M), a]; C = np.cov(X.T)
tfid = st.AGES[name]["fid"]
if name == "M33 X-7":
    mus = np.logspace(-13.3, -10.7, 11)
else:
    mus = np.logspace(-20, -17.5, 9)
rng = np.random.default_rng(42)
lx = np.log(M)
lgrid = np.linspace(lx.min() - 0.5, lx.max() + 0.5, 300); dl = lgrid[1] - lgrid[0]
out = []
t0 = time.time()
for mu in mus:
    W, AS = st.pileup_weights(np.exp(lgrid), mu, tfid, **FID)
    rec = {k: [] for k in ["ps", "B", "pt", "box", "ps_null", "B_null"]}
    for k in range(NMOCK):
        for hyp in ("true", "null"):
            Mt = M[rng.integers(len(M))]; a0 = rng.uniform(0, 1)
            at = ph.regge_spin(np.array([Mt]), np.array([a0]), mu, tfid, **FID)[0] if hyp == "true" else a0
            cen = np.array([np.log(Mt), at]) + rng.multivariate_normal([0, 0], C)
            d = rng.multivariate_normal(cen, C, size=1500)
            d = d[(d[:, 1] > 0) & (d[:, 1] < 1)][:600]
            if len(d) < 50:
                d = np.c_[np.full(50, cen[0]), np.full(50, np.clip(cen[1], 0.001, 0.999))]
            Ms, as_ = np.exp(d[:, 0]), d[:, 1]
            ps = st.p_surv(Ms, as_, mu, tfid, **FID)
            dens = st.GaussDens(cen, C)
            # truncation normalisation of the Gaussian likelihood-posterior to 0<a<1
            from scipy.stats import norm
            Z = norm.cdf((1 - cen[1]) / np.sqrt(C[1, 1])) - norm.cdf((0 - cen[1]) / np.sqrt(C[1, 1]))
            pile = sum(np.sum(W[l] * dens(lgrid, AS[l])) * dl for l in range(W.shape[0])) / max(Z, 1e-6)
            if hyp == "true":
                rec["ps"].append(ps); rec["B"].append(ps + pile)
                rec["pt"].append(float(ph.unstable(np.array([np.exp(cen[0])]), np.array([min(cen[1], 0.999)]), mu, tfid, **FID)[0]))
                bx = st.box_rule(np.exp(d[:, 0]), d[:, 1], [mu], tfid, nsig=1, ngrid=5, **FID)[0]
                rec["box"].append(bx <= 0)
            else:
                rec["ps_null"].append(ps); rec["B_null"].append(ps + pile)
    r = {k: np.array(v) for k, v in rec.items()}
    row = dict(mu=float(mu), fe_psurv=float(np.mean(r["ps"] < 0.05)), fe_B=float(np.mean(r["B"] < 0.05)),
               fe_point=float(np.mean(r["pt"])), fe_box=float(np.mean(r["box"])),
               mean_invB=float(np.mean(1 / np.clip(r["B"], 1e-6, None))),
               power_psurv=float(np.mean(r["ps_null"] < 0.05)), power_B=float(np.mean(r["B_null"] < 0.05)),
               median_ps_true=float(np.median(r["ps"])))
    out.append(row); print(json.dumps(row), f"{time.time()-t0:.0f}s", flush=True)
tag = "M33" if name.startswith("M33") else "IRAS"
json.dump(out, open(os.path.join(ROOT, f"outputs/injection_{tag}.json"), "w"), indent=1)
