"""1-D mass scans: survival probability (headline), comparison approximations, Bayes factor, systematics."""
import numpy as np, json, os, time, sys
import physics as ph, stats as st
from data_overview import load, ROOT

FID = dict(rate_model="calibrated")
res = {}
t0 = time.time()
for name in ["M33 X-7", "IRAS 09149-6206"]:
    M, a = load(name)
    lo, hi = st.MU_RANGE[name]
    mus = np.logspace(np.log10(lo), np.log10(hi), 241)
    A = st.AGES[name]
    R = dict(mus=mus)
    R["fid"] = st.p_surv_curve(M, a, mus, A["fid"], **FID)
    R["alt"] = st.p_surv_curve(M, a, mus, A["alt"], **FID)
    R["marg"] = st.p_surv_curve(M, a, mus, A["marg"], **FID)
    R["fid_detw"] = st.p_surv_curve(M, a, mus, A["fid"], rate_model="detweiler")
    R["marg_detw"] = st.p_surv_curve(M, a, mus, A["marg"], rate_model="detweiler")
    # comparisons at fiducial timescale
    R["shuffled"] = np.array([st.p_surv_curve(*st.shuffled(M, a, s), mus, A["fid"], **FID) for s in range(5)])
    R["gauss_full"] = st.p_surv_curve(*st.gaussian_draws(M, a, diag=False), mus, A["fid"], **FID)
    R["gauss_diag"] = st.p_surv_curve(*st.gaussian_draws(M, a, diag=True), mus, A["fid"], **FID)
    R["point_median"] = st.point_rule(M, a, mus, A["fid"], **FID)
    R["box1"] = st.box_rule(M, a, mus, A["fid"], nsig=1, **FID)
    R["box2"] = st.box_rule(M, a, mus, A["fid"], nsig=2, **FID)
    # same comparisons age-marginalised
    R["shuffled_marg"] = np.array([st.p_surv_curve(*st.shuffled(M, a, s), mus, A["marg"], **FID) for s in range(3)])
    # systematics
    R["sys_l1"] = st.p_surv_curve(M, a, mus, A["fid"], lmax=1, **FID)
    R["sys_x0.1"] = st.p_surv_curve(M, a, mus, A["fid"], rate_scale=0.1, **FID)
    R["sys_x10"] = st.p_surv_curve(M, a, mus, A["fid"], rate_scale=10, **FID)
    R["sys_tmin"] = st.p_surv_curve(M, a, mus, A["marg"][0], **FID)
    print(name, "P_surv done", time.time() - t0, flush=True)
    # Bayes factor (extension)
    kde = st.KDE2(M, a)
    lx = np.log(M)
    lgrid = np.linspace(lx.min() - 0.3 * lx.std() - 0.05, lx.max() + 0.3 * lx.std() + 0.05, 300)
    for key, t in [("fid", A["fid"]), ("alt", A["alt"])]:
        B = np.array([st.bayes_factor(M, a, mu, t, kde, lgrid, **FID)[0] for mu in mus])
        R["B_" + key] = B
    R["B_marg"] = np.mean([[st.bayes_factor(M, a, mu, t, kde, lgrid, **FID)[0] for mu in mus] for t in A["marg"][::2]], 0)
    for bw in (0.5, 2.0):
        k2 = st.KDE2(M, a, bw_scale=bw)
        R[f"B_fid_bw{bw}"] = np.array([st.bayes_factor(M, a, mu, A["fid"], k2, lgrid, **FID)[0] for mu in mus])
    print(name, "B done", time.time() - t0, flush=True)
    res[name] = R

np.save(os.path.join(ROOT, "outputs/scan1d.npy"), res, allow_pickle=True)

# summary table of excluded intervals
lines = []
for name, R in res.items():
    mus = R["mus"]
    for key in [k for k in R if k != "mus"]:
        v = np.atleast_2d(R[key])
        for i, row in enumerate(v):
            lab = key if v.shape[0] == 1 else f"{key}[seed{i}]"
            thr = 0.5 if key.startswith("point") or key.startswith("box") else 0.05
            iv = st.intervals(mus, row < thr) if not key.startswith("box") else st.intervals(mus, row <= 0.0)
            lines.append(dict(bh=name, curve=lab, min=float(row.min()), mu_at_min=float(mus[np.argmin(row)]),
                              excluded=st.fmt_int(iv), intervals=[list(x) for x in iv]))
json.dump(lines, open(os.path.join(ROOT, "outputs/scan1d_intervals.json"), "w"), indent=1)
for L in lines:
    print(f"{L['bh']:16s} {L['curve']:22s} min={L['min']:.3g} @ {L['mu_at_min']:.2e}  excl: {L['excluded']}")
