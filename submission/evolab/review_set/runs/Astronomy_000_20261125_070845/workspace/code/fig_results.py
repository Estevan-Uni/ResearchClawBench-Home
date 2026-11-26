"""Figures for 1-D survival curves, comparisons, Bayes factor, systematics."""
import numpy as np, os, json
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import stats as st
from data_overview import ROOT

IMG = os.path.join(ROOT, "report/images")
res = np.load(os.path.join(ROOT, "outputs/scan1d.npy"), allow_pickle=True).item()


def shade(ax, mus, curve, color, alpha=0.15):
    for lo, hi in st.intervals(mus, curve < 0.05):
        ax.axvspan(lo, hi, color=color, alpha=alpha, lw=0)


# ---- Fig 4: headline P_surv for both BHs
fig, axs = plt.subplots(1, 2, figsize=(13, 4.6))
for ax, name in zip(axs, res):
    R = res[name]; mus = R["mus"]
    tl = "3e6" if name == "M33 X-7" else "1e10"
    ax.plot(mus, R["fid"], "C0", lw=2.2, label=r"fiducial: $t=t_{\rm Sal}=4.5\times10^7$ yr")
    ax.plot(mus, R["alt"], "C1", lw=1.6, label=f"t = {tl} yr")
    ax.plot(mus, R["marg"], "C2", lw=1.6, label="age-marginalised (log-uniform)")
    ax.plot(mus, R["fid_detw"], "C0", ls=":", lw=1.4, label=r"fiducial $t$, uncalibrated analytic rates")
    shade(ax, mus, R["fid"], "C0")
    ax.axhline(0.05, color="r", ls="--", lw=1, label="5% threshold")
    ax.set_xscale("log"); ax.set_ylim(-0.02, 1.02); ax.set_xlim(mus[0], mus[-1])
    ax.set_xlabel(r"$\mu$ [eV]"); ax.set_ylabel(r"$P_{\rm surv}(\mu)$ = P(BH not spun down | data)")
    ax.set_title(name); ax.legend(fontsize=7.5, loc="center" if name == "M33 X-7" else "lower right")
plt.tight_layout(); plt.savefig(os.path.join(IMG, "fig4_psurv.png"), dpi=140); plt.close()

# ---- Fig 5: correlated vs uncorrelated / Gaussian / point rules
fig, axs = plt.subplots(1, 2, figsize=(13, 4.6))
for ax, name in zip(axs, res):
    R = res[name]; mus = R["mus"]
    ax.plot(mus, R["fid"], "k", lw=2.4, label="full correlated posterior (samples)")
    for i, s in enumerate(R["shuffled"]):
        ax.plot(mus, s, "C3", lw=1, alpha=0.7, label="uncorrelated (a* permuted, 5 seeds)" if i == 0 else None)
    ax.plot(mus, R["gauss_full"], "C0", ls="--", lw=1.5, label="Gaussian, full covariance")
    ax.plot(mus, R["gauss_diag"], "C1", ls="-.", lw=1.5, label="Gaussian, diagonal covariance")
    ax.plot(mus, R["box1"], "C2", ls=":", lw=1.5, label=r"$1\sigma$ box: stable fraction")
    ax.plot(mus, R["point_median"], "C4", lw=1, alpha=0.6, drawstyle="steps-mid", label="median point estimate (0/1)")
    ax.axhline(0.05, color="r", ls="--", lw=1)
    ax.set_xscale("log"); ax.set_ylim(-0.02, 1.02); ax.set_xlim(mus[0], mus[-1])
    ax.set_xlabel(r"$\mu$ [eV]"); ax.set_ylabel(r"$P_{\rm surv}$"); ax.set_title(f"{name}: t = 4.5e7 yr")
    if name == "M33 X-7":
        ax.legend(fontsize=7, loc="lower left", bbox_to_anchor=(0.2, 0.08))
    else:
        ax.legend(fontsize=7, loc="lower right")
    if name == "M33 X-7":
        ins = ax.inset_axes([0.27, 0.55, 0.33, 0.33])
        for s in R["shuffled"]:
            ins.plot(mus, s, "C3", lw=0.8, alpha=0.7)
        ins.plot(mus, R["fid"], "k", lw=1.6); ins.plot(mus, R["gauss_diag"], "C1", ls="-.")
        ins.axhline(0.05, color="r", ls="--", lw=0.8)
        ins.set_xscale("log"); ins.set_xlim(6e-12, 1.6e-11); ins.set_ylim(-0.01, 0.4); ins.tick_params(labelsize=6)
        ins.set_title("upper edge", fontsize=7)
plt.tight_layout(); plt.savefig(os.path.join(IMG, "fig5_comparisons.png"), dpi=140); plt.close()

# ---- Fig 6: Bayes factor (extension)
fig, axs = plt.subplots(1, 2, figsize=(13, 4.6))
for ax, name in zip(axs, res):
    R = res[name]; mus = R["mus"]
    ax.plot(mus, R["B_fid"], "C0", lw=2, label=r"$\mathcal{B}$, $t=4.5\times10^7$ yr")
    ax.plot(mus, R["B_alt"], "C1", lw=1.4, label=r"$\mathcal{B}$, alt. $t$")
    ax.plot(mus, R["B_marg"], "C2", lw=1.4, label=r"$\mathcal{B}$, age-marginalised")
    ax.plot(mus, np.clip(R["fid"], 1e-4, None), "k:", lw=1.4, label=r"$P_{\rm surv}$ (fiducial) for reference")
    ax.plot(mus, R["B_fid_bw0.5"], "C0", lw=0.8, ls="--", label="KDE bandwidth ×0.5 / ×2")
    ax.plot(mus, R["B_fid_bw2.0"], "C0", lw=0.8, ls="--")
    ax.axhline(0.05, color="r", ls="--", lw=1)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_ylim(1e-4, 3); ax.set_xlim(mus[0], mus[-1])
    ax.set_xlabel(r"$\mu$ [eV]"); ax.set_ylabel(r"$\mathcal{B}(\mu)=p(d|H_\mu)/p(d|H_0)$"); ax.set_title(name)
    ax.legend(fontsize=7, loc="lower left")
plt.tight_layout(); plt.savefig(os.path.join(IMG, "fig6_bayesfactor.png"), dpi=140); plt.close()

# ---- Fig 7: systematics on M33 X-7 / IRAS
fig, axs = plt.subplots(1, 2, figsize=(13, 4.6))
for ax, name in zip(axs, res):
    R = res[name]; mus = R["mus"]
    for key, lab, c in [("fid", "fiducial", "k"), ("fid_detw", "uncalibrated analytic rates", "C0"),
                        ("sys_x0.1", "rates ×0.1", "C1"), ("sys_x10", "rates ×10", "C2"),
                        ("sys_l1", "l=1 only", "C3"), ("sys_tmin", "t = min of age prior", "C4")]:
        ax.plot(mus, R[key], c, lw=2 if key == "fid" else 1.2, label=lab)
    ax.axhline(0.05, color="r", ls="--", lw=1)
    ax.set_xscale("log"); ax.set_ylim(-0.02, 1.02); ax.set_xlim(mus[0], mus[-1]); ax.set_xlabel(r"$\mu$ [eV]")
    ax.set_ylabel(r"$P_{\rm surv}$"); ax.set_title(f"{name}: systematics"); ax.legend(fontsize=7)
plt.tight_layout(); plt.savefig(os.path.join(IMG, "fig7_systematics.png"), dpi=140); plt.close()
print("figs ok")
