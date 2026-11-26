import numpy as np, os, json
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from data_overview import ROOT

fig, axs = plt.subplots(1, 2, figsize=(13, 4.6))
for ax, tag, name in zip(axs, ["M33", "IRAS"], ["M33 X-7 (500 mocks/mass)", "IRAS 09149-6206 (300 mocks/mass)"]):
    d = json.load(open(os.path.join(ROOT, f"outputs/injection_{tag}.json")))
    mu = np.array([r["mu"] for r in d])
    for k, lab, c, ls in [("fe_psurv", r"false excl.: $P_{\rm surv}<0.05$", "k", "-"),
                          ("fe_B", r"false excl.: $\mathcal{B}<0.05$", "C0", "-"),
                          ("fe_box", r"false excl.: whole $1\sigma$ box unstable", "C2", "-"),
                          ("fe_point", "false excl.: best-fit point unstable", "C3", "-"),
                          ("power_psurv", r"power (no boson, $a_0\sim U$): $P_{\rm surv}<0.05$", "k", "--"),
                          ("power_B", r"power: $\mathcal{B}<0.05$", "C0", "--")]:
        ax.plot(mu, [r[k] for r in d], color=c, ls=ls, marker="o", ms=3, label=lab)
    ax.axhline(0.05, color="r", ls=":", lw=1)
    ax.set_xscale("log"); ax.set_ylim(-0.02, 1.0); ax.set_xlabel(r"true $\mu$ [eV]"); ax.set_ylabel("rate")
    ax.set_title(name + ", t = 4.5e7 yr"); ax.legend(fontsize=7)
plt.tight_layout(); plt.savefig(os.path.join(ROOT, "report/images/fig11_injection.png"), dpi=140)
for tag in ["M33", "IRAS"]:
    d = json.load(open(os.path.join(ROOT, f"outputs/injection_{tag}.json")))
    print(tag, {k: max(r[k] for r in d) for k in ["fe_psurv", "fe_B", "fe_box", "fe_point", "power_psurv", "power_B"]},
          "mean_invB range", min(r["mean_invB"] for r in d), max(r["mean_invB"] for r in d))
