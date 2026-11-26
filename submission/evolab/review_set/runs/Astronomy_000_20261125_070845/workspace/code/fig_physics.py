"""Rate check vs numerical results and Regge-plane instability regions with posterior samples."""
import numpy as np, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import physics as ph, stats as st
from data_overview import load, ROOT

IMG = os.path.join(ROOT, "report/images")
# ---- rate check
fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
al = np.linspace(0.01, 1.6, 3000)
for l, c in zip([1, 2, 3], ["C0", "C1", "C2"]):
    v = ph.MomegaI(al, 0.99, l); vc = v * ph.calib_factor(al, l)
    ax[0].semilogy(al, np.where(v > 0, v, np.nan), c, ls="--", label=f"l=m={l} analytic (Detweiler/BCP)")
    ax[0].semilogy(al, np.where(vc > 0, vc, np.nan), c, label=f"l=m={l} calibrated (fiducial)")
lead = 0.99 * al ** 9 / 48
ax[0].semilogy(al[al < 0.6], lead[al < 0.6], "k:", label=r"leading order $a\alpha^9/48$")
ax[0].plot(0.42, 1.5e-7, "r*", ms=14, label="Dolan (2007) numerical peak, l=1")
ax[0].set_ylim(1e-14, 1e-5); ax[0].set_xlabel(r"$\alpha = GM\mu/\hbar c$"); ax[0].set_ylabel(r"$M\omega_I$  ($a_*=0.99$)")
ax[0].legend(fontsize=7); ax[0].set_title("Growth-rate check")
# spin-down times for M33 X-7 median
M0, a0 = 15.66, 0.836
mus = np.logspace(-14, -10, 600)
for l, c in zip(range(1, 7), [f"C{i}" for i in range(6)]):
    t1 = ph.t_spindown(M0, a0, mus, l, rate_model="calibrated")
    t2 = ph.t_spindown(M0, a0, mus, l, rate_model="detweiler")
    ax[1].loglog(mus, t1, c, label=f"l={l}"); ax[1].loglog(mus, t2, c, ls="--", lw=0.8)
ax[1].axhline(4.5e7, color="k", ls="-.", lw=1, label=r"$t_{\rm Sal}=4.5\times10^7$ yr")
ax[1].axhline(3e6, color="gray", ls=":", lw=1, label=r"$3\times10^6$ yr")
ax[1].set_ylim(1e-6, 1e12); ax[1].set_xlabel(r"$\mu$ [eV]"); ax[1].set_ylabel("spin-down time [yr]")
ax[1].set_title(r"M33 X-7 at median ($M=15.7M_\odot$, $a_*=0.84$); dashed: uncalibrated"); ax[1].legend(fontsize=7, ncol=2)
plt.tight_layout(); plt.savefig(os.path.join(IMG, "fig2_rates.png"), dpi=140); plt.close()

# ---- Regge planes
fig, axs = plt.subplots(1, 2, figsize=(12, 4.8))
for ax, name, mlist in [(axs[0], "M33 X-7", [3e-13, 1e-12, 3e-12, 6e-12]),
                        (axs[1], "IRAS 09149-6206", [3e-20, 1e-19, 3e-19, 1e-18])]:
    M, a = load(name)
    t = st.AGES[name]["fid"]
    lm = np.log10(M)
    Mg = np.logspace(lm.min() - 0.6, lm.max() + 0.6, 400) if name != "M33 X-7" else np.linspace(5, 30, 400)
    ag = np.linspace(0, 0.999, 400)
    MM, AA = np.meshgrid(Mg, ag)
    ax.scatter(M, a, s=1, color="k", alpha=0.25, rasterized=True)
    for mu, c in zip(mlist, ["C0", "C1", "C2", "C3"]):
        U = ph.unstable(MM, AA, mu, t, rate_model="calibrated").astype(float)
        ax.contourf(MM, AA, U, levels=[0.5, 1.5], colors=[c], alpha=0.07)
        ax.contour(MM, AA, U, levels=[0.5], colors=[c], linewidths=2)
        ax.plot([], [], c, label=rf"$\mu$={mu:.0e} eV: P$_{{\rm surv}}$={st.p_surv(M, a, mu, t, rate_model='calibrated'):.2f}")
    if name != "M33 X-7": ax.set_xscale("log")
    ax.set_xlabel(r"$M\ [M_\odot]$"); ax.set_ylabel(r"$a_*$"); ax.set_ylim(0.0 if name == "M33 X-7" else 0.7, 1)
    ax.set_title(f"{name}: unstable regions, t = 4.5e7 yr"); ax.legend(fontsize=7, loc="lower right")
plt.tight_layout(); plt.savefig(os.path.join(IMG, "fig3_regge.png"), dpi=140); plt.close()
print("ok")
