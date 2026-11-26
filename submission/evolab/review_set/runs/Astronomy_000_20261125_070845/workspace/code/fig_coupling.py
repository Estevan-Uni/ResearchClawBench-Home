"""Self-interaction coupling plane (mu, g=1/f): contours, GeV^-1 table, consistency checks."""
import numpy as np, os, json, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import physics as ph, stats as st
from data_overview import ROOT, load

IMG = os.path.join(ROOT, "report/images")
MODLAB = {"bosenova": "bosenova cap", "mixing": "level-mixing equilibrium"}


def get_P(D, model, c, which):
    nt = len(D["ages"])
    if which == "marg":
        return np.mean([D[f"{model}_{c:g}_t{i}"] for i in range(2, nt)], 0)
    return D[f"{model}_{c:g}_t{which}"]


def gmax_curve(P, g):
    """largest excluded g (P<0.05) per mu column, requiring exclusion to be connected to g->0."""
    out = np.full(P.shape[0], np.nan)
    for i, row in enumerate(P):
        ex = row < 0.05
        if not ex[0]:
            continue
        j = np.argmax(~ex) if (~ex).any() else len(ex)
        if j == 0:
            continue
        # log-interpolate between g[j-1] (excluded) and g[j] (allowed) on P=0.05
        if j < len(g):
            p1, p2 = row[j - 1], row[j]
            fr = (0.05 - p1) / (p2 - p1) if p2 != p1 else 0
            out[i] = 10 ** (np.log10(g[j - 1]) + fr * (np.log10(g[j]) - np.log10(g[j - 1])))
        else:
            out[i] = g[-1]
    return out


def coupling_plot(D, name, fname, which_list):
    mus, g = D["mus"], D["g"]
    fig, axs = plt.subplots(1, len(which_list), figsize=(7.6 * len(which_list), 5.6), squeeze=False)
    fig.subplots_adjust(wspace=0.42, right=0.86)
    for ax, (which, tlab) in zip(axs[0], which_list):
        P = get_P(D, "mixing", 1.0, which)
        cf = ax.pcolormesh(mus, g, P.T, cmap="viridis", vmin=0, vmax=1, shading="nearest", rasterized=True)
        for model, cs, col in [("bosenova", [0.5, 5.0, 50.0], "w"), ("mixing", [1 / 25., 1.0, 25.0], "C3")]:
            for c, ls in zip(cs, [":", "-", "--"]):
                Pm = get_P(D, model, c, which)
                if (Pm < 0.05).any():
                    ax.contour(mus, g, Pm.T, levels=[0.05], colors=[col], linestyles=[ls], linewidths=2 if ls == "-" else 1.2)
                ax.plot([], [], color=col, ls=ls, label=f"{MODLAB[model]}, c={c:.3g}")
        ax.plot(mus, 1 / ph.f_qcd(mus), "orange", lw=2, label="QCD axion")
        ax.axhline(1 / ph.MPL, color="0.7", ls="-.", lw=1); ax.text(mus[1], 1.3 / ph.MPL, r"$f=M_{\rm Pl}$", color="0.8", fontsize=8)
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_ylim(g[0], g[-1])
        ax.set_xlabel(r"$\mu$ [eV]"); ax.set_ylabel(r"$g \equiv 1/f$ [GeV$^{-1}$]")
        ax.set_title(f"{name}: {tlab}\n(colour: $P_{{\\rm surv}}$, level mixing c=1; lines: $P_{{\\rm surv}}=0.05$)", fontsize=9)
        sec = ax.secondary_yaxis("right", functions=(lambda x: 1 / np.clip(x, 1e-40, None), lambda x: 1 / np.clip(x, 1e-40, None)))
        sec.set_ylabel(r"$f$ [GeV]")
        ax.text(0.03, 0.04, "EXCLUDED below contours\n(weak coupling: spin-down completes)", transform=ax.transAxes, color="w", fontsize=8)
        ax.text(0.03, 0.88, "allowed (strong self-interaction\nquenches superradiance)", transform=ax.transAxes, color="w", fontsize=8)
        ax.legend(fontsize=6.5, loc="upper right" if name.startswith("IRAS") else "center right", framealpha=0.85)
    cax = fig.add_axes([0.915, 0.15, 0.012, 0.7]); plt.colorbar(cf, cax=cax, label=r"$P_{\rm surv}$")
    plt.savefig(os.path.join(IMG, fname), dpi=140, bbox_inches="tight"); plt.close()


if __name__ == "__main__":
    out = {}
    D = np.load(os.path.join(ROOT, "outputs/scan2d_M33.npy"), allow_pickle=True).item()
    mus, g = D["mus"], D["g"]
    coupling_plot(D, "M33 X-7", "fig8_coupling_M33.png",
                  [(0, r"$t=t_{\rm Sal}=4.5\times10^7$ yr"), ("marg", r"age-marginalised, log-U[$10^5$, $3\times10^6$] yr")])
    # g_max curves for all models / ages
    curves = {}
    for model, cs in [("bosenova", [0.5, 5.0, 50.0]), ("mixing", [1 / 25., 1.0, 25.0])]:
        for c in cs:
            for which in [0, 1, "marg"]:
                curves[f"{model}_{c:g}_{which}"] = gmax_curve(get_P(D, model, c, which), g)
    # table at selected masses
    gm_ref = curves["mixing_1_0"]
    ok = ~np.isnan(gm_ref)
    sel_idx = sorted(set([np.where(ok)[0][0], np.where(ok)[0][-1], int(np.nanargmax(gm_ref))]
                         + [int(np.argmin(np.abs(mus - m))) for m in [3e-13, 1e-12, 2.04e-12, 3e-12, 6e-12]]))
    rows = []
    for i in sel_idx:
        r = dict(mu=float(mus[i]))
        for k, v in curves.items():
            r[k] = float(v[i])
        rows.append(r)
    out["table"] = rows
    out["curves"] = {k: v.tolist() for k, v in curves.items()}; out["mus"] = mus.tolist()
    # f->infinity consistency check vs 1-D P_surv
    M, a = load("M33 X-7")
    p1d = np.array([st.p_surv(M, a, mu, st.T_SALPETER, rate_model="calibrated") for mu in mus])
    out["finf_check_maxdiff"] = float(np.max(np.abs(get_P(D, "mixing", 1.0, 0)[:, 0] - p1d)))
    # excluded mu-range along QCD axion line, and for the f->inf limit
    for model, c in [("bosenova", 5.0), ("mixing", 1.0)]:
        P = get_P(D, model, c, 0)
        gq = 1 / ph.f_qcd(mus)
        pq = np.array([np.interp(np.log10(gq[i]), np.log10(g), P[i]) for i in range(len(mus))])
        out[f"qcd_{model}"] = st.fmt_int(st.intervals(mus, pq < 0.05))
    out["finf_excl"] = st.fmt_int(st.intervals(mus, p1d < 0.05))
    # figure: g_max(mu) curves
    fig, ax = plt.subplots(figsize=(8, 5))
    for model, cs, col in [("bosenova", [0.5, 5.0, 50.0], "C0"), ("mixing", [1 / 25., 1.0, 25.0], "C3")]:
        for c, ls in zip(cs, [":", "-", "--"]):
            ax.plot(mus, curves[f"{model}_{c:g}_0"], color=col, ls=ls, lw=2 if ls == "-" else 1.2,
                    label=f"{MODLAB[model]}, c={c:.3g}, t=4.5e7 yr")
        ax.plot(mus, curves[f"{model}_{cs[1]:g}_marg"], color=col, ls="-", lw=1, alpha=0.5,
                label=f"{MODLAB[model]}, c={cs[1]:g}, age-marginalised")
        ax.plot(mus, curves[f"{model}_{cs[1]:g}_1"], color=col, ls="-.", lw=1, alpha=0.8,
                label=f"{MODLAB[model]}, c={cs[1]:g}, t=3e6 yr")
    ax.plot(mus, 1 / ph.f_qcd(mus), "orange", lw=2, label="QCD axion")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_ylim(1e-20, 1e-10)
    ax.set_xlabel(r"$\mu$ [eV]"); ax.set_ylabel(r"95% bound: excluded $g=1/f\ <\ g_{\max}(\mu)$ [GeV$^{-1}$]")
    ax.fill_between(mus, 1e-20, np.nan_to_num(curves["mixing_1_0"], nan=1e-20), color="C3", alpha=0.12)
    ax.set_title("M33 X-7: upper edge of the excluded self-coupling band"); ax.legend(fontsize=7)
    plt.tight_layout(); plt.savefig(os.path.join(IMG, "fig9_gmax_M33.png"), dpi=140); plt.close()
    # IRAS
    fn = os.path.join(ROOT, "outputs/scan2d_IRAS.npy")
    if os.path.exists(fn):
        DI = np.load(fn, allow_pickle=True).item()
        coupling_plot(DI, "IRAS 09149-6206", "fig10_coupling_IRAS.png",
                      [(0, r"$t=4.5\times10^7$ yr"), (1, r"$t=10^{10}$ yr (optimistic)")])
        for model, c in [("bosenova", 5.0), ("mixing", 1.0)]:
            for w in [0, 1]:
                P = get_P(DI, model, c, w)
                out[f"IRAS_{model}_t{w}_minP"] = float(P.min())
                gmI = gmax_curve(P, DI["g"])
                out[f"IRAS_{model}_t{w}_gmax"] = [[float(m), float(x)] for m, x in zip(DI["mus"], gmI) if not np.isnan(x)][::5]
    json.dump(out, open(os.path.join(ROOT, "outputs/coupling_results.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k not in ("curves", "mus")}, indent=1)[:6000])
