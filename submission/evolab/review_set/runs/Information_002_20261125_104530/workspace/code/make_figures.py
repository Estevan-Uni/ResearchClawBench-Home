"""Figures for HF results, algebra checks and the data overview."""
import json, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, "/workspace/code")
from moire import *

IMG = "/workspace/report/images/"
O = "/workspace/outputs/"
summary = {}

# ---- bare nu=2 overlap on a fine mesh ----
m = Model(Vzt=-20.0)
k, _ = kmesh(24)
Eh = {t: np.array([np.linalg.eigvalsh(m.htilde(kk, t))[:3] for kk in k]) for t in (1, -1)}
b1max = max(Eh[t][:, 0].max() for t in (1, -1)); b2min = min(Eh[t][:, 1].min() for t in (1, -1))
summary["bare_nu2_indirect_gap_24mesh_meV"] = float(b2min - b1max)
summary["bare_band1_width_meV"] = float(Eh[1][:, 0].max() - Eh[1][:, 0].min())

# ---- HF bands ----
nu2 = json.load(open(O + "hf_nu2_mesh.json")); nu1 = json.load(open(O + "hf_nu1_mesh.json"))
fig, axs = plt.subplots(1, 2, figsize=(11, 4.4))
for ax, res, title in ((axs[0], [r for r in nu2 if r["N"] == 9][0], r"$\nu=2$ TRS ansatz, $\epsilon=15$, $V_{zt}=-20$ meV"),
                       (axs[1], [r for r in nu1 if r["N"] == 9 and r["ansatz"] == "VP"][0], r"$\nu=1$ VP ansatz, $\epsilon=15$, $V_{zt}=-10$ meV")):
    B = res["bands"]; x = np.array(B["x"])
    # electron-picture energies = - hole energies (plot as valence bands like paper)
    for t, col in (("1", "tab:orange"), ("-1", "tab:blue")):
        Ehf = -np.array(B["E_hf"][t]); E0 = -np.array(B["E_bare"][t])
        ax.plot(x, Ehf, color=col, lw=1.6, label=("+K HF" if t == "1" else "-K HF"))
        ax.plot(x, E0, color=col, lw=0.8, ls=":", alpha=0.8)
    ax.axhline(-res["mu_occ_max"], color="k", ls="--", lw=0.8); ax.axhline(-res["mu_unocc_min"], color="k", ls="--", lw=0.8)
    for xx in B["xt"]: ax.axvline(xx, color="gray", lw=0.5)
    ax.set_xticks(B["xt"]); ax.set_xticklabels(B["lab"]); ax.set_xlim(x[0], x[-1])
    ax.set_ylabel(r"$-E^{\rm hole}$ (meV)  [valence-band picture]")
    ch = res["chern_hole_basis"]
    ax.set_title(f"{title}\n9x9 mesh: gap={res['gap_meV']:.2f} meV, C(hole)={ch}", fontsize=9)
    ax.set_ylim(-110, 80)
h, l = axs[0].get_legend_handles_labels()
axs[0].legend([h[0], h[4]], ["+K (HF; dotted = bare)", "-K (HF; dotted = bare)"], fontsize=8, loc="lower left")
plt.tight_layout(); plt.savefig(IMG + "hf_bands.png", dpi=150); plt.close()

# ---- convergence: gap vs mesh and SCF histories ----
fig, axs = plt.subplots(1, 2, figsize=(10, 3.8))
ax = axs[0]
for res, lab, mk in ((nu2, r"$\nu=2$ TRS ($V_{zt}=-20$)", "o"), ([r for r in nu1 if r["ansatz"] == "VP"], r"$\nu=1$ VP ($V_{zt}=-10$)", "s")):
    ax.plot([r["N"] for r in res], [r["gap_meV"] for r in res], marker=mk, label=lab)
try:
    sh = json.load(open(O + "hf_shell.json"))
    for r in sh:
        ax.scatter([6], [r["gap_meV"]], marker="x" if r["nshell"] == 3 else "+", s=60, color="k" if r["nu"] == 2 else "gray",
                   zorder=5)
        ax.annotate(f"nshell={r['nshell']}", (6, r["gap_meV"]), fontsize=7, xytext=(4, 2), textcoords="offset points")
    summary["shell_check"] = [{k: r[k] for k in ("nu", "nshell", "gap_meV", "chern_hole_basis", "niter")} for r in sh]
except FileNotFoundError:
    pass
ax.set_xlabel("N (N x N Gamma-centred mesh)"); ax.set_ylabel("HF charge gap (meV)"); ax.legend(fontsize=8)
ax.set_title(r"mesh / plane-wave convergence ($\epsilon=15$)", fontsize=9)
ax = axs[1]
for res, lab in ((nu2, r"$\nu=2$"), ([r for r in nu1 if r["ansatz"] == "VP"], r"$\nu=1$ VP")):
    for r in res:
        d = np.array([h[1] for h in r["hist"]])
        ax.semilogy(np.arange(1, len(d) + 1), d, label=f"{lab}, N={r['N']}")
ax.set_xlabel("SCF iteration"); ax.set_ylabel(r"max $|\rho_{\rm out}-\rho_{\rm in}|$"); ax.legend(fontsize=7, ncol=2)
ax.set_title("SCF convergence (linear mixing 0.6)", fontsize=9)
plt.tight_layout(); plt.savefig(IMG + "hf_convergence.png", dpi=150); plt.close()

# ---- scans ----
fig, axs = plt.subplots(1, 2, figsize=(10, 3.8))
try:
    s2 = json.load(open(O + "hf_nu2_scan.json"))
    try:
        s2 += json.load(open(O + "hf_nu2_fine.json"))
    except FileNotFoundError:
        pass
    ax = axs[0]
    for eps, col in ((15.0, "tab:green"), (10.0, "tab:purple")):
        rr = sorted([r for r in s2 if r["eps"] == eps], key=lambda r: r["Vzt"])
        V = [r["Vzt"] for r in rr]; g = [r["gap_meV"] for r in rr]
        ax.plot(V, g, "-", color=col, label=rf"$\epsilon={eps:.0f}$")
        for r in rr:
            c = r["chern_hole_basis"]["+K"]
            mk = "o" if c not in (0, 0.0, None) else "s"
            ax.scatter(r["Vzt"], r["gap_meV"], marker=mk, facecolor=col if mk == "o" else "white", edgecolor=col, s=40, zorder=3)
            if r["flag"]:
                ax.annotate("slow", (r["Vzt"], r["gap_meV"]), fontsize=6, xytext=(2, 4), textcoords="offset points")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel(r"$V_{zt}$ (meV)"); ax.set_ylabel("HF gap (meV)")
    ax.set_title(r"$\nu=2$ TRS, 6x6: filled = $|C_{\pm K}|=1$ ($Z_2$ TI), open = trivial", fontsize=9); ax.legend(fontsize=8)
    summary["nu2_scan"] = [{k: r[k] for k in ("eps", "Vzt", "gap_meV", "chern_hole_basis", "niter", "flag")} for r in s2]
except FileNotFoundError:
    pass
try:
    s1 = json.load(open(O + "hf_nu1_scan.json"))
    ax = axs[1]
    rr = sorted(s1, key=lambda r: r["Vzt"])
    ax.plot([r["Vzt"] for r in rr], [r["gap_meV"] for r in rr], "-", color="tab:red")
    for r in rr:
        c = r["chern_hole_basis"]["+K"]
        mk = "o" if c not in (0, 0.0, None) else "s"
        ax.scatter(r["Vzt"], r["gap_meV"], marker=mk, facecolor="tab:red" if mk == "o" else "white", edgecolor="tab:red", s=40, zorder=3)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel(r"$V_{zt}$ (meV)"); ax.set_ylabel("HF gap (meV)")
    ax.set_title(r"$\nu=1$ VP ansatz, $\epsilon=15$, 6x6: filled = $|C|=1$", fontsize=9)
    summary["nu1_scan"] = [{k: r[k] for k in ("Vzt", "gap_meV", "chern_hole_basis", "niter", "flag", "valley_filling")} for r in s1]
except FileNotFoundError:
    pass
plt.tight_layout(); plt.savefig(IMG + "hf_scans.png", dpi=150); plt.close()

# ---- algebra checks ----
a = json.load(open(O + "algebra_checks.json")); sp_ = json.load(open(O + "single_particle_checks.json"))
items = [("S7 FFT coeffs = h entries", sp_["step7_fft_max_abs_error_meV"], True),
         ("S8 sum b h b^dag", a["step8_sum_b_h_bdag_vs_original"], True),
         ("S9 Tr h - b^dag h^T b", a["step9_gold_Trh_minus_bdag_hT_b"], True),
         ("S9 WRONG: drop Tr h", a["step9_WRONG_drop_trace"], False),
         ("S9 WRONG: no transpose", a["step9_WRONG_no_transpose"], False),
         ("S11 6-term Wick <H_MF>=<H>", a["steps11-13"]["Ewick6_minus_exact"], True),
         ("S11 WRONG: drop c-numbers (auto.md)", a["steps11-13"]["WRONG_wick_no_cnumbers"], False),
         ("S12-13 4-term(1/2A)=2-term(1/A)", a["steps11-13"]["quad4_vs_hf2_operator"], True),
         ("S13 HF op = dE/drho", a["steps11-13"]["grad_check_hf2"], True),
         ("S13 WRONG: 1/2A after merging", a["steps11-13"]["WRONG_hf2_halfA_grad"], False),
         ("S14 Hartree reduced=unreduced", a["step14_hartree_reduced_vs_unreduced"], True),
         ("S15 Fock reduced=unreduced", a["step15_fock_reduced_vs_unreduced"], True),
         ("S15 WRONG: Fock with V(|q1-q4|)", a["step15_WRONG_fock_with_V(q1-q4)"], False)]
bf = None
try:
    bf = float(open(O + "bruteforce_hf.txt").read().split("diff")[-1])
    items.append(("S16 code Sigma vs brute-force <H_int>", bf, True))
except Exception:
    pass
fig, ax = plt.subplots(figsize=(7.5, 5.2))
vals = [max(v, 1e-17) for _, v, _ in items]
ax.barh(range(len(items)), np.log10(vals), color=["tab:green" if g else "tab:red" for _, _, g in items])
ax.set_yticks(range(len(items))); ax.set_yticklabels([n for n, _, _ in items], fontsize=8); ax.invert_yaxis()
ax.axvline(-10, color="k", ls=":", lw=0.8)
ax.set_xlabel(r"log$_{10}$ max |deviation|  (green: gold form; red: wrong variant)")
ax.set_title("Exact verification of template steps (Fock space / coefficient space)", fontsize=10)
plt.tight_layout(); plt.savefig(IMG + "algebra_checks.png", dpi=150); plt.close()

# ---- data overview figure: placeholder counts per step & rubric for GPT-4 ----
import yaml
Y = [t for t in yaml.safe_load(open("/workspace/data/2111.01152/2111.01152.yaml")) if "task" in t]
ITEMS = ["in_paper", "prompt_quality", "follow_instructions", "physics_logic", "math_derivation", "final_answer_accuracy"]
R = np.array([[t["score"][k] for k in ITEMS] for t in Y])
npl = [len(t.get("placeholder") or {}) for t in Y]
fig, axs = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw={"width_ratios": [1, 1.4]})
axs[0].barh(range(16), npl, color="#4C72B0"); axs[0].invert_yaxis()
axs[0].set_yticks(range(16)); axs[0].set_yticklabels([f"{i+1}. {t['task'][:45]}" for i, t in enumerate(Y)], fontsize=7)
axs[0].set_xlabel("# placeholders in template"); axs[0].set_title("Template chain (16 steps)", fontsize=9)
im = axs[1].imshow(R, cmap="RdYlGn", vmin=0, vmax=2, aspect="auto")
for i in range(16):
    for j in range(6):
        axs[1].text(j, i, R[i, j], ha="center", va="center", fontsize=7)
axs[1].set_xticks(range(6)); axs[1].set_xticklabels([s.replace("_", "\n") for s in ITEMS], fontsize=7)
axs[1].set_yticks(range(16)); axs[1].set_yticklabels([str(i + 1) for i in range(16)], fontsize=7)
axs[1].set_title("GPT-4 step rubric (human grades from yaml)", fontsize=9)
plt.tight_layout(); plt.savefig(IMG + "data_overview.png", dpi=150); plt.close()

json.dump(summary, open(O + "figure_summary.json", "w"), indent=1, default=str)
print(json.dumps(summary, indent=1, default=str))
