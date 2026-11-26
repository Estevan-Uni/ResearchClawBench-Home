"""Figures for understanding (LaTeX), robustness sweep, pathway probe, encoder probe, T2I grids."""
import json, os, glob, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from PIL import Image
from score_latex import score, GT
from robust_tf import apply
W = "/workspace"; I = f"{W}/report/images"; O = f"{W}/outputs"
def rj(p): return [json.loads(l) for l in open(p)] if os.path.exists(p) else []
core = {r["id"]: r for r in rj(f"{O}/understanding_core.jsonl")}
summary = {}
# ---- Fig 3: formula -> LaTeX, rendered back
if "eq_latex" in core:
    ans = core["eq_latex"]["answers"][0]["a"]; s = score(ans); summary["eq_latex"] = s
    fig, ax = plt.subplots(3, 1, figsize=(10, 6.4), gridspec_kw={"height_ratios": [2.2, 1, 1]})
    ax[0].imshow(Image.open(f"{W}/data/equation.png")); ax[0].set_title("(a) input image", fontsize=10); ax[0].axis("off")
    for a, tex, t in [(ax[1], s["extracted"], f"(b) Janus-1.3B output, re-rendered with matplotlib mathtext (normalised edit distance = {s['ned']:.3f}, exact={s['exact']}, brackets balanced={s['balanced']})"),
                      (ax[2], GT, "(c) hand-transcribed ground truth, rendered")]:
        a.axis("off"); a.set_title(t, fontsize=9)
        try: a.text(0.5, 0.4, f"${tex}$", fontsize=20, ha="center", va="center")
        except Exception as e: a.text(0.5, 0.4, f"render failed: {e}", ha="center")
    plt.tight_layout(); plt.savefig(f"{I}/fig3_latex_ocr.png", dpi=150); plt.close()
# ---- Fig 4: robustness
rob = rj(f"{O}/understanding_robust.jsonl")
if rob:
    rows = [{"id": "clean", "kind": "clean", "v": 0, **score(core["eq_latex"]["answers"][0]["a"])}] if "eq_latex" in core else []
    for r in rob:
        rows.append({"id": r["id"], "kind": r["transform"]["kind"], "v": r["transform"]["v"], **score(r["answers"][0]["a"])})
    summary["robust"] = rows
    kinds = ["scale", "blur", "noise", "rotate", "jpeg"]
    fig = plt.figure(figsize=(15, 7.2)); gs = fig.add_gridspec(3, 5, height_ratios=[1, 1, 1.5])
    eq = Image.open(f"{W}/data/equation.png").convert("RGB")
    for i, r in enumerate(rows[:10]):
        a = fig.add_subplot(gs[i // 5, i % 5]); im = eq if r["kind"] == "clean" else apply(eq, {"kind": r["kind"], "v": r["v"]})
        a.imshow(im); a.axis("off")
        a.set_title(f"{r['kind']}={r['v']}  NED={r['ned']:.2f}", fontsize=9, color="green" if r["ned"] == 0 else ("darkorange" if r["ned"] < 0.2 else "red"))
    a = fig.add_subplot(gs[2, :])
    xs = np.arange(len(rows)); ned = [r["ned"] for r in rows]
    cols = ["#2a9d4b" if n == 0 else ("#f0a030" if n < 0.2 else "#d0453a") for n in ned]
    a.bar(xs, ned, color=cols); a.set_xticks(xs); a.set_xticklabels([f"{r['kind']}\n{r['v']}" for r in rows], fontsize=7)
    a.set_ylabel("normalised edit distance\n(0 = exact)"); a.set_ylim(0, max(1.0, max(ned) * 1.05))
    for x, n in zip(xs, ned): a.text(x, n + 0.01, f"{n:.2f}", ha="center", fontsize=7)
    a.set_title("Formula→LaTeX robustness of the SigLIP understanding pathway (n=1 greedy decode per condition)", fontsize=10)
    plt.tight_layout(); plt.savefig(f"{I}/fig4_ocr_robustness.png", dpi=140); plt.close()
json.dump(summary, open(f"{O}/latex_scores.json", "w"), indent=1, ensure_ascii=False)
# ---- Fig 7: pathway probe (VQ recon + geometry)
pp = f"{O}/probes/pathway_probe.json"
if os.path.exists(pp):
    P = json.load(open(pp))
    fig, ax = plt.subplots(2, 4, figsize=(16, 7.5))
    for i, n in enumerate(["equation", "doge"]):
        ax[i, 0].imshow(Image.open(f"{O}/probes/{n}_input384.png")); ax[i, 0].set_title(f"{n}: 384² model input", fontsize=9)
        ax[i, 1].imshow(Image.open(f"{O}/probes/{n}_vqrecon.png")); ax[i, 1].set_title(f"VQ encode→decode (gen. encoder)\nPSNR={P[n]['psnr_vq']:.1f} dB, {P[n]['n_unique_codes']} unique codes", fontsize=9)
        for a in ax[i, :2]: a.axis("off")
        g = np.load(f"{O}/probes/{n}_geometry.npz")
        ax[i, 2].hist(g["norm_und"], 40, alpha=.6, label="SigLIP→und. adaptor", color="#1f77b4", density=True)
        ax[i, 2].hist(g["norm_gen"], 40, alpha=.6, label="VQ→gen. adaptor", color="#ff7f0e", density=True)
        ax[i, 2].hist(g["norm_txt"], 40, alpha=.6, label="text token emb.", color="gray", density=True)
        ax[i, 2].set_xscale("log"); ax[i, 2].set_xlabel("L2 norm of LLM input vector"); ax[i, 2].legend(fontsize=7); ax[i, 2].set_title("embedding norms", fontsize=9)
        ax[i, 3].hist(g["cos_und_gen_samepos"], 40, alpha=.7, label="cos(und_i, gen_i) same patch", color="purple", density=True)
        ax[i, 3].hist(g["cos_und_txt_max"], 40, alpha=.6, label="max cos(und, text vocab)", color="#1f77b4", density=True)
        ax[i, 3].hist(g["cos_gen_txt_max"], 40, alpha=.6, label="max cos(gen, text vocab)", color="#ff7f0e", density=True)
        ax[i, 3].set_xlabel("cosine similarity"); ax[i, 3].legend(fontsize=7); ax[i, 3].set_title("geometry in shared input space", fontsize=9)
    plt.tight_layout(); plt.savefig(f"{I}/fig7_pathway_probe.png", dpi=130); plt.close()
# ---- Fig 8: encoder probe
ep = f"{O}/probes/encoder_probe.json"
if os.path.exists(ep):
    E = json.load(open(ep)); ks = ["siglip_raw_mean", "und_mean", "und_grid", "gen_mean", "gen_grid"]
    lab = ["SigLIP raw\n(mean)", "SigLIP→und.\nadaptor (mean)", "SigLIP→und.\nadaptor (4×4)", "VQ→gen.\nadaptor (mean)", "VQ→gen.\nadaptor (4×4)"]
    c = ["#9ecae1", "#1f77b4", "#08519c", "#fdae6b", "#e6550d"]
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.8))
    for a, m, t in [(ax[0], "probe_acc", "semantic class: 5-fold linear-probe accuracy"), (ax[1], "thumb_r2", "low-level detail: 8×8 RGB thumbnail ridge R²")]:
        v = [E[k][m] for k in ks]; a.bar(range(5), v, color=c); a.set_xticks(range(5)); a.set_xticklabels(lab, fontsize=8); a.set_title(t, fontsize=10)
        for x, y in zip(range(5), v): a.text(x, y + 0.01, f"{y:.2f}", ha="center", fontsize=9)
    ax[0].axhline(0.1, ls="--", c="gray", lw=1); ax[0].text(4.4, 0.11, "chance", fontsize=8, ha="right")
    for k, cc, l in zip(ks, c, lab):
        ax[2].errorbar(E[k]["thumb_r2"], E[k]["probe_acc"], xerr=E[k].get("thumb_r2_std", 0), yerr=E[k].get("probe_acc_std", 0), fmt="o", ms=10, c=cc, label=l.replace("\n", " "))
    ax[2].legend(fontsize=8, loc="center right")
    ax[2].set_xlabel("thumbnail R² (pixel detail)"); ax[2].set_ylabel("class accuracy (semantics)"); ax[2].set_title(f"dissociation (n={E['n_images']} Imagenette val images, 10 classes)", fontsize=10)
    plt.tight_layout(); plt.savefig(f"{I}/fig8_encoder_probe.png", dpi=140); plt.close()
print(json.dumps({k: (v if k != "robust" else len(v)) for k, v in summary.items()}, indent=1)[:2000])
