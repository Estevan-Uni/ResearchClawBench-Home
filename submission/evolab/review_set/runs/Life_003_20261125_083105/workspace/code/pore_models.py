"""Pore-model characterisation: base x position effects, ranked-current substitution heatmaps,
single-base substitution profiles, secondary-peak ("reader head") test, pairwise interactions,
effective context width, and sequence dependence of std / dwell."""
import json
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from common import *

res = {}
cache = {}


def group_means(y, g, n):
    s = np.bincount(g, weights=y, minlength=n)
    c = np.bincount(g, minlength=n)
    return s / np.maximum(c, 1)


def eta2(y, g, n):
    m = group_means(y, g, n)
    fit = m[g]
    return float(np.sum((fit - y.mean()) ** 2) / np.sum((y - y.mean()) ** 2))


def additive_fit(y, X, positions):
    """Exact additive (main-effect) fit for a balanced full factorial design."""
    fit = np.full_like(y, y.mean())
    for p in positions:
        m = group_means(y, X[:, p], 4)
        fit += m[X[:, p]] - y.mean()
    return fit


def r2(y, fit):
    return float(1 - np.sum((y - fit) ** 2) / np.sum((y - y.mean()) ** 2))


def local_maxima(v, exclude):
    out = []
    for i in range(len(v)):
        if i in exclude:
            continue
        left = v[i - 1] if i > 0 else -np.inf
        right = v[i + 1] if i < len(v) - 1 else -np.inf
        if v[i] > left and v[i] > right:
            out.append(i)
    return out


for name in MODELS:
    df, K, X = load_model(name)
    y = df.current_mean.values.astype(float)
    sd = df.current_std.values
    dw = df.dwell_time.values.astype(float)
    R = {"K": K}
    # 1 marginal effects
    eff = np.array([[group_means(y, X[:, p], 4)[b] - y.mean() for b in range(4)] for p in range(K)])
    et = np.array([eta2(y, X[:, p], 4) for p in range(K)])
    R["marginal_effect"] = eff.tolist()
    R["eta2"] = et.tolist()
    c = int(np.argmax(et))
    R["dominant_pos"] = c + 1
    # purine/pyrimidine
    pur = np.isin(X, [0, 2]).sum(1)
    gc = np.isin(X, [1, 2]).sum(1)
    R["spearman_purine_count"] = float(pd.Series(pur).corr(pd.Series(y), method="spearman"))
    R["spearman_gc_count"] = float(pd.Series(gc).corr(pd.Series(y), method="spearman"))
    # 2 ranked-current composition
    nb = 64
    rank = np.argsort(np.argsort(y))
    bins = (rank * nb // len(y)).astype(int)
    comp = np.zeros((K, nb, 4))
    for p in range(K):
        for b in range(4):
            comp[p, :, b] = group_means((X[:, p] == b).astype(float), bins, nb)
    # 3 single-base substitutions
    pw = 4 ** (K - 1 - np.arange(K))
    codes = df.code.values
    sub = {}
    for p in range(K):
        d_all, z_all = [], []
        for shift in (1, 2, 3):
            nbase = (X[:, p] + shift) % 4
            nc = codes + (nbase.astype(np.int64) - X[:, p]) * pw[p]
            d = np.abs(y[nc] - y)
            d_all.append(d)
            z_all.append(d / np.sqrt(0.5 * (sd ** 2 + sd[nc] ** 2)))
        d = np.concatenate(d_all); z = np.concatenate(z_all)
        sub[p] = dict(median=float(np.median(d)), q25=float(np.quantile(d, .25)), q75=float(np.quantile(d, .75)),
                      q10=float(np.quantile(d, .1)), q90=float(np.quantile(d, .9)),
                      z_median=float(np.median(z)), z_q25=float(np.quantile(z, .25)), z_q75=float(np.quantile(z, .75)),
                      z_q10=float(np.quantile(z, .1)), z_q90=float(np.quantile(z, .9)),
                      frac_z_gt1=float(np.mean(z > 1)), frac_z_gt2=float(np.mean(z > 2)))
    R["substitution"] = sub
    dmed = np.array([sub[p]["median"] for p in range(K)])
    # 4 secondary-peak test
    nb_excl = {c - 1, c, c + 1} & set(range(K))
    fit_core = additive_fit(y, X, sorted(nb_excl))
    resid = y - fit_core
    et_res = np.array([eta2(resid, X[:, p], 4) if p not in nb_excl else np.nan for p in range(K)])
    R["core_positions"] = [p + 1 for p in sorted(nb_excl)]
    R["core_R2"] = r2(y, fit_core)
    R["eta2_resid_after_core"] = [None if np.isnan(v) else float(v) for v in et_res]
    R["local_max_eta2_offcentre"] = [p + 1 for p in local_maxima(et, {c})]
    R["local_max_dmu_offcentre"] = [p + 1 for p in local_maxima(dmed, {c})]
    left = et[:c].sum(); right = et[c + 1:].sum()
    R["eta2_left_sum"] = float(left); R["eta2_right_sum"] = float(right)
    R["asymmetry_index"] = float((right - left) / (right + left))
    # mirrored-pair asymmetry where both sides exist
    mirror = [(c - d, c + d) for d in range(1, K) if c - d >= 0 and c + d < K]
    R["mirror_pairs"] = [dict(offset=d + 1, left=float(et[a]), right=float(et[b]))
                         for d, (a, b) in enumerate(mirror)]
    # monotone decay test: is eta2 monotone non-increasing moving away from c on each side?
    R["monotone_left"] = bool(np.all(np.diff(et[:c + 1]) >= 0))
    R["monotone_right"] = bool(np.all(np.diff(et[c:]) <= 0))
    # ANOVA decomposition (balanced): additive, all pairwise, higher order
    fit_add = additive_fit(y, X, range(K))
    r_add = y - fit_add
    sst = np.sum((y - y.mean()) ** 2)
    R["R2_additive"] = r2(y, fit_add)
    pair = np.full((K, K), np.nan)
    fit_pairs = fit_add.copy()
    for i in range(K):
        for j in range(i + 1, K):
            g = X[:, i].astype(int) * 4 + X[:, j]
            m = group_means(r_add, g, 16)
            pair[i, j] = pair[j, i] = np.sum(m[g] ** 2) / sst
            fit_pairs += m[g]
    R["pair_gain"] = np.nan_to_num(pair).tolist()
    R["R2_additive_plus_all_pairs"] = r2(y, fit_pairs)
    R["R2_additive_plus_adjacent_pairs"] = float(R["R2_additive"] + sum(pair[i, i + 1] for i in range(K - 1)))
    iu = np.triu_indices(K, 1)
    top = np.argsort(-pair[iu])[:5]
    R["top_pairs"] = [dict(i=int(iu[0][t] + 1), j=int(iu[1][t] + 1), gain=float(pair[iu][t])) for t in top]
    nonadj = [(i, j) for i, j in zip(*iu) if j - i > 1]
    R["max_nonadjacent_pair_gain"] = float(max(pair[i, j] for i, j in nonadj))
    R["sum_adjacent_pair_gain"] = float(sum(pair[i, i + 1] for i in range(K - 1)))
    R["sum_nonadjacent_pair_gain"] = float(sum(pair[i, j] for i, j in nonadj))
    # residual-on-centre second peak in full model: off-centre eta2 of additive residual from centre-only
    fit_c = additive_fit(y, X, [c])
    R["eta2_resid_after_centre_only"] = [float(eta2(y - fit_c, X[:, p], 4)) if p != c else None for p in range(K)]
    # 5 effective context: best contiguous window full-interaction R2
    win = {}
    for w in range(1, min(K, 6) + 1):
        best = (-1, None)
        for s in range(K - w + 1):
            g = np.zeros(len(y), dtype=np.int64)
            for p in range(s, s + w):
                g = g * 4 + X[:, p]
            v = eta2(y, g, 4 ** w)
            if v > best[0]:
                best = (v, s + 1)
        win[w] = dict(R2=float(best[0]), start=best[1])
    R["window_R2"] = win
    # std / dwell sequence dependence
    R["eta2_std"] = [eta2(sd, X[:, p], 4) for p in range(K)]
    R["eta2_dwell"] = [eta2(dw, X[:, p], 4) for p in range(K)]
    R["R2_additive_std"] = r2(sd, additive_fit(sd, X, range(K)))
    R["R2_additive_dwell"] = r2(dw, additive_fit(dw, X, range(K)))
    R["corr_mean_std"] = float(np.corrcoef(y, sd)[0, 1])
    R["corr_mean_dwell"] = float(np.corrcoef(y, dw)[0, 1])
    # dwell distribution shape
    R["dwell_quantiles"] = np.quantile(dw, [.05, .25, .5, .75, .95]).tolist()
    # k-mer separability: fraction of k-mers whose nearest neighbour in current is within 1 sd
    ys = np.sort(y)
    gaps = np.diff(ys)
    R["median_nn_gap"] = float(np.median(gaps))
    R["frac_sub_detectable_z1_centre"] = sub[c]["frac_z_gt1"]
    res[name] = R
    cache[name] = dict(eff=eff, et=et, et_res=et_res, comp=comp, sub=sub, pair=pair, y=y, sd=sd, dw=dw,
                       X=X, c=c, K=K, codes=codes)
    print(name, "eta2", np.round(et, 3), "R2add", round(R["R2_additive"], 4), "pairs",
          round(R["R2_additive_plus_all_pairs"], 4), "localmax", R["local_max_eta2_offcentre"],
          "asym", round(R["asymmetry_index"], 3))

# cross-model: r10.4.1 vs RNA004 (same K)
a = cache["DNA r10.4.1 (9-mer)"]; b = cache["RNA004 (9-mer)"]
res["cross_r1041_rna004"] = dict(
    pearson_mean=float(np.corrcoef(a["y"], b["y"])[0, 1]),
    mean_abs_diff=float(np.mean(np.abs(a["y"] - b["y"]))),
    eta2_max_abs_diff=float(np.max(np.abs(a["et"] - b["et"]))),
    effect_max_abs_diff=float(np.max(np.abs(a["eff"] - b["eff"]))),
    pearson_std=float(np.corrcoef(a["sd"], b["sd"])[0, 1]),
    pearson_dwell=float(np.corrcoef(a["dw"], b["dw"])[0, 1]))
# base-effect structure A vs C, G vs T
for name in MODELS:
    e = np.array(res[name]["marginal_effect"])
    res[name]["effect_sum_A_C_over_range"] = float(np.max(np.abs(e[:, 0] + e[:, 1])) / np.ptp(e))
    res[name]["effect_sum_G_T_over_range"] = float(np.max(np.abs(e[:, 2] + e[:, 3])) / np.ptp(e))
json.dump(res, open(f"{OUT}/pore_model_results.json", "w"), indent=1)

# ---------------- figures ----------------
names = list(MODELS)
# Fig 1: marginal effect heatmaps + eta2
fig, axes = plt.subplots(2, 4, figsize=(17, 7.5), gridspec_kw=dict(height_ratios=[1.3, 1]))
vmax = max(np.abs(cache[n]["eff"]).max() for n in names)
for k, n in enumerate(names):
    C = cache[n]; ax = axes[0, k]
    im = ax.imshow(C["eff"].T, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_yticks(range(4)); ax.set_yticklabels(list("ACGU") if "RNA" in n else list(BASES))
    ax.set_xticks(range(C["K"])); ax.set_xticklabels(range(1, C["K"] + 1))
    ax.set_title(n); ax.set_xlabel("k-mer position")
    for p in range(C["K"]):
        for bb in range(4):
            v = C["eff"][p, bb]
            if abs(v) > 0.15 * vmax:
                ax.text(p, bb, f"{v:.2f}", ha="center", va="center", fontsize=6.5 if C["K"] == 9 else 8,
                        color="white" if abs(v) > 0.6 * vmax else "black")
    ax = axes[1, k]
    ax.bar(range(1, C["K"] + 1), C["et"], color="#1f5aa6", label="η² (marginal)")
    er = np.nan_to_num(C["et_res"], nan=0)
    ax.plot(range(1, C["K"] + 1), np.where(np.isnan(C["et_res"]), np.nan, C["et_res"]), "o-", color="#d62839",
            label="η² of residual after\nremoving core positions")
    ax.set_xticks(range(1, C["K"] + 1)); ax.set_xlabel("k-mer position"); ax.set_ylabel("η² (current mean)")
    ax.set_ylim(0, max(0.6, C["et"].max() * 1.1))
    if k == 0:
        ax.legend(fontsize=7.5, frameon=False, loc="upper left")
fig.colorbar(im, ax=axes[0, :], shrink=0.8, label="marginal effect on current (model SD units)")
fig.suptitle("Base × position effects on k-mer current level", y=0.995)
plt.savefig(f"{IMG}/fig1_base_position_effects.png", dpi=150, bbox_inches="tight"); plt.close()

# Fig 2: ranked-current base-composition heatmap (Uncalled4-style)
fig, axes = plt.subplots(1, 4, figsize=(18, 4.8))
for k, n in enumerate(names):
    C = cache[n]; ax = axes[k]
    comp = C["comp"]; Kk = C["K"]
    img = np.ones((Kk, comp.shape[1], 3))
    dom = comp.argmax(2); mx = comp.max(2)
    inten = np.clip((mx - 0.25) / 0.5, 0, 1)
    cols = np.array([to_rgb(BASE_COLORS[bb]) for bb in BASES])
    for p in range(Kk):
        for j in range(comp.shape[1]):
            img[p, j] = 1 - inten[p, j] * (1 - cols[dom[p, j]])
    ax.imshow(img, aspect="auto", interpolation="nearest", extent=[0, 1, Kk + 0.5, 0.5])
    ax.set_yticks(range(1, Kk + 1)); ax.set_ylabel("k-mer position"); ax.set_xlabel("k-mer rank by current (low → high)")
    ax.set_title(n)
from matplotlib.patches import Patch
fig.legend(handles=[Patch(color=BASE_COLORS[bb], label=(bb if bb != "T" else "T/U")) for bb in BASES], loc="lower center",
           ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.06))
fig.suptitle("Most-enriched base per position across 64 current-rank bins (colour saturation = fraction above 0.25, full at 0.75)")
plt.savefig(f"{IMG}/fig2_ranked_current_composition.png", dpi=150, bbox_inches="tight"); plt.close()

# Fig 2b: detailed composition lines for dominant + selected positions
fig, axes = plt.subplots(4, 3, figsize=(14, 12), sharex=True, sharey=True)
for k, n in enumerate(names):
    C = cache[n]; c = C["c"]
    order = np.argsort(-C["et"])
    pos = [order[0], order[1], order[2]]
    for j, p in enumerate(pos):
        ax = axes[k, j]
        xs = (np.arange(C["comp"].shape[1]) + 0.5) / C["comp"].shape[1]
        for bb in range(4):
            ax.plot(xs, C["comp"][p, :, bb], color=BASE_COLORS[BASES[bb]], lw=1.8, label=BASES[bb])
        ax.axhline(0.25, color="k", lw=0.5, ls=":")
        ax.set_title(f"{n}: position {p + 1} (η²={C['et'][p]:.3f})", fontsize=9)
        if j == 0: ax.set_ylabel("base fraction in bin")
        if k == 3: ax.set_xlabel("current rank quantile")
axes[0, 0].legend(ncol=4, fontsize=8, frameon=False)
plt.tight_layout(); plt.savefig(f"{IMG}/figS1_composition_lines.png", dpi=130); plt.close()

# Fig 3: substitution profiles
fig, axes = plt.subplots(2, 4, figsize=(17, 7))
for k, n in enumerate(names):
    C = cache[n]; s = C["sub"]; P = np.arange(1, C["K"] + 1)
    for row, pre in enumerate(["", "z_"]):
        ax = axes[row, k]
        med = [s[p][pre + "median"] for p in range(C["K"])]
        ax.fill_between(P, [s[p][pre + "q10"] for p in range(C["K"])], [s[p][pre + "q90"] for p in range(C["K"])],
                        color="#1f5aa6", alpha=0.15, label="10–90%")
        ax.fill_between(P, [s[p][pre + "q25"] for p in range(C["K"])], [s[p][pre + "q75"] for p in range(C["K"])],
                        color="#1f5aa6", alpha=0.35, label="IQR")
        ax.plot(P, med, "o-", color="#1f5aa6", label="median")
        if row == 1: ax.axhline(1, color="#d62839", ls="--", lw=1, label="|Δμ| = σ")
        ax.set_xticks(P); ax.set_xlabel("substituted position")
        ax.set_ylabel("|Δμ| (model SD units)" if row == 0 else "|Δμ| / σ_kmer")
        if row == 0: ax.set_title(n)
        if k == 0: ax.legend(fontsize=8, frameon=False)
fig.suptitle("Single-base substitution profiles (all 3 substitutions at every position of every k-mer)")
plt.tight_layout(); plt.savefig(f"{IMG}/fig3_substitution_profiles.png", dpi=150); plt.close()

# Fig 4: pairwise interaction gain heatmaps
fig, axes = plt.subplots(1, 4, figsize=(18, 4.4))
for k, n in enumerate(names):
    C = cache[n]; ax = axes[k]
    P = np.nan_to_num(C["pair"]) * 100
    im = ax.imshow(P, cmap="magma_r", vmin=0)
    ax.set_xticks(range(C["K"])); ax.set_xticklabels(range(1, C["K"] + 1))
    ax.set_yticks(range(C["K"])); ax.set_yticklabels(range(1, C["K"] + 1))
    ax.set_title(f"{n}\nadditive R²={res[n]['R2_additive']:.4f}; +all pairs={res[n]['R2_additive_plus_all_pairs']:.4f}", fontsize=9)
    plt.colorbar(im, ax=ax, shrink=0.8, label="ΔR² from pair interaction (%)")
fig.suptitle("Pairwise (incl. non-adjacent) interaction R² gain over the additive model")
plt.tight_layout(); plt.savefig(f"{IMG}/fig4_pairwise_interactions.png", dpi=150); plt.close()

# Fig 5: window R2 + std/dwell eta2
fig, axes = plt.subplots(1, 3, figsize=(16, 4.3))
for n in names:
    w = res[n]["window_R2"]
    axes[0].plot(list(map(int, w)), [w[x]["R2"] for x in w], "o-", label=n)
    axes[1].plot(range(1, res[n]["K"] + 1), res[n]["eta2_std"], "o-", label=n)
    axes[2].plot(range(1, res[n]["K"] + 1), res[n]["eta2_dwell"], "o-", label=n)
axes[0].set_xlabel("window width (contiguous positions)"); axes[0].set_ylabel("best full-interaction R²")
axes[0].set_title("Effective context width (current mean)")
axes[1].set_title("Sequence dependence of current_std"); axes[1].set_ylabel("η²"); axes[1].set_xlabel("position")
axes[2].set_title("Sequence dependence of dwell_time"); axes[2].set_ylabel("η²"); axes[2].set_xlabel("position")
axes[0].legend(fontsize=8, frameon=False)
for ax in axes[1:]: ax.set_ylim(0, 0.01)
plt.tight_layout(); plt.savefig(f"{IMG}/fig5_context_std_dwell.png", dpi=150); plt.close()
print(json.dumps(res["cross_r1041_rna004"], indent=1))
