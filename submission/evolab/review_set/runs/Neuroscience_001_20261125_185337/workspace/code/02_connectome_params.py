"""Connectome overview + learned parameter analysis across the 50-model ensemble.
Outputs: outputs/connectome_*.csv, outputs/params_*.npz/csv, figures."""
import sys
sys.path.insert(0, "/workspace/code")
import numpy as np, pandas as pd, h5py, json
import fastsim as fs
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.stats import spearmanr, pearsonr

OUT = "/workspace/outputs"; IMG = "/workspace/report/images"
T = fs.topo()
net = fs.get_network()
types = T.cell_types
nt = len(types)
facts = dict(n_nodes=T.n_nodes, n_edges=T.n_edges, n_types=nt,
             n_hex_columns=int(T.input_index.shape[1]))

# ---------- type-level connectivity -------------------------------------------
fs.load_state(0)
P0 = fs.semantic_params()
keys_sc = fs.topo().param_keys["syn_count"]  # (src, tgt, du, dv)
keys_e = fs.topo().param_keys["syn_strength"]  # (src, tgt)
sc = pd.DataFrame(keys_sc, columns=["src", "tgt", "du", "dv"])
sc["n_syn"] = P0["syn_count"]
et = pd.DataFrame(keys_e, columns=["src", "tgt"])
et["sign"] = P0["sign"]
facts["n_edge_types"] = len(et)
facts["n_filter_offsets"] = len(sc)
facts["n_free_params"] = int(len(P0["bias"]) + len(P0["time_const"]) + len(P0["syn_strength"]))
tot = sc.groupby(["src", "tgt"]).n_syn.sum().reset_index()
tot = tot.merge(et, on=["src", "tgt"])
tot.to_csv(f"{OUT}/connectome_type_pairs.csv", index=False)
M = np.zeros((nt, nt))  # rows = presynaptic, cols = postsynaptic
for _, r in tot.iterrows():
    M[types.index(r.src), types.index(r.tgt)] = r.n_syn * r.sign
pd.DataFrame(M, index=types, columns=types).to_csv(f"{OUT}/connectome_type_matrix_pre_rows.csv")
# out-degree / sign classes
out_sign = tot.groupby("src").sign.agg(lambda s: "E" if (s > 0).all() else ("I" if (s < 0).all() else "mixed"))
facts["types_without_outputs"] = [t for t in types if t not in out_sign.index]
facts["n_excitatory_types"] = int((out_sign == "E").sum()); facts["n_inhibitory_types"] = int((out_sign == "I").sum())
out_sign.to_csv(f"{OUT}/connectome_output_sign_class.csv")

fig, ax = plt.subplots(figsize=(11, 10))
A = np.abs(M); A[A == 0] = np.nan
im = ax.imshow(np.where(M > 0, A, np.nan), cmap="Reds", norm=LogNorm(1e-1, 2e2))
im2 = ax.imshow(np.where(M < 0, A, np.nan), cmap="Blues", norm=LogNorm(1e-1, 2e2))
ax.set_xticks(range(nt)); ax.set_xticklabels(types, rotation=90, fontsize=6)
ax.set_yticks(range(nt)); ax.set_yticklabels(types, fontsize=6)
ax.set_xlabel("postsynaptic cell type (target)"); ax.set_ylabel("presynaptic cell type (source)")
ax.set_title(f"Connectome used by the DMN: {nt} cell types, {len(et)} connected type pairs\n"
             "(colour = total synapses per postsynaptic cell; red excitatory, blue inhibitory)")
c1 = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.01); c1.set_label("excitatory #syn")
c2 = plt.colorbar(im2, ax=ax, fraction=0.03, pad=0.06); c2.set_label("inhibitory #syn")
plt.tight_layout(); plt.savefig(f"{IMG}/fig_connectome_matrix.png", dpi=150); plt.close()

# ---------- spatial filters onto T4c / T5c -------------------------------------
def hex_xy(u, v):
    return np.sqrt(3) * (u + v / 2), 1.5 * v


fig, axs = plt.subplots(2, 6, figsize=(16, 5.6))
for row, tgt in enumerate(["T4c", "T5c"]):
    sub = sc[sc.tgt == tgt]
    srcs = sub.groupby("src").n_syn.sum().sort_values(ascending=False).index[:6]
    for k, s in enumerate(srcs):
        ax = axs[row, k]
        d = sub[sub.src == s]
        x, y = hex_xy(d.du.values, d.dv.values)
        sgn = et[(et.src == s) & (et.tgt == tgt)].sign.values[0]
        sca = ax.scatter(x, y, c=d.n_syn.values, cmap="Reds" if sgn > 0 else "Blues", s=170, marker="h",
                         vmin=0, edgecolor="k", lw=.3)
        ax.set_title(f"{s} → {tgt} ({'+' if sgn>0 else '−'}, Σ={d.n_syn.sum():.1f})", fontsize=9)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlim(-6, 6); ax.set_ylim(-5, 5)
        plt.colorbar(sca, ax=ax, fraction=.04)
        ax.plot(0, 0, "k+")
plt.suptitle("Connectome spatial input filters (synapses per offset (du,dv)) onto T4c and T5c; '+' marks the home column")
plt.tight_layout(); plt.savefig(f"{IMG}/fig_T4T5_filters.png", dpi=150); plt.close()

# ---------- learned parameters across ensemble --------------------------------
losses = np.array([h5py.File(f"{fs.DATA_DIR}/{i:03d}/validation_loss.h5")["data"][()] for i in range(50)])
np.save(f"{OUT}/validation_losses.npy", losses)
B, TAU, S = [], [], []
for i in range(50):
    p = fs.model_params(i)
    B.append(p["bias"]); TAU.append(p["time_const"]); S.append(p["syn_strength"])
B, TAU, S = map(np.array, (B, TAU, S))
np.savez(f"{OUT}/params_ensemble.npz", bias=B, tau=TAU, strength=S, losses=losses,
         node_types=np.array(fs.topo().param_keys["bias"]), edge_src=et.src.values, edge_tgt=et.tgt.values)
node_types_p = fs.topo().param_keys["bias"]
tau_min = TAU.min()
facts["tau_min"] = float(tau_min); facts["frac_tau_le_20.5ms"] = float(np.mean(TAU < 0.0205))
facts["note_tau"] = "no explicit clamp; training dt=0.02 s and effective tau=max(tau,dt) => no gradient below 20 ms"
facts["tau_init"] = 0.05
facts["loss_min"], facts["loss_max"] = float(losses.min()), float(losses.max())

# between-model consistency of learned strengths (pairwise Spearman of log strengths)
logS = np.log10(S + 1e-8)
eff = S * np.array([tot.set_index(["src","tgt"]).loc[(a, b), "n_syn"] for a, b in zip(et.src, et.tgt)])[None]
r_pairs = [spearmanr(logS[i], logS[j])[0] for i in range(50) for j in range(i + 1, 50)]
r_bias = [pearsonr(B[i], B[j])[0] for i in range(50) for j in range(i + 1, 50)]
r_tau = [spearmanr(TAU[i], TAU[j])[0] for i in range(50) for j in range(i + 1, 50)]
facts["median_pairwise_spearman_log_strength"] = float(np.median(r_pairs))
facts["median_pairwise_pearson_bias"] = float(np.median(r_bias))
facts["median_pairwise_spearman_tau"] = float(np.nanmedian(r_tau))
# strength change relative to init (0.01/<N>)
init_s = 0.01 / sc.groupby(["src", "tgt"], sort=False).n_syn.mean().reindex(pd.MultiIndex.from_frame(et[["src", "tgt"]])).values
ratio = S / init_s[None]
facts["median_strength_over_init"] = float(np.median(ratio))
facts["frac_strength_zero"] = float(np.mean(S <= 1e-8))
json.dump(facts, open(f"{OUT}/02_connectome_facts.json", "w"), indent=1)
print(json.dumps(facts, indent=1))

fig, axs = plt.subplots(2, 2, figsize=(16, 8), gridspec_kw=dict(height_ratios=[1, 1]))
order = np.argsort(np.median(B, 0))
ax = axs[0, 0]
ax.boxplot(B[:, order], showfliers=False, widths=.6)
ax.plot(np.arange(1, nt + 1), B[0, order], "r.", label="best model (000)")
ax.set_xticks(range(1, nt + 1)); ax.set_xticklabels(np.array(node_types_p)[order], rotation=90, fontsize=6)
ax.axhline(0.5, ls=":", c="gray", label="init mean 0.5"); ax.set_ylabel("resting potential (bias)"); ax.legend(fontsize=7)
ax.set_title("Learned resting potentials (50 models)")
ax = axs[0, 1]
order2 = np.argsort(np.median(TAU, 0))
ax.boxplot(TAU[:, order2] * 1000, showfliers=False, widths=.6)
ax.plot(np.arange(1, nt + 1), TAU[0, order2] * 1000, "r.")
ax.set_xticks(range(1, nt + 1)); ax.set_xticklabels(np.array(node_types_p)[order2], rotation=90, fontsize=6)
ax.axhline(50, ls=":", c="gray"); ax.set_ylabel("time constant (ms)")
ax.axhline(20, ls="--", c="r", lw=.8)
ax.set_title(f"Learned time constants ({100*facts['frac_tau_le_20.5ms']:.0f}% ≤ 20 ms = training dt, effective floor)")
ax = axs[1, 0]
ax.hist(np.log10(ratio.ravel() + 1e-6), bins=80, color="gray")
ax.set_xlabel("log10(learned strength / initial strength)"); ax.set_ylabel("# (model, edge type)")
ax.set_title("Unit synaptic strengths: change from initialisation 0.01/<N>")
ax = axs[1, 1]
ax.hist(r_pairs, bins=30, alpha=.6, label="log strength (Spearman)")
ax.hist(r_bias, bins=30, alpha=.6, label="bias (Pearson)")
ax.hist(r_tau, bins=30, alpha=.6, label="tau (Spearman)")
ax.set_xlabel("between-model correlation (1225 model pairs)"); ax.legend(); ax.set_title("Parameter consistency across ensemble")
plt.tight_layout(); plt.savefig(f"{IMG}/fig_learned_params.png", dpi=150); plt.close()
