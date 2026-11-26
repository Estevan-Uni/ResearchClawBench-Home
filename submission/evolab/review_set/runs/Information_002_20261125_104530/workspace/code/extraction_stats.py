"""Placeholder-extraction accuracy, inter-annotator agreement, automatic scorer, and step-rubric analysis
for the 2111.01152 template chain.  Also contains the agent's rubric grading of auto.md and of its own execution."""
import re, json, difflib, itertools, csv
import numpy as np
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

Y = yaml.safe_load(open("/workspace/data/2111.01152/2111.01152.yaml"))
tasks = [t for t in Y if "task" in t]
assert len(tasks) == 16, len(tasks)
GR = ["Haining", "Will", "Yasaman"]
SHORT = {"Haining": "H", "Will": "W", "Yasaman": "Y"}


def ptype(name):
    n = name.lower()
    if "|" in n:
        return "choice"
    if " " in n.strip():               # whole-sentence switch (true/false)
        return "choice"
    if "definition" in n:
        return "definitions"
    if n.endswith("symbol") or n.endswith("_op") or n in ("index_1", "index_2", "op", "hole_op", "particle_op", "k_symbol", "r_symbol", "real_variable", "system", "variable", "order", "ham_op"):
        return "symbol"
    return "expression"


rows = []
for ti, t in enumerate(tasks):
    for name, p in (t.get("placeholder") or {}).items():
        if not isinstance(p, dict):
            continue
        sc = p.get("score")
        rec = dict(step=ti + 1, task=t["task"], name=name, type=ptype(name), LLM=p.get("LLM"), human=p.get("human"))
        for g in GR:
            v = sc.get(g) if isinstance(sc, dict) else None
            rec[g] = v if isinstance(v, (int, float)) else np.nan   # '(?)' and missing -> NaN
        rows.append(rec)
n_all = len(rows)
graded = [r for r in rows if any(not np.isnan(r[g]) for g in GR)]
for r in graded:
    vals = [r[g] for g in GR if not np.isnan(r[g])]
    r["mean"] = float(np.mean(vals))

stats = {"n_placeholders_total": n_all, "n_graded": len(graded),
         "n_missing_cells('(?)' or absent)": int(sum(np.isnan(r[g]) for r in graded for g in GR))}
M = np.array([[r[g] for g in GR] for r in graded])
stats["normalized_accuracy_overall(mean/2)"] = float(np.nanmean(M) / 2)
stats["per_grader_norm_acc"] = {g: float(np.nanmean(M[:, i]) / 2) for i, g in enumerate(GR)}
bytype = {}
for ty in ("choice", "symbol", "expression", "definitions"):
    sel = [r["mean"] for r in graded if r["type"] == ty]
    bytype[ty] = dict(n=len(sel), norm_acc=float(np.mean(sel) / 2) if sel else None)
stats["by_type"] = bytype
bystep = {}
for s in range(1, 17):
    sel = [r["mean"] for r in graded if r["step"] == s]
    bystep[s] = dict(n=len(sel), norm_acc=float(np.mean(sel) / 2) if sel else None)
stats["by_step"] = bystep
# choice placeholders of convention type
conv = [r for r in graded if r["name"] in ("real|momentum", "single-particle|second-quantized", "electrons|holes")]
stats["convention_choices_norm_acc"] = float(np.mean([r["mean"] for r in conv]) / 2)
stats["convention_choices_n"] = len(conv)


# ---------------- agreement ----------------
def cohen(a, b, weighted=False):
    cats = [0, 1, 2]
    a = np.asarray(a, int); b = np.asarray(b, int)
    O = np.zeros((3, 3))
    for x, y in zip(a, b):
        O[x, y] += 1
    O /= O.sum()
    E = np.outer(O.sum(1), O.sum(0))
    Wt = np.array([[abs(i - j) / 2 if weighted else float(i != j) for j in cats] for i in cats])
    return 1 - (Wt * O).sum() / (Wt * E).sum()


def kripp_alpha(data, metric="interval"):
    """data: units x coders with NaN for missing"""
    vals = []
    pairs_num = 0.0
    units = [u[~np.isnan(u)] for u in data]
    units = [u for u in units if len(u) >= 2]
    allv = np.concatenate(units)
    d = (lambda x, y: (x - y) ** 2) if metric == "interval" else (lambda x, y: float(x != y))
    Do = 0.0; n = len(allv)
    for u in units:
        m = len(u)
        Do += sum(d(x, y) for x, y in itertools.permutations(u, 2)) / (m - 1)
    Do /= n
    De = sum(d(x, y) for x, y in itertools.permutations(allv, 2)) / (n * (n - 1))
    return 1 - Do / De


agree = {}
for g1, g2 in itertools.combinations(GR, 2):
    i, j = GR.index(g1), GR.index(g2)
    m = ~np.isnan(M[:, i]) & ~np.isnan(M[:, j])
    agree[f"{SHORT[g1]}-{SHORT[g2]}"] = dict(n=int(m.sum()), kappa=float(cohen(M[m, i], M[m, j])),
                                           kappa_w=float(cohen(M[m, i], M[m, j], True)),
                                           exact_agreement=float(np.mean(M[m, i] == M[m, j])))
agree["krippendorff_alpha_interval"] = float(kripp_alpha(M, "interval"))
agree["krippendorff_alpha_nominal"] = float(kripp_alpha(M, "nominal"))
stats["agreement"] = agree


# ---------------- automatic scorer ----------------
def norm_tex(s):
    s = str(s or "")
    s = re.sub(r"\\(bm|mathbf|mathfrak|text|mathrm|left|right|,|;|!|quad)", "", s)
    s = s.replace("$", "").replace("{", "").replace("}", "").replace(" ", "").replace("\\", "")
    return s.lower()


def auto_score(r):
    llm, ref = r["LLM"], r["human"] if r["human"] not in (None, "") else r["LLM"]
    if llm in (None, ""):
        return 0, 0.0
    if r["type"] == "choice":
        return (2 if str(llm).strip().lower() == str(ref).strip().lower() else 0), float(str(llm).strip().lower() == str(ref).strip().lower())
    ratio = difflib.SequenceMatcher(None, norm_tex(llm), norm_tex(ref)).ratio()
    return (2 if ratio > 0.8 else 1 if ratio > 0.4 else 0), ratio


for r in graded:
    r["auto"], r["auto_ratio"] = auto_score(r)
auto = np.array([r["auto"] for r in graded]); mean = np.array([r["mean"] for r in graded])
scorer = {"spearman_vs_mean": float(spearmanr(auto, mean).correlation),
          "auto_norm_acc": float(auto.mean() / 2)}
for i, g in enumerate(GR):
    m = ~np.isnan(M[:, i])
    scorer[f"spearman_vs_{SHORT[g]}"] = float(spearmanr(auto[m], M[m, i]).correlation)
hum_given = np.array([r["human"] not in (None, "") for r in graded])
scorer["n_with_explicit_reference"] = int(hum_given.sum())
scorer["spearman_vs_mean_only_explicit_reference"] = float(spearmanr(auto[hum_given], mean[hum_given]).correlation)
scorer["exact_bin_agreement_with_rounded_mean"] = float(np.mean(auto == np.rint(mean)))
stats["auto_scorer"] = scorer

with open("/workspace/outputs/placeholder_table.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["step", "name", "type", "H", "W", "Y", "mean", "auto", "auto_ratio"])
    for r in graded:
        w.writerow([r["step"], r["name"], r["type"]] + [r[g] for g in GR] + [round(r["mean"], 3), r["auto"], round(r["auto_ratio"], 3)])

# ---------------- step rubric (GPT-4 run, human graded) ----------------
ITEMS = ["in_paper", "prompt_quality", "follow_instructions", "physics_logic", "math_derivation", "final_answer_accuracy"]
R_gpt4 = np.array([[t["score"][k] for k in ITEMS] for t in tasks], float)
stats["gpt4_rubric_mean_by_item"] = {k: float(R_gpt4[:, i].mean()) for i, k in enumerate(ITEMS)}
stats["gpt4_steps_with_final_answer<2"] = [i + 1 for i in range(16) if R_gpt4[i, 5] < 2]

# ---------------- agent rubric grading of auto.md and of agent's own execution ----------------
EXE = ["follow_instructions", "physics_logic", "math_derivation", "final_answer_accuracy"]
AUTO = {  # step: (scores, reason)
    1: ((1, 1, 1, 0), "Writes full H_tau incl. potentials instead of the diagonal kinetic matrix; +hbar^2 sign (electron-like) and bottom layer also shifted by tau*kappa."),
    2: ((0, 1, 1, 0), "Prompt says only E_{t,+-K} are shifted and dispersion is for holes; answer keeps +hbar^2(k-tau kappa)^2/2m in BOTH layers (two errors), plus potentials."),
    3: ((2, 2, 2, 2), "Correct Hermitian 2x2 blocks Delta_b, Delta_T, Delta_T^*, Delta_t+V_zt; block-diagonal 4x4."),
    4: ((2, 2, 2, 2), "All four entries (incl. -K block -w(1+omega^-1 e^{-ig2 r}+...)) match gold; Delta_t=V_zt used without double counting."),
    5: ((2, 1, 1, 1), "Correct int Psi^dag H Psi with 4-spinor, but kinetic written -hbar^2(nabla+i tau kappa)^2/2m = +hbar^2(k+tau kappa)^2/2m in both layers (sign and shift wrong)."),
    6: ((1, 2, 1, 1), "Keeps sum over tau (better than GPT-4) but leaves 2-spinor matrix product instead of explicit sum_{l1,l2}; kinetic again +hbar^2(-i nabla - tau kappa)^2/2m in both layers."),
    7: ((2, 2, 2, 2), "Correct FT with 1/sqrt(V), defines h_{k1 l1,k2 l2} = (1/V) int e^{-i(k1-k2)r}[H_tau]_{l1l2}, Bloch selection rule, sum over tau retained."),
    8: ((2, 2, 2, 2), "b h b^dag exactly as gold."),
    9: ((2, 2, 2, 2), "Tr h - sum b^dag h^* b, identical to gold (h^T = h^* for Hermitian h); minor index typo delta_{tau,tau'}."),
    10: ((2, 2, 2, 2), "Correct 1/2A, V(|k_a-k_d|), momentum delta, operator order."),
    11: ((1, 2, 1, 1), "Four quadratic Wick terms correct in sign/order, but the two c-number <><> terms of EXAMPLE 1 were dropped (template asks to follow the example)."),
    12: ((2, 2, 2, 2), "Four quadratic terms, correct."),
    13: ((2, 2, 2, 2), "Explicit relabelling; V(|k2-k3|)=V(|k1-k4|) justified; 1/A two-term result equals gold."),
    14: ((2, 2, 2, 2), "Inserts delta_{k1k4}, then delta_{k2+q1+q2,k3+q3+q4} -> delta_{k2k3} delta_{q1+q2,q3+q4} with BZ argument; final equals gold."),
    15: ((2, 2, 2, 2), "Same reduction for Fock; V(|k1+q1-k2-q4|) retained; equals gold."),
    16: ((2, 2, 2, 2), "Correct combined H+F; performs an unnecessary round-trip relabelling that returns to the gold form."),
}
AGENT = {s: ((2, 2, 2, 2), "") for s in range(1, 17)}
AGENT_REASON = {
    1: "diag(E_{+K,b},E_{+K,t},E_{-K,b},E_{-K,t}) in requested order; only top layer shifted (stated in step 2).",
    2: "E_b=-hbar^2k^2/2m_b, E_{t,+-K}=-hbar^2(k-+kappa)^2/2m_t (hole-like sign), k=-i d_r.",
    3: "4x4 potential with Delta_b, Delta_T,+-K, Hermitian conjugates, Delta_t.",
    4: "Explicit Delta_b, Delta_T,+-K; states Delta_t=V_zt (gold) vs Delta_t=0 + V_zt (paper), counted once.",
    5: "int d^2r Psi^dag H^0 Psi with explicit 4x4 H^0 and spinors Psi_+-.",
    6: "sum_tau sum_{l1,l2} int psi^dag H^0 psi, zero off-diagonal valley blocks removed.",
    7: "FT + explicit plane-wave matrix elements (kinetic, Delta_b couplings V_b e^{+-i psi}, Delta_T couplings) verified by FFT.",
    8: "b h b^dag.", 9: "Tr h - sum b^dag h^T b (verified in Fock space).", 10: "1/2A form, V(q)=2 pi e^2 tanh(qd)/(eps q).",
    11: "Full 6-term Wick expansion (verified vs <Phi|H|Phi>).", 12: "Four quadratic terms.", 13: "1/A two-term form (verified).",
    14: "Reduced Hartree with V(|q1-q4|), delta_{q1+q2,q3+q4} (verified).", 15: "Reduced Fock with V(|k1+q1-k2-q4|) (verified).",
    16: "Combined H^HF with explicit Sigma^H, Sigma^F(k) and total energy; implemented and brute-force checked."}
R_auto = np.array([AUTO[s][0] for s in range(1, 17)], float)
R_agent = np.array([AGENT[s][0] for s in range(1, 17)], float)
R_gpt4_exe = R_gpt4[:, 2:]
stats["execution_item_means"] = {
    "GPT-4 (human-graded)": {k: float(R_gpt4_exe[:, i].mean()) for i, k in enumerate(EXE)},
    "auto.md (agent-graded)": {k: float(R_auto[:, i].mean()) for i, k in enumerate(EXE)},
    "agent execution (self-graded)": {k: float(R_agent[:, i].mean()) for i, k in enumerate(EXE)}}
stats["phase_means_final_answer"] = {}
for lab, R in (("GPT-4", R_gpt4_exe), ("auto.md", R_auto), ("agent", R_agent)):
    stats["phase_means_final_answer"][lab] = {"steps1-6": float(R[:6, 3].mean()), "steps7-11": float(R[6:11, 3].mean()),
                                               "steps12-16": float(R[11:, 3].mean())}
with open("/workspace/outputs/rubric_grades.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["step", "task", "run"] + EXE + ["reason"])
    for s in range(1, 17):
        w.writerow([s, tasks[s - 1]["task"], "GPT-4(human)"] + list(R_gpt4_exe[s - 1].astype(int)) + [str(tasks[s - 1].get("score"))])
        w.writerow([s, tasks[s - 1]["task"], "auto.md(agent-graded)"] + list(AUTO[s][0]) + [AUTO[s][1]])
        w.writerow([s, tasks[s - 1]["task"], "agent(self-graded)"] + list(AGENT[s][0]) + [AGENT_REASON[s]])
json.dump(stats, open("/workspace/outputs/extraction_stats.json", "w"), indent=1)
print(json.dumps(stats, indent=1))

# ---------------- figures ----------------
plt.rcParams.update({"font.size": 9})
fig, axs = plt.subplots(1, 3, figsize=(14, 4.0), gridspec_kw={"width_ratios": [2.2, 1, 1.2]})
ax = axs[0]
steps = np.arange(1, 17)
acc = [bystep[s]["norm_acc"] if bystep[s]["norm_acc"] is not None else np.nan for s in steps]
ns = [bystep[s]["n"] for s in steps]
ax.bar(steps, acc, color="#4C72B0")
for s, a, n_ in zip(steps, acc, ns):
    ax.text(s, (0 if np.isnan(a) else a) + 0.02, f"n={n_}", ha="center", fontsize=7)
ax.set_xticks(steps); ax.set_ylim(0, 1.1); ax.set_xlabel("template step (1-16)"); ax.set_ylabel("normalized extraction accuracy")
ax.set_title("GPT-4 placeholder extraction accuracy by step (mean of 3 graders / 2)")
ax = axs[1]
tys = list(bytype); ax.bar(tys, [bytype[t]["norm_acc"] for t in tys], color=["#DD8452", "#55A868", "#C44E52", "#8172B3"])
for i, t in enumerate(tys):
    ax.text(i, bytype[t]["norm_acc"] + 0.02, f"n={bytype[t]['n']}", ha="center", fontsize=8)
ax.set_ylim(0, 1.1); ax.set_title("by placeholder type")
ax = axs[2]
labs = ["H-W", "H-Y", "W-Y"]
x = np.arange(3)
ax.bar(x - 0.2, [agree[l]["kappa"] for l in labs], 0.4, label="Cohen kappa")
ax.bar(x + 0.2, [agree[l]["kappa_w"] for l in labs], 0.4, label="linear-weighted kappa")
ax.axhline(agree["krippendorff_alpha_interval"], color="k", ls="--", lw=1, label=f"Krippendorff alpha={agree['krippendorff_alpha_interval']:.2f}")
ax.set_xticks(x); ax.set_xticklabels(labs); ax.set_ylim(0, 1); ax.legend(fontsize=7); ax.set_title("inter-grader agreement")
plt.tight_layout(); plt.savefig("/workspace/report/images/extraction_accuracy.png", dpi=150); plt.close()

# rubric heatmap
fig, axs = plt.subplots(1, 3, figsize=(13, 5.2), sharey=True)
names = [f"{s}. {tasks[s-1]['task'][:38]}" for s in range(1, 17)]
for ax, R, title in zip(axs, (R_gpt4_exe, R_auto, R_agent), ("GPT-4 (human-graded, yaml)", "auto.md run (agent-graded)", "agent's own execution (self-graded)")):
    im = ax.imshow(R, cmap="RdYlGn", vmin=0, vmax=2, aspect="auto")
    for i in range(16):
        for j in range(4):
            ax.text(j, i, int(R[i, j]), ha="center", va="center", fontsize=8)
    ax.set_xticks(range(4)); ax.set_xticklabels(["follow", "physics", "math", "final"], rotation=0)
    ax.set_title(title, fontsize=9)
axs[0].set_yticks(range(16)); axs[0].set_yticklabels(names, fontsize=7)
plt.tight_layout(); plt.savefig("/workspace/report/images/rubric_heatmap.png", dpi=150); plt.close()

# auto scorer scatter
fig, ax = plt.subplots(figsize=(4.6, 3.8))
jit = np.random.default_rng(0).normal(scale=0.06, size=len(auto))
ax.scatter(mean, auto + jit, c=[{"choice": 0, "symbol": 1, "expression": 2, "definitions": 3}[r["type"]] for r in graded], cmap="tab10", s=18)
ax.set_xlabel("human mean score"); ax.set_ylabel("automatic score (jittered)")
ax.set_title(f"auto scorer: Spearman rho={scorer['spearman_vs_mean']:.2f}\n(vs H {scorer['spearman_vs_H']:.2f}, W {scorer['spearman_vs_W']:.2f}, Y {scorer['spearman_vs_Y']:.2f})", fontsize=9)
plt.tight_layout(); plt.savefig("/workspace/report/images/auto_scorer.png", dpi=150); plt.close()
