"""Step 8: validate HADDOCK3 mutational scoring against SKEMPI 2.0 (barnase-barstar).

Inputs : outputs/skempi_1brs.csv, outputs/mutscan/<template>_scores.csv,
         outputs/alascan/run_xtal/2_alascan/scan_emscoring_1.tsv, outputs/interface_residues.csv
Outputs: outputs/validation/*.csv (+ printed statistics)
"""
import os, sys, json
import numpy as np, pandas as pd
from scipy import stats


class LinearRegression:
    """Minimal OLS with intercept (numpy)."""
    def fit(self, X, y):
        A = np.c_[np.ones(len(X)), X]
        beta = np.linalg.lstsq(A, y, rcond=None)[0]
        self.intercept_, self.coef_ = beta[0], beta[1:]
        return self

    def predict(self, X):
        return self.intercept_ + X @ self.coef_


def loo_predict(X, y):
    out = np.empty(len(y))
    for i in range(len(y)):
        m = np.ones(len(y), bool); m[i] = False
        out[i] = LinearRegression().fit(X[m], y[m]).predict(X[i:i + 1])[0]
    return out


def roc_auc_score(labels, scores):
    labels, scores = np.asarray(labels, bool), np.asarray(scores, float)
    u = stats.mannwhitneyu(scores[labels], scores[~labels]).statistic
    return u / (labels.sum() * (~labels).sum())

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = f"{ROOT}/outputs/validation"
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(0)


def boot_ci(x, y, f, n=2000):
    x, y = np.asarray(x), np.asarray(y)
    v = []
    for _ in range(n):
        i = rng.integers(0, len(x), len(x))
        if np.std(x[i]) > 0 and np.std(y[i]) > 0:
            v.append(f(x[i], y[i]))
    return np.percentile(v, [2.5, 97.5])


def corr(x, y):
    r = stats.pearsonr(x, y)
    rho = stats.spearmanr(x, y)
    lo, hi = boot_ci(x, y, lambda a, b: stats.pearsonr(a, b)[0])
    slo, shi = boot_ci(x, y, lambda a, b: stats.spearmanr(a, b)[0])
    return dict(n=len(x), pearson=r[0], pearson_p=r[1], pearson_lo=lo, pearson_hi=hi,
                spearman=rho[0], spearman_p=rho[1], spearman_lo=slo, spearman_hi=shi)


# ---------------- experimental ddG per unique mutation ----------------
sk = pd.read_csv(f"{ROOT}/outputs/skempi_1brs.csv")
exp_all = sk.groupby("mutations").agg(ddG=("ddG", "mean"), ddG_sd=("ddG", "std"), n_meas=("ddG", "size"),
                                      n_mut=("n_mut", "first"), is_ala=("is_ala", "first")).reset_index()
exp_std = sk[sk.salt == "standard"].groupby("mutations").ddG.mean().rename("ddG_stdsalt")
exp_all = exp_all.join(exp_std, on="mutations")
# experimental reproducibility: pairs of independent measurements of the same mutation
rep = sk.groupby("mutations").ddG.agg(list)
pairs = [(v[0], v[1]) for v in rep if len(v) >= 2]
rep_r = stats.pearsonr(*zip(*pairs))[0] if len(pairs) > 2 else np.nan

iface = pd.read_csv(f"{ROOT}/outputs/interface_residues.csv")
iface["key"] = iface.chain + iface.resid.astype(str)


def struct_feats(m):
    ks = [x[1] + x[2:-1] for x in m.split(",")]
    sub = iface.set_index("key").loc[ks]
    return pd.Series(dict(bsa_res=sub.bsa.sum(), ncont_res=sub.n_partner_res.sum()))


exp_all = pd.concat([exp_all, exp_all.mutations.apply(struct_feats)], axis=1)


def analyse(template):
    sc = pd.read_csv(f"{ROOT}/outputs/mutscan/{template}_scores.csv")
    df = exp_all.merge(sc.rename(columns={"label": "mutations"}), on="mutations")
    df.to_csv(f"{OUT}/{template}_merged.csv", index=False)
    res = []
    subsets = {"all": df, "single": df[df.n_mut == 1], "double": df[df.n_mut == 2],
               "single_ala": df[(df.n_mut == 1) & df.is_ala]}
    preds = {"HADDOCK score": "d_score", "vdW": "d_vdw", "elec": "d_elec", "desolv": "d_desolv",
             "-dBSA": "d_bsa", "residue BSA": "bsa_res", "residue contacts": "ncont_res"}
    for sname, sub in subsets.items():
        for pname, col in preds.items():
            x = sub[col] * (-1 if pname == "-dBSA" else 1)
            res.append(dict(template=template, subset=sname, predictor=pname, **corr(x, sub.ddG)))
    # re-weighted HADDOCK components with leave-one-mutation-out CV (all mutations)
    X = df[["d_vdw", "d_elec", "d_desolv"]].values
    y = df.ddG.values
    cvp = loo_predict(X, y)
    res.append(dict(template=template, subset="all", predictor="re-weighted (LOO-CV)", **corr(cvp, y)))
    lr = LinearRegression().fit(X, y)
    df["pred_cv"] = cvp
    # hotspot classification (single mutants, ddG >= 2 kcal/mol)
    s = df[df.n_mut == 1]
    auc = {p: roc_auc_score(s.ddG >= 2, s[c] * (-1 if p == "-dBSA" else 1)) for p, c in preds.items()}
    return df, pd.DataFrame(res), lr, auc


if __name__ == "__main__":
    templates = sys.argv[1:] or ["xtal"]
    allres, aucs, weights = [], {}, {}
    for t in templates:
        df, res, lr, auc = analyse(t)
        df.to_csv(f"{OUT}/{t}_merged.csv", index=False)
        allres.append(res)
        aucs[t] = auc
        weights[t] = dict(intercept=lr.intercept_, w_vdw=lr.coef_[0], w_elec=lr.coef_[1], w_desolv=lr.coef_[2])
    res = pd.concat(allres)
    res.to_csv(f"{OUT}/correlations.csv", index=False)
    pd.DataFrame(aucs).to_csv(f"{OUT}/hotspot_auc.csv")
    json.dump(dict(weights=weights, replicate_pairs=len(pairs), replicate_pearson=rep_r), open(f"{OUT}/misc.json", "w"), indent=1)
    pd.set_option("display.width", 250)
    print(res.round(3).to_string())
    print(pd.DataFrame(aucs).round(3))
    print(weights, "replicate r=%.2f (n=%d)" % (rep_r, len(pairs)))

    # ---------- alascan module (crystal) vs experimental Ala singles ----------
    al = pd.read_csv(f"{ROOT}/outputs/alascan/run_xtal/2_alascan/scan_emscoring_1.tsv", sep="\t", comment="#")
    al["mutations"] = al.ori_resname.map(lambda r: {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V"}[r]) + al.chain + al.res.astype(str) + "A"
    al.to_csv(f"{OUT}/alascan_xtal.csv", index=False)
    m = al.merge(exp_all[(exp_all.n_mut == 1) & exp_all.is_ala], on="mutations")
    m.to_csv(f"{OUT}/alascan_vs_skempi.csv", index=False)
    print("alascan module vs SKEMPI Ala singles:", {k: round(v, 3) if isinstance(v, float) else v for k, v in corr(-m.delta_score, m.ddG).items()})
    print(m[["mutations", "delta_score", "z_score", "ddG"]].sort_values("ddG").to_string())

    # ---------- double-mutant cycles (coupling energies) ----------
    df = pd.read_csv(f"{OUT}/xtal_merged.csv").set_index("mutations")
    from Bio.PDB import PDBParser
    st = PDBParser(QUIET=True).get_structure("x", f"{ROOT}/outputs/docking/inputs/ref_1brs.pdb")[0]

    def mindist(a, b):
        ra, rb = st[a[1]][int(a[2:-1])], st[b[1]][int(b[2:-1])]
        return min(np.linalg.norm(x.coord - y.coord) for x in ra for y in rb)

    rows = []
    for mut in df.index[df.n_mut == 2]:
        a, b = mut.split(",")
        if a in df.index and b in df.index:
            rows.append(dict(double=mut, a=a, b=b, dist=mindist(a, b),
                             coupling_exp=df.loc[mut, "ddG"] - df.loc[a, "ddG"] - df.loc[b, "ddG"],
                             coupling_pred=df.loc[mut, "d_score"] - df.loc[a, "d_score"] - df.loc[b, "d_score"],
                             additive_exp=df.loc[a, "ddG"] + df.loc[b, "ddG"], ddG_double=df.loc[mut, "ddG"],
                             additive_pred=df.loc[a, "d_score"] + df.loc[b, "d_score"], pred_double=df.loc[mut, "d_score"]))
    cyc = pd.DataFrame(rows)
    cyc.to_csv(f"{OUT}/double_mutant_cycles.csv", index=False)
    print("double-mutant cycles:", len(cyc))
    print("coupling exp vs pred:", corr(cyc.coupling_pred, cyc.coupling_exp))
    print("coupling exp vs distance (spearman):", stats.spearmanr(cyc.dist, cyc.coupling_exp))
    print("coupling pred vs distance (spearman):", stats.spearmanr(cyc.dist, cyc.coupling_pred))
    print("exp: double vs additive r=%.2f" % stats.pearsonr(cyc.additive_exp, cyc.ddG_double)[0])
