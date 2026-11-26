"""Input audit: shapes, NaNs, ranges, completeness, ID alignment, absent metadata."""
import os, json
import numpy as np, pandas as pd
from common import *

rep = {}
for name in MODELS:
    raw = pd.read_csv(os.path.join(DATA, MODELS[name]))
    df, K, X = load_model(name)
    rep[name] = dict(shape=list(raw.shape), columns=list(raw.columns), K=K,
                     n_unique=int(raw.kmer.nunique()), complete=bool(raw.kmer.nunique() == 4 ** K),
                     nan=int(raw.isna().sum().sum()),
                     has_U=bool(raw.kmer.str.contains("U").any()),
                     file_sorted_by_code=bool(np.all(np.diff(pd.Series(raw.kmer).map(lambda s: 0).values) >= 0)),
                     **{f"{c}_{s}": float(getattr(df[c], s)()) for c in ["current_mean", "current_std", "dwell_time"]
                        for s in ["min", "max", "mean", "std"]})
    # check file row order relative to code
    codes_file = df.set_index("kmer").loc[raw.kmer.str.replace("U", "T"), "code"].values
    rep[name]["file_sorted_by_code"] = bool(np.all(np.diff(codes_file) > 0))
    rep[name]["first_rows"] = raw.kmer.head(6).tolist()
    rep[name]["dwell_integer"] = bool(np.allclose(df.dwell_time, np.round(df.dwell_time)))

perf = pd.read_csv(os.path.join(DATA, "performance_summary.csv"))
rep["performance"] = dict(shape=list(perf.shape), columns=list(perf.columns),
                          nan_rows=perf[perf.isna().any(axis=1)][["Chemistry", "Tool"]].values.tolist())
lab = pd.read_csv(os.path.join(DATA, "m6a_labels.csv"))
pu = pd.read_csv(os.path.join(DATA, "m6a_predictions_uncalled4.csv"))
pn = pd.read_csv(os.path.join(DATA, "m6a_predictions_nanopolish.csv"))
rep["m6a"] = dict(n=len(lab), n_pos=int(lab.label.sum()), columns=[list(lab.columns), list(pu.columns), list(pn.columns)],
                  ids_aligned=bool((lab.site_id.values == pu.site_id.values).all() and (lab.site_id.values == pn.site_id.values).all()),
                  ids_unique=bool(lab.site_id.is_unique), id_range=[int(lab.site_id.min()), int(lab.site_id.max())],
                  nan=[int(pu.probability.isna().sum()), int(pn.probability.isna().sum())],
                  u4_exact0=int((pu.probability == 0).sum()), u4_exact1=int((pu.probability == 1).sum()),
                  np_exact0=int((pn.probability == 0).sum()), np_exact1=int((pn.probability == 1).sum()),
                  u4_range=[float(pu.probability.min()), float(pu.probability.max())],
                  np_range=[float(pn.probability.min()), float(pn.probability.max())])
rep["absent_metadata"] = ["read coverage per site", "gene / transcript ID", "cell line", "motif (DRACH) context",
                          "per-read predictions", "alignment-quality (model MAD) values", "compressed / gzip file sizes",
                          "replicate timings / CPU-thread counts", "raw signal (FAST5/POD5/BLOW5)", "BAM alignments"]
print(json.dumps(rep, indent=1))
json.dump(rep, open(os.path.join(OUT, "audit.json"), "w"), indent=1)
