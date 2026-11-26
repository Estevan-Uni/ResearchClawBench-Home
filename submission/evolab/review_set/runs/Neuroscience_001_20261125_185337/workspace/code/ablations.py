"""Inference-time perturbations of trained DMNs (no retraining).

Every function returns either a modified semantic parameter dict P (same layout as
fastsim.semantic_params) or an explicit edge list (tgt, src, w) that FastSim accepts
via `custom_edges`. Offset convention (verified): du = u_target - u_source."""
import sys
sys.path.insert(0, "/workspace/code")
import numpy as np
import pandas as pd
import fastsim as fs

T = fs.topo()
TYPES = T.cell_types
K_NODE = fs.topo().param_keys["bias"]  # per-type order of node params
K_EDGE = fs.topo().param_keys["syn_strength"]  # (src,tgt) order of edge-type params
K_SC = fs.topo().param_keys["syn_count"]  # (src,tgt,du,dv)
EDGE_DF = pd.DataFrame(K_EDGE, columns=["src", "tgt"])
SC_DF = pd.DataFrame(K_SC, columns=["src", "tgt", "du", "dv"])
# map filter-offset -> edge type index
_et_index = {k: i for i, k in enumerate(K_EDGE)}
SC_DF["et"] = [_et_index[(a, b)] for a, b in zip(SC_DF.src, SC_DF.tgt)]

# coordinate lookup per type: (u,v) -> node index
_coord = {}
for ct in TYPES:
    idx = T.layer_index[ct]
    _coord[ct] = {(int(T.u[i]), int(T.v[i])): int(i) for i in idx}


def output_sign_class():
    P = fs.model_params(0)
    s = pd.Series(P["sign"], index=pd.MultiIndex.from_tuples(K_EDGE))
    cls = {}
    for ct in TYPES:
        if ct in s.index.get_level_values(0):
            v = s.loc[ct].values
            cls[ct] = "E" if (v > 0).all() else ("I" if (v < 0).all() else "mixed")
        else:
            cls[ct] = "none"
    return cls


SIGN_CLASS = None


def sign_class():
    global SIGN_CLASS
    if SIGN_CLASS is None:
        SIGN_CLASS = output_sign_class()
    return SIGN_CLASS


# ------------------------------------------------------------------ params ----
def init_params(P_ref, seed):
    """Untrained DMN: initial distributions of the training config
    (bias ~ N(0.5, 0.05) per type, tau = 50 ms, alpha = 0.01 / <N>_(ti,tj))."""
    rng = np.random.default_rng(seed)
    P = {k: v.copy() for k, v in P_ref.items()}
    P["bias"] = rng.normal(0.5, 0.05, size=len(P["bias"]))
    P["time_const"] = np.full(len(P["time_const"]), 0.05)
    mean_n = SC_DF.assign(n=P_ref["syn_count"]).groupby("et").n.mean().reindex(range(len(K_EDGE))).values
    P["syn_strength"] = 0.01 / mean_n
    return P


def permuted_params(P_ref, seed):
    """Trained parameter values randomly re-assigned across cell types / edge types
    (keeps marginal distributions, destroys type identity)."""
    rng = np.random.default_rng(seed)
    P = {k: v.copy() for k, v in P_ref.items()}
    perm = rng.permutation(len(P["bias"]))
    P["bias"] = P_ref["bias"][perm]
    P["time_const"] = P_ref["time_const"][rng.permutation(len(P["time_const"]))]
    P["syn_strength"] = P_ref["syn_strength"][rng.permutation(len(P["syn_strength"]))]
    return P


def polarity_only_counts(P_ref):
    """Keep sign, connectivity support and trained strengths, but replace every filter
    count with a constant per target type (total synapses onto each type preserved)."""
    P = {k: v.copy() for k, v in P_ref.items()}
    df = SC_DF.assign(n=P_ref["syn_count"])
    tot = df.groupby("tgt").n.transform("sum")
    k = df.groupby("tgt").n.transform("size")
    P["syn_count"] = (tot / k).values
    return P


def tie_params(P_ref, groups):
    """Average node params within groups of types and edge strengths across edge types
    whose (group(src), group(tgt)) coincide. groups: dict type -> group label
    (types not in dict are their own group)."""
    P = {k: v.copy() for k, v in P_ref.items()}
    g = lambda t: groups.get(t, t)
    nodes = pd.DataFrame({"t": K_NODE})
    nodes["g"] = nodes.t.map(g)
    for name in ["bias", "time_const"]:
        P[name] = pd.Series(P_ref[name]).groupby(nodes.g.values).transform("mean").values
    e = EDGE_DF.copy()
    e["gsrc"] = e.src.map(g); e["gtgt"] = e.tgt.map(g)
    e["sign"] = P_ref["sign"]
    P["syn_strength"] = pd.Series(P_ref["syn_strength"]).groupby([e.gsrc.values, e.gtgt.values, e.sign.values]).transform("mean").values
    return P


def groups_58():
    gr = {f"T4{x}": "T4" for x in "abcd"}
    gr.update({f"T5{x}": "T5" for x in "abcd"})
    return gr


def groups_EI():
    cls = sign_class()
    return {t: f"class_{c}" for t, c in cls.items()}


# ------------------------------------------------------------------ edges -----
def edges_from_filters(table):
    """table: DataFrame with columns src, tgt, du, dv, w. Returns (tgt_idx, src_idx, w)
    by placing each filter around every target cell (source at (u-du, v-dv))."""
    tg, sr, ww = [], [], []
    for (s, t), sub in table.groupby(["src", "tgt"], sort=False):
        cs, ctgt = _coord[s], _coord[t]
        for j_uv, j in ctgt.items():
            for du, dv, w in zip(sub.du.values, sub.dv.values, sub.w.values):
                i = cs.get((j_uv[0] - du, j_uv[1] - dv))
                if i is not None:
                    tg.append(j); sr.append(i); ww.append(w)
    return np.array(tg), np.array(sr), np.array(ww)


def filter_table(P):
    df = SC_DF.copy()
    df["w"] = P["syn_count"] * P["sign"][df.et.values] * P["syn_strength"][df.et.values]
    return df


def collapse_spatial(P, targets=None):
    """All offsets of a filter summed into the home column (du=dv=0) -- removes spatial
    structure (for all edge types or only those onto `targets`)."""
    df = filter_table(P)
    sel = df.tgt.isin(targets) if targets is not None else np.ones(len(df), bool)
    a = df[sel].groupby(["src", "tgt"], sort=False).w.sum().reset_index()
    a["du"] = 0; a["dv"] = 0
    return pd.concat([df[~sel][["src", "tgt", "du", "dv", "w"]], a[["src", "tgt", "du", "dv", "w"]]])


def mirror_spatial(P, targets):
    """Point-mirror all filters onto `targets` ((du,dv) -> (-du,-dv))."""
    df = filter_table(P)[["src", "tgt", "du", "dv", "w"]].copy()
    sel = df.tgt.isin(targets)
    df.loc[sel, "du"] *= -1; df.loc[sel, "dv"] *= -1
    return df


def shuffle_sources(P, seed):
    """Sign-class preserving shuffle: each edge type (s,t) keeps its filter, count and
    strength but its source type s is replaced by a random type of the same output
    sign class (E or I), distinct for every target."""
    rng = np.random.default_rng(seed)
    cls = sign_class()
    pools = {c: [t for t, cc in cls.items() if cc == c] for c in ["E", "I", "mixed"]}
    df = filter_table(P)[["src", "tgt", "du", "dv", "w"]].copy()
    new_src = {}
    for t, sub in EDGE_DF.groupby("tgt", sort=False):
        used = set()
        for s in sub.src.values:
            c = cls[s]
            cand = [x for x in pools[c] if x not in used and x != t]
            if not cand:
                cand = pools[c]
            ns = cand[rng.integers(len(cand))]
            used.add(ns)
            new_src[(s, t)] = ns
    df["src"] = [new_src[(s, t)] for s, t in zip(df.src, df.tgt)]
    return df
