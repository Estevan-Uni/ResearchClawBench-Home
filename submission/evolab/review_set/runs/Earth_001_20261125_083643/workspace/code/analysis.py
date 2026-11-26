#!/usr/bin/env python
"""Independent reproduction of the U.S. cloud-seeding records descriptor (2000-2025).

Single deterministic script. Reads the public CSV + state GeoJSON, writes
outputs/*.csv, outputs/summary.json and report/images/*.png.
Run:  python code/analysis.py   (from /workspace)
"""
import json, re, difflib, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, ListedColormap
from matplotlib.patches import Polygon as MplPolygon, Patch
from matplotlib.collections import PatchCollection
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.proportion import proportion_confint

warnings.filterwarnings("ignore")
SEED = 20250101
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "dataset1_cloud_seeding_records"
OUT = ROOT / "outputs"; OUT.mkdir(exist_ok=True)
IMG = ROOT / "report" / "images"; IMG.mkdir(parents=True, exist_ok=True)
S = {}  # summary dictionary -> summary.json
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 110, "savefig.dpi": 200, "savefig.bbox": "tight"})


def wilson(k, n):
    if n == 0:
        return (np.nan, np.nan)
    lo, hi = proportion_confint(k, n, alpha=0.05, method="wilson")
    return (float(lo), float(hi))


def r(x, nd=4):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), nd)

# ---------------------------------------------------------------------------
# 1. Load + normalise
# ---------------------------------------------------------------------------
raw = pd.read_csv(DATA / "cloud_seeding_us_2000_2025.csv", dtype=str, keep_default_na=True)
df = raw.copy()
STRCOLS = [c for c in df.columns if c not in ("filename",)]
for c in STRCOLS:
    s = df[c].astype("object")
    mask = s.notna()
    s.loc[mask] = (s[mask].astype(str).str.lower().str.strip()
                   .str.replace(r"\s*,\s*", ", ", regex=True).str.replace(r"\s+", " ", regex=True))
    df[c] = s
df["year"] = df["year"].astype(int)
df["start"] = pd.to_datetime(df["start_date"], format="%m/%d/%y", errors="coerce")
df["end"] = pd.to_datetime(df["end_date"], format="%m/%d/%y", errors="coerce")
# missing flags BEFORE fill
for c in ["apparatus", "target_area", "control_area", "start_date", "end_date"]:
    df[c + "_missing"] = df[c].isna()
# fill AFTER string ops (G1 pitfall)
for c in STRCOLS:
    if c != "year":
        df[c] = df[c].fillna("not reported")

STATE_ABBR = {"california": "CA", "colorado": "CO", "utah": "UT", "texas": "TX", "idaho": "ID",
              "nevada": "NV", "wyoming": "WY", "north dakota": "ND", "kansas": "KS",
              "south dakota": "SD", "oregon": "OR", "montana": "MT", "oklahoma": "OK"}
ABBR_STATE = {v: k for k, v in STATE_ABBR.items()}
df["st"] = df["state"].map(STATE_ABBR)

# Scope checks
scope = {
    "n_records": int(len(df)), "n_columns_raw": int(raw.shape[1]),
    "year_min": int(df.year.min()), "year_max": int(df.year.max()),
    "n_states": int(df.state.nunique()),
    "n_agent_strings": int(raw.agent.nunique()), "n_purpose_strings": int(raw.purpose.nunique()),
    "n_season_strings": int(raw.season.nunique()), "n_operator_strings": int(df.operator_affiliation.nunique()),
    "n_project_names_raw": int(df.project.nunique()),
    "missing_counts": {c: int(raw[c].isna().sum()) for c in raw.columns},
    "apparatus_values": {str(k): int(v) for k, v in raw.apparatus.value_counts(dropna=False).items()},
}
assert scope["n_records"] == 832 and scope["n_states"] == 13
assert scope["year_min"] == 2000 and scope["year_max"] == 2025
S["scope"] = scope
print("scope", scope)

# year semantics
ok = df.end.notna()
S["year_semantics"] = {
    "share_year_eq_end_year": r((df.loc[ok, "year"] == df.loc[ok, "end"].dt.year).mean()),
    "n_with_end": int(ok.sum()),
    "share_year_eq_start_year": r((df.loc[df.start.notna(), "year"] == df.loc[df.start.notna(), "start"].dt.year).mean()),
    "n_with_start": int(df.start.notna().sum()),
}

# project name normalisation
STOP = r"\b(the|program|programme|project|projects|cloud|seeding|weather|modification|operations|operational|initial|report|winter|summer|wmp|inc|of|and|for)\b"

def norm_proj(p):
    p = re.sub(r"[^a-z0-9 ]", " ", p.lower())
    p = re.sub(r"\b(19|20)\d\d\b", " ", p)
    p = re.sub(STOP, " ", p)
    p = re.sub(r"\bdistrict i\b", "district 1", p); p = re.sub(r"\bdistrict ii\b", "district 2", p)
    p = re.sub(r"\btoiyable\b", "toiyabe", p); p = re.sub(r"\bmountain\b", "mountains", p)
    p = re.sub(r"\bprecipitaion\b", "precipitation", p)
    return " ".join(p.split())

df["proj_norm"] = df.project.map(norm_proj)
df["proj_key"] = df.proj_norm + "|" + df.state
S["projects"] = {"n_raw_names": int(df.project.nunique()), "n_normalised_names": int(df.proj_norm.nunique()),
                 "n_project_state_keys": int(df.proj_key.nunique())}
dup = df.groupby(["proj_key", "year"]).size()
S["projects"]["n_project_year_duplicate_groups"] = int((dup > 1).sum())
S["projects"]["n_excess_duplicate_records"] = int((dup[dup > 1] - 1).sum())
S["projects"]["n_project_state_years"] = int(len(dup))

# ---------------------------------------------------------------------------
# 2. Technical validation: filename evidence + rule-based plausibility
# ---------------------------------------------------------------------------
DATE_RE = re.compile(r"(\d{1,2})[.\-](\d{1,2})[.\-](\d{4})\s*-\s*(\d{1,2})[.\-](\d{1,2})[.\-](\d{2,4})")
MOD_RE = re.compile(r"^(\d{4})(CA|CO|UT|TX|ID|NV|WY|ND|KS|SD|OR|MT|OK)")

fv = []
for i, row in df.iterrows():
    fn = row.filename
    rec = {"idx": i, "filename": fn, "convention": "other"}
    m = DATE_RE.search(fn)
    if m and not MOD_RE.match(fn):
        rec["convention"] = "legacy"
        sm_, sd_, sy_, em_, ed_, ey_ = [int(x) for x in m.groups()]
        if ey_ < 100:
            ey_ += 2000
        rec.update(fn_start=(sm_, sd_, sy_), fn_end=(em_, ed_, ey_))
    mm = MOD_RE.match(fn)
    if mm:
        rec["convention"] = "modern"
        rec["fn_year"] = int(mm.group(1)); rec["fn_code"] = mm.group(2).upper()
    fv.append(rec)
fv = pd.DataFrame(fv).set_index("idx")
df["convention"] = fv.convention


def tup(ts):
    return None if pd.isna(ts) else (ts.month, ts.day, ts.year)

checks = []  # one row per (record, test)
for i, row in df.iterrows():
    f = fv.loc[i]
    if f.convention == "legacy":
        for fld, key, col in [("start_date", "fn_start", "start"), ("end_date", "fn_end", "end")]:
            ext = tup(row[col]); ref = f[key]
            if ext is None:
                agree = False; note = "extracted missing"
            else:
                agree = ext == tuple(ref)
                note = ""
                # tolerate impossible calendar dates in filename (e.g. 06.31) -> compare month/year only
                if not agree:
                    try:
                        pd.Timestamp(year=ref[2], month=ref[0], day=ref[1])
                    except Exception:
                        agree = (ext[0], ext[2]) == (ref[0], ref[2]) or (ext[0] == ref[0] % 12 + 1 and ext[2] == ref[2])
                        note = "impossible filename date"
            checks.append(dict(idx=i, field=fld, source="filename", agree=agree, note=note,
                               extracted=str(ext), reference=str(tuple(ref))))
        checks.append(dict(idx=i, field="year", source="filename", agree=row.year == f.fn_end[2],
                           note="", extracted=row.year, reference=f.fn_end[2]))
    elif f.convention == "modern":
        code = f.fn_code
        ref_state = ABBR_STATE.get(code)
        ambiguous = code == "WY" and row.state != "wyoming" and f.fn_year >= 2023
        xborder = (row.st != code) and (row.state in str(row.target_area))
        checks.append(dict(idx=i, field="state", source="filename", agree=(row.st == code),
                           note="WY token likely = water year" if ambiguous else ("cross-border: target area lies in extracted state" if xborder else ""),
                           extracted=row.st, reference=code))
        checks.append(dict(idx=i, field="year", source="filename", agree=row.year in (f.fn_year, f.fn_year + 1),
                           note="", extracted=row.year, reference=f"{f.fn_year} or {f.fn_year+1}"))
chk = pd.DataFrame(checks)
chk.to_csv(OUT / "T_validation_filename_checks_long.csv", index=False)

fn_tab = []
for (fld, conv), g in chk.assign(conv=chk.idx.map(df.convention)).groupby(["field", "conv"]):
    n = len(g); k = int(g.agree.sum()); lo, hi = wilson(k, n)
    fn_tab.append(dict(field=fld, convention=conv, n_parsable=n, n_agree=k, pct_agree=100 * k / n,
                       ci_lo=100 * lo, ci_hi=100 * hi))
# state excluding the ambiguous "WY = water year" filenames
g = chk[(chk.field == "state") & (chk.note == "")]
k = int(g.agree.sum()); n = len(g); lo, hi = wilson(k, n)
fn_tab.append(dict(field="state (excl. WY water-year / cross-border codes)", convention="modern", n_parsable=n,
                   n_agree=k, pct_agree=100 * k / n, ci_lo=100 * lo, ci_hi=100 * hi))
fn_tab = pd.DataFrame(fn_tab)
fn_tab.to_csv(OUT / "T1_validation_filename_agreement.csv", index=False)
print(fn_tab)
S["validation_filename"] = fn_tab.round(2).to_dict(orient="records")
disagree = chk[~chk.agree].merge(df[["filename", "project", "state", "year", "start_date", "end_date"]],
                                 left_on="idx", right_index=True, suffixes=("", "_rec"))
def classify_fn(rw):
    if rw.field == "state" and rw.note: return "filename convention ambiguity (extracted value supported by record)"
    if rw.field in ("start_date", "end_date", "year"):
        f = fv.loc[rw.idx]
        fs, fe = f.fn_start, f.fn_end
        if rw.extracted in ("None", "nan"): return "impossible filename date -> extracted missing"
        try:
            inverted = (fe[2], fe[0], fe[1]) < (fs[2], fs[0], fs[1]) or fe[2] < 1000
        except Exception:
            inverted = False
        if inverted: return "filename typo (inverted/truncated range); extracted value internally consistent"
        ext = eval(rw.extracted) if rw.field != "year" else None
        ref = eval(rw.reference) if rw.field != "year" else None
        if ext and ref and ext[1] == 10 and ref[1] == 15 and ext[0] == ref[0]:
            return "probable extraction error (day '15' read as '10')"
        if rw.field == "start_date" and ext and ref and ext[2] != ref[2]:
            return "filename typo (inverted/truncated range); extracted value internally consistent"
        return "unresolved discrepancy (manual check needed)"
    return "probable extraction error"
disagree["classification"] = disagree.apply(classify_fn, axis=1) if len(disagree) else []
S["validation_filename_disagreement_classes"] = disagree.groupby(["field", "classification"]).size().astype(int).to_dict()
S["validation_filename_disagreement_classes"] = {f"{a} | {b}": v for (a, b), v in S["validation_filename_disagreement_classes"].items()}
disagree.to_csv(OUT / "T1b_validation_filename_disagreements.csv", index=False)

# --- rule-based plausibility checks (all records + seeded sample of 200) ---
MSEASON = {12: "winter", 1: "winter", 2: "winter", 3: "spring", 4: "spring", 5: "spring",
           6: "summer", 7: "summer", 8: "summer", 9: "fall", 10: "fall", 11: "fall"}
SEASON_VOCAB = {"winter", "spring", "summer", "fall"}
PURPOSE_VOCAB = {"augment snowpack", "increase precipitation", "increase runoff", "suppress hail",
                 "suppress fog", "research"}
APP_VOCAB = {"ground", "airborne", "ground, airborne"}
AGENT_VOCAB = {"silver iodide", "sodium iodide", "ammonium iodide", "calcium chloride", "ionized air", "acetone",
               "hygroscopic aerosols", "hygroscopic", "hygroscopic agents", "hygroscopic materials", "dry ice",
               "sodium chloride", "carbon dioxide", "propane", "liquid propane", "water", "potassium chloride",
               "cesium iodide", "ammonium perchlorate", "sodium perchlorate", "sulfur dioxide", "calcium carbonate",
               "silver chloride", "potassium iodide"}  # vocabulary of chemically valid seeding materials


def midpoint_season(s, e):
    if pd.isna(s) or pd.isna(e) or e < s:
        return None
    return MSEASON[(s + (e - s) / 2).month]


def name_sim(row):
    if row.convention != "legacy":
        return np.nan
    stem = re.split(r"_|previou?s", row.filename, flags=re.I)[0]
    a = norm_proj(stem); b = row.proj_norm
    return difflib.SequenceMatcher(None, a, b).ratio()


df["name_sim"] = df.apply(name_sim, axis=1)
agent_tokens = df.agent.str.split(", ")
purpose_tokens = df.purpose.str.split(", ")
season_tokens = df.season.str.split(", ")

rules = pd.DataFrame(index=df.index)
rules["project"] = (df.project != "not reported") & (df.name_sim.isna() | (df.name_sim >= 0.6))
rules["year"] = df.year.between(2000, 2025) & (df.end.isna() | (df.year == df.end.dt.year))
seas_ok = []; seas_mid = []
for i in df.index:
    toks = set(season_tokens[i]); ms = midpoint_season(df.start[i], df.end[i])
    s_, e_ = df.start[i], df.end[i]
    if pd.isna(s_) or pd.isna(e_) or e_ < s_:
        ov = True
    else:
        months = pd.period_range(s_, e_, freq="M").month
        ov = len(toks & {MSEASON[m] for m in months}) > 0
    seas_ok.append(toks <= SEASON_VOCAB and ov)
    seas_mid.append(toks <= SEASON_VOCAB and (ms is None or ms in toks))
rules["season"] = seas_ok
S["season_strict_midpoint_rule"] = {"n_pass": int(sum(seas_mid)), "pct": r(100 * np.mean(seas_mid), 2),
    "note": "strict rule: season of operating-period midpoint must be listed; fails for year-round/water-year permits"}
rules["state"] = df.state.isin(STATE_ABBR) & ~df.index.isin(
    chk[(chk.field == "state") & (~chk.agree) & (chk.note == "")].idx)
rules["operator_affiliation"] = df.operator_affiliation != "not reported"
ion_air_aircraft = (df.agent == "ionized air") & (df.apparatus != "ground")
rules["agent"] = agent_tokens.map(lambda t: set(t) <= AGENT_VOCAB) & ~ion_air_aircraft
rules["apparatus"] = df.apparatus.isin(APP_VOCAB)
rules["purpose"] = purpose_tokens.map(lambda t: set(t) <= PURPOSE_VOCAB and len(t) == len(set(t)))
rules["target_area"] = df.target_area != "not reported"
dur = (df.end - df.start).dt.days
rules["start_date"] = df.start.notna() & (dur.isna() | ((dur >= 0) & (dur <= 366)))
rules["end_date"] = df.end.notna() & (dur.isna() | ((dur >= 0) & (dur <= 366)))
# control_area is legitimately optional (only some programmes run target/control evaluations): excluded
rules.to_csv(OUT / "T_validation_rules_all_records.csv")

rng = np.random.default_rng(SEED)
sample_idx = np.sort(rng.choice(df.index.values, size=200, replace=False))
pd.Series(sample_idx, name="idx").to_csv(OUT / "validation_sample_idx.csv", index=False)


def cause(field, i):
    """Classify a failure as source error vs probable extraction error vs legitimately absent."""
    f = fv.loc[i]; row = df.loc[i]
    if field in ("start_date", "end_date"):
        if pd.isna(row.start) or pd.isna(row.end):
            return "missing in record (source form incomplete or extraction miss)"
        if f.convention == "legacy" and tup(row.start) == tuple(f.fn_start) and tup(row.end) == tuple(f.fn_end):
            return "source error (filename carries same dates)"
        return "probable extraction or source error (no filename evidence)"
    if field == "agent":
        return "probable extraction error (ionized air on aircraft programme)" if ion_air_aircraft[i] else "garbled/non-chemical token in formulation"
    if field == "apparatus":
        return "probable extraction error (apparatus missing; ionized-air record)" if row.agent == "ionized air" else "missing"
    if field == "purpose":
        return "extraction artefact (duplicated token)"
    if field == "year":
        if f.convention == "legacy" and row.year == f.fn_end[2]:
            return "source error (filename carries same dates)"
        return "probable extraction error (year vs end date)"
    if field == "season":
        return "season label vs operating-period midpoint mismatch"
    if field == "state":
        return "probable extraction error (state vs filename code)"
    if field == "target_area":
        return "missing target area"
    if field == "project":
        stem = re.split(r"_|previou?s", row.filename, flags=re.I)[0]
        if len(norm_proj(stem).split()) <= 1 or re.search(r"noaa|n\.?-?a", stem.lower()):
            return "uninformative filename stem (name not checkable)"
        return "name differs from filename stem (manual check needed)"
    return "other"

val_rows = []; fail_rows = []
for label, idx in [("all_records", df.index.values), ("random_sample_200", sample_idx)]:
    for fld in rules.columns:
        v = rules.loc[idx, fld]; n = len(v); k = int(v.sum()); lo, hi = wilson(k, n)
        val_rows.append(dict(set=label, field=fld, n=n, n_pass=k, pct_pass=100 * k / n, ci_lo=100 * lo, ci_hi=100 * hi))
        if label == "all_records":
            for i in v[~v].index:
                fail_rows.append(dict(idx=i, field=fld, filename=df.filename[i], cause=cause(fld, i),
                                      in_sample=i in set(sample_idx)))
val = pd.DataFrame(val_rows); val.to_csv(OUT / "T2_validation_rule_pass_rates.csv", index=False)
fails = pd.DataFrame(fail_rows); fails.to_csv(OUT / "T2b_validation_failures_classified.csv", index=False)
print(val)
S["validation_rules"] = val.round(2).to_dict(orient="records")
S["validation_failure_causes"] = {f"{a} | {b}": int(c) for (a, b), c in fails.groupby(["field", "cause"]).size().items()}
rec_all = rules.all(axis=1)
S["validation_record_level"] = {
    "all_records_all_fields_pass": int(rec_all.sum()), "all_records_pct": r(100 * rec_all.mean(), 2),
    "sample_all_fields_pass": int(rec_all[sample_idx].sum()), "sample_pct": r(100 * rec_all[sample_idx].mean(), 2),
    "sample_ci": [r(100 * x, 2) for x in wilson(int(rec_all[sample_idx].sum()), 200)],
    "field_level_cells_pass_sample": int(rules.loc[sample_idx].values.sum()),
    "field_level_cells_sample": int(rules.loc[sample_idx].size),
}
S["validation_record_level"]["field_level_pct_sample"] = r(
    100 * S["validation_record_level"]["field_level_cells_pass_sample"] / S["validation_record_level"]["field_level_cells_sample"], 2)

# known artefacts
S["artefacts"] = {
    "ionized_air_records": int((df.agent == "ionized air").sum()),
    "ionized_air_non_ground": int(ion_air_aircraft.sum()),
    "ionized_air_non_ground_states": df.loc[ion_air_aircraft, "st"].value_counts().to_dict(),
    "missing_apparatus_all_ionized_air": bool((df.loc[df.apparatus_missing, "agent"] == "ionized air").all()),
    "n_missing_apparatus": int(df.apparatus_missing.sum()),
    "n_end_before_start": int((dur < 0).sum()), "n_duration_gt_366": int((dur > 366).sum()),
    "n_duplicate_purpose_tokens": int(purpose_tokens.map(lambda t: len(t) != len(set(t))).sum()),
    "n_wy_water_year_tokens": int((chk.note == "WY token likely = water year").sum()),
    "n_cross_border_state_codes": int(chk.note.str.startswith("cross-border").sum()),
}
inv = df[dur < 0]
S["artefacts"]["inverted_dates_match_filename"] = int(sum(
    fv.loc[i].convention == "legacy" and tup(df.start[i]) == tuple(fv.loc[i].fn_start) for i in inv.index))

# Validation figure
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1.25, 1]})
ax = axes[0]
v2 = val.pivot(index="field", columns="set", values="pct_pass")
order = v2["all_records"].sort_values().index
y = np.arange(len(order))
for j, (st, col, off) in enumerate([("all_records", "#4C72B0", -0.18), ("random_sample_200", "#DD8452", 0.18)]):
    sub = val[val.set == st].set_index("field").loc[order]
    ax.barh(y + off, sub.pct_pass, height=0.34, color=col, label=st.replace("_", " "))
    ax.errorbar(sub.pct_pass, y + off, xerr=[sub.pct_pass - sub.ci_lo, sub.ci_hi - sub.pct_pass],
                fmt="none", ecolor="k", lw=0.8, capsize=2)
ax.set_yticks(y); ax.set_yticklabels(order); ax.set_xlim(80, 100.5)
ax.set_xlabel("Proxy consistency rate (%) with Wilson 95% CI")
ax.set_title("(a) Rule-based plausibility per field", loc="left")
ax.legend(loc="upper left", bbox_to_anchor=(0.0, -0.13), ncol=2, frameon=False)
ax = axes[1]
ft = fn_tab.copy(); ft["lab"] = ft.field + " [" + ft.convention + ", n=" + ft.n_parsable.astype(str) + "]"
ft = ft.sort_values("pct_agree")
yy = np.arange(len(ft))
ax.barh(yy, ft.pct_agree, color="#55A868", height=0.6)
ax.errorbar(ft.pct_agree, yy, xerr=[ft.pct_agree - ft.ci_lo, ft.ci_hi - ft.pct_agree], fmt="none", ecolor="k", lw=0.8, capsize=2)
for yi, p in zip(yy, ft.pct_agree):
    ax.text(80.4, yi, f"{p:.1f}%", va="center", ha="left", color="white", fontsize=8, fontweight="bold")
ax.set_yticks(yy); ax.set_yticklabels(ft.lab, fontsize=8); ax.set_xlim(80, 100.5)
ax.set_xlabel("Agreement with filename-encoded value (%)")
ax.set_title("(b) Field vs. filename evidence", loc="left")
plt.tight_layout(); plt.savefig(IMG / "fig1_validation_proxy_accuracy.png"); plt.close()

# ---------------------------------------------------------------------------
# 3. Spatial concentration
# ---------------------------------------------------------------------------
gj = json.load(open(DATA / "us_states.geojson"))
polys = {}
for feat in gj["features"]:
    nm = feat["properties"]["name"].lower()
    if nm in ("alaska", "hawaii", "puerto rico"):
        continue
    g = feat["geometry"]
    rings = [g["coordinates"][0]] if g["type"] == "Polygon" else [p[0] for p in g["coordinates"]]
    polys[nm] = [np.asarray(rg) for rg in rings]


def ring_area_centroid(rg):
    x, y = rg[:, 0], rg[:, 1]
    a = x[:-1] * y[1:] - x[1:] * y[:-1]
    A = a.sum() / 2
    cx = ((x[:-1] + x[1:]) * a).sum() / (6 * A); cy = ((y[:-1] + y[1:]) * a).sum() / (6 * A)
    return abs(A), cx, cy

cent = {}
for nm, rings in polys.items():
    best = max(rings, key=lambda rg: ring_area_centroid(rg)[0])
    _, cx, cy = ring_area_centroid(best)
    cent[nm] = (cx, cy, np.ptp(best[:, 0]), np.ptp(best[:, 1]))

st_counts = df.state.value_counts()
proj_counts = df.groupby("state").proj_key.nunique()
# project-cluster bootstrap CI for state shares
projs = df.proj_key.unique()
grp = {p: g.state.values for p, g in df.groupby("proj_key")}
B = 2000; boot = np.zeros((B, len(st_counts)))
rng_b = np.random.default_rng(SEED + 1)
states_order = list(st_counts.index)
for b in range(B):
    pick = rng_b.choice(projs, size=len(projs), replace=True)
    allst = np.concatenate([grp[p] for p in pick])
    vc = pd.Series(allst).value_counts()
    boot[b] = [vc.get(s, 0) / len(allst) for s in states_order]
top3 = ["california", "colorado", "utah"]
top3_boot = boot[:, [states_order.index(s) for s in top3]].sum(1)
st_tab = pd.DataFrame({"state": states_order, "abbr": [STATE_ABBR[s] for s in states_order],
                       "records": st_counts.values, "share_pct": 100 * st_counts.values / len(df),
                       "boot_ci_lo": 100 * np.percentile(boot, 2.5, 0), "boot_ci_hi": 100 * np.percentile(boot, 97.5, 0),
                       "projects": [proj_counts[s] for s in states_order]})
st_tab["cum_share_pct"] = st_tab.share_pct.cumsum()
st_tab.to_csv(OUT / "T3_state_counts.csv", index=False)
assert st_tab.records.sum() == 832 and states_order[:3] == top3
S["spatial"] = {
    "state_records": {STATE_ABBR[s]: int(c) for s, c in st_counts.items()},
    "state_projects": {STATE_ABBR[s]: int(proj_counts[s]) for s in states_order},
    "top3_share_pct": r(st_tab.share_pct[:3].sum(), 2),
    "top3_share_boot_ci": [r(100 * np.percentile(top3_boot, 2.5), 2), r(100 * np.percentile(top3_boot, 97.5), 2)],
    "western_11_share_pct": r(100 * df.state.isin(["california", "colorado", "utah", "idaho", "nevada", "wyoming", "montana", "oregon"]).mean(), 2),
    "plains_share_pct": r(100 * df.state.isin(["texas", "north dakota", "kansas", "south dakota", "oklahoma"]).mean(), 2),
    "hhi_state": r(((st_counts / len(df)) ** 2).sum(), 4),
    "top5_share_pct": r(st_tab.share_pct[:5].sum(), 2),
}

fig, ax = plt.subplots(figsize=(11, 6.4))
cmap = plt.get_cmap("YlGnBu"); norm = Normalize(0, st_counts.max())
for nm, rings in polys.items():
    c = st_counts.get(nm, 0)
    fc = cmap(norm(c)) if c > 0 else "#e6e6e6"
    for rg in rings:
        ax.add_patch(MplPolygon(rg, closed=True, fc=fc, ec="white", lw=0.5))
rng_j = np.random.default_rng(SEED + 2)
LABPOS = {"CA": (-124.2, 33.2), "NV": (-117.0, 41.6), "UT": (-111.6, 37.1), "CO": (-105.5, 37.1), "ID": (-114.6, 46.2), "TX": (-99.0, 27.6), "KS": (-98.5, 39.6), "OK": (-97.5, 36.0), "SD": (-100.3, 43.6), "ND": (-100.3, 48.6)}
app_col = {"ground": "#D55E00", "airborne": "#7B2D8E", "ground, airborne": "#E69F00", "not reported": "#555555"}
for nm in st_counts.index:
    cx, cy, w, h = cent[nm]
    sub = df[df.state == nm]
    sc = min(0.6, 0.12 * min(w, h)) if len(sub) > 1 else 0
    jx = rng_j.normal(0, sc * 1.3, len(sub)); jy = rng_j.normal(0, sc, len(sub))
    ax.scatter(cx + jx, cy + jy, s=6, c=sub.apparatus.map(app_col), alpha=0.75, lw=0, zorder=3)
    lx, ly = LABPOS.get(STATE_ABBR[nm], (cx, cy + max(0.9, 2.2 * sc)))
    ax.text(lx, ly, f"{STATE_ABBR[nm]}: {st_counts[nm]} rec\n{proj_counts[nm]} proj",
            ha="center", va="bottom", fontsize=7, zorder=4,
            bbox=dict(fc="white", ec="none", alpha=0.7, pad=0.6))
ax.set_xlim(-125, -66.5); ax.set_ylim(24, 50); ax.set_aspect(1 / np.cos(np.deg2rad(37)))
ax.axis("off")
sm_ = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
cb = plt.colorbar(sm_, ax=ax, shrink=0.6, pad=0.01); cb.set_label("Records per state (2000–2025)")
ax.legend(handles=[Patch(color=v, label=k) for k, v in app_col.items()] + [Patch(color="#e6e6e6", label="no records")],
          title="Point = one record (jittered)\napparatus", loc="lower left", frameon=False, fontsize=8)
ax.set_title(f"Reported cloud-seeding records by state, 2000–2025 (n={len(df)}); "
             f"CA+CO+UT = {S['spatial']['top3_share_pct']:.1f}%", loc="left")
plt.savefig(IMG / "fig2_spatial_map.png"); plt.close()

# ---------------------------------------------------------------------------
# 4. Annual activity
# ---------------------------------------------------------------------------
years = np.arange(2000, 2026)
ann = df.groupby("year").size().reindex(years, fill_value=0)
ann_st = pd.crosstab(df.year, df.state).reindex(years, fill_value=0)[states_order]
ann_st.to_csv(OUT / "T4_annual_by_state.csv")
ann_proj = df.groupby("year").proj_key.nunique().reindex(years, fill_value=0)


def mann_kendall(x):
    x = np.asarray(x, float); n = len(x)
    s = sum(np.sign(x[j] - x[i]) for i in range(n - 1) for j in range(i + 1, n))
    _, tc = np.unique(x, return_counts=True)
    var = (n * (n - 1) * (2 * n + 5) - sum(t * (t - 1) * (2 * t + 5) for t in tc)) / 18
    z = (s - np.sign(s)) / np.sqrt(var) if s != 0 else 0
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    tau = s / (n * (n - 1) / 2)
    return dict(S=float(s), z=float(z), p=float(p), tau=float(tau))


def pettitt(x):
    x = np.asarray(x, float); n = len(x)
    U = np.array([sum(np.sign(x[i] - x[j]) for i in range(t + 1) for j in range(t + 1, n)) for t in range(n - 1)])
    k = int(np.argmax(np.abs(U))); K = abs(U[k])
    p = min(1.0, 2 * np.exp(-6 * K ** 2 / (n ** 3 + n ** 2)))
    return dict(t_index=k, K=float(K), p=float(p))

trend = {}
for lab, yrs in [("2000_2025", years), ("2000_2024", years[:-1])]:
    x = ann.loc[yrs].values
    mk = mann_kendall(x); sen = stats.theilslopes(x, yrs)
    pt = pettitt(x)
    trend[lab] = dict(mk=mk, sen_slope_per_yr=float(sen[0]), sen_ci=[float(sen[2]), float(sen[3])],
                      pettitt_last_year_before_change=int(yrs[pt["t_index"]]), pettitt_K=pt["K"], pettitt_p=pt["p"],
                      mean_before=float(x[:pt["t_index"] + 1].mean()), mean_after=float(x[pt["t_index"] + 1:].mean()))
peak_year = int(ann.idxmax()); trough_year = int(ann.loc[2005:2024].idxmin())
# piecewise Poisson with knots at peak & trough (exclude 2025)
pdf_ = pd.DataFrame({"y": ann.loc[2000:2024].values, "t": np.arange(2000, 2025)})
pdf_["t0"] = pdf_.t - 2000
pdf_["k1"] = np.clip(pdf_.t - peak_year, 0, None); pdf_["k2"] = np.clip(pdf_.t - trough_year, 0, None)
pw = smf.glm("y ~ t0 + k1 + k2", data=pdf_, family=sm.families.Poisson()).fit()
disp = float(pw.pearson_chi2 / pw.df_resid)
pw_q = smf.glm("y ~ t0 + k1 + k2", data=pdf_, family=sm.families.Poisson()).fit(scale=max(1.0, disp))
b = pw_q.params; cov = pw_q.cov_params()
def seg(names):
    est = sum(b[n] for n in names); se = np.sqrt(sum(cov.loc[a, c] for a in names for c in names))
    return dict(irr=float(np.exp(est)), pct_per_yr=float(100 * (np.exp(est) - 1)),
                ci=[float(100 * (np.exp(est - 1.96 * se) - 1)), float(100 * (np.exp(est + 1.96 * se) - 1))],
                p=float(2 * (1 - stats.norm.cdf(abs(est / se)))))
pw_res = {"knots": [peak_year, trough_year], "dispersion": disp,
          f"seg_2000_{peak_year}": seg(["t0"]), f"seg_{peak_year}_{trough_year}": seg(["t0", "k1"]),
          f"seg_{trough_year}_2024": seg(["t0", "k1", "k2"])}
pdf_["fit"] = pw.predict(pdf_)
rebound = dict(mean_2018_2021=float(ann.loc[2018:2021].mean()), mean_2022_2024=float(ann.loc[2022:2024].mean()))
S["annual"] = {"records_by_year": {int(k): int(v) for k, v in ann.items()},
               "projects_by_year": {int(k): int(v) for k, v in ann_proj.items()},
               "peak_year": peak_year, "peak_records": int(ann.max()), "trough_year": trough_year,
               "trough_records": int(ann.loc[trough_year]), "mean_2000_2009": float(ann.loc[2000:2009].mean()),
               "mean_2010_2019": float(ann.loc[2010:2019].mean()), "mean_2020_2025": float(ann.loc[2020:2025].mean()),
               "trend": trend, "piecewise_poisson": pw_res, "rebound": rebound,
               "states_active_by_year": {int(k): int(v) for k, v in (ann_st > 0).sum(1).items()}}
pd.DataFrame({"records": ann, "projects": ann_proj}).to_csv(OUT / "T4b_annual_totals.csv")

fig, ax = plt.subplots(figsize=(11, 5))
pal = list(plt.get_cmap("tab20").colors)
pal13 = [pal[i] for i in [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 1, 5, 9]]
bottom = np.zeros(len(years))
for s, c in zip(states_order, pal13):
    ax.bar(years, ann_st[s].values, bottom=bottom, color=c, width=0.8, label=f"{STATE_ABBR[s]} ({st_counts[s]})",
           edgecolor="white", lw=0.3)
    bottom += ann_st[s].values
ax.bar([2025], [ann[2025]], width=0.8, fill=False, hatch="///", ec="grey", lw=0)
ax.plot(pdf_.t, pdf_.fit, color="k", lw=1.5, label="Piecewise Poisson fit\n(knots %d, %d)" % (peak_year, trough_year))
for yv, v in ann.items():
    ax.text(yv, v + 0.6, str(v), ha="center", fontsize=7)
ax.annotate("2025 possibly\nincomplete", xy=(2025, ann[2025]), xytext=(2022.6, 41), fontsize=7,
            arrowprops=dict(arrowstyle="->", lw=0.7))
t = trend["2000_2024"]
ax.text(0.99, 0.97, f"Mann–Kendall 2000–2024: τ={t['mk']['tau']:.2f}, p={t['mk']['p']:.3f}\n"
        f"Sen slope = {t['sen_slope_per_yr']:.2f} records/yr\n"
        f"Pettitt change after {t['pettitt_last_year_before_change']} (p={t['pettitt_p']:.3f})",
        transform=ax.transAxes, ha="right", va="top", fontsize=8, bbox=dict(fc="white", ec="0.8"))
ax.set_xticks(years); ax.set_xticklabels(years, rotation=90)
ax.set_ylabel("Number of records (by season-ending year)"); ax.set_xlim(1999.4, 2025.6)
ax.legend(ncol=1, fontsize=7, frameon=False, bbox_to_anchor=(1.0, 1.0), loc="upper left", title="State (total)")
ax.set_title("Annual reported cloud-seeding activity, stacked by state", loc="left")
plt.savefig(IMG / "fig3_annual_stacked_by_state.png"); plt.close()

# ---------------------------------------------------------------------------
# 5. Purpose composition
# ---------------------------------------------------------------------------
CONCEPT = {"augment snowpack": "Augment snowpack", "increase precipitation": "Increase precipitation",
           "increase runoff": "Increase runoff", "suppress hail": "Suppress hail",
           "suppress fog": "Suppress fog (visibility improvement)", "research": "Research"}
raw_tok = [CONCEPT[t] for toks in purpose_tokens for t in toks]
dd_tok = [CONCEPT[t] for toks in purpose_tokens for t in dict.fromkeys(toks)]
pr = pd.Series(raw_tok).value_counts(); pdd = pd.Series(dd_tok).value_counts()
rec_share = pd.Series({c: purpose_tokens.map(lambda t: k in t).mean() * 100 for k, c in CONCEPT.items()})
pur_tab = pd.DataFrame({"mentions_raw": pr, "share_raw_pct": 100 * pr / pr.sum(),
                        "mentions_dedup": pdd, "share_dedup_pct": 100 * pdd / pdd.sum(),
                        "pct_of_records": rec_share}).sort_values("mentions_raw", ascending=False)
pur_tab.to_csv(OUT / "T5_purpose_mentions.csv")
combo = df.purpose.str.split(", ").map(lambda t: " + ".join(sorted(dict.fromkeys(t)))).value_counts()
combo.to_csv(OUT / "T5b_purpose_combinations.csv")
df["n_purposes"] = purpose_tokens.map(lambda t: len(set(t)))
S["purpose"] = {"total_mentions_raw": int(pr.sum()), "total_mentions_dedup": int(pdd.sum()),
                "mentions_raw": pr.astype(int).to_dict(), "mentions_dedup": pdd.astype(int).to_dict(),
                "share_raw_pct": (100 * pr / pr.sum()).round(2).to_dict(),
                "share_dedup_pct": (100 * pdd / pdd.sum()).round(2).to_dict(),
                "pct_records": rec_share.round(2).to_dict(),
                "snowpack_plus_precip_share_raw_pct": r(100 * (pr["Augment snowpack"] + pr["Increase precipitation"]) / pr.sum(), 2),
                "snowpack_plus_precip_share_dedup_pct": r(100 * (pdd["Augment snowpack"] + pdd["Increase precipitation"]) / pdd.sum(), 2),
                "pct_multi_purpose_records": r(100 * (df.n_purposes > 1).mean(), 2),
                "top_combinations": combo.head(8).astype(int).to_dict()}
assert S["purpose"]["snowpack_plus_precip_share_raw_pct"] > 70
pst = pd.DataFrame({c: [purpose_tokens[df.state == s].map(lambda t: k in t).mean() * 100 for s in states_order]
                    for k, c in CONCEPT.items()}, index=[STATE_ABBR[s] for s in states_order])
pst.to_csv(OUT / "T5c_purpose_by_state_pct_records.csv")

fig, axes = plt.subplots(1, 2, figsize=(12, 4.3), gridspec_kw={"width_ratios": [1.15, 1]})
ax = axes[0]; o = pur_tab.sort_values("mentions_raw")
yy = np.arange(len(o))
ax.barh(yy, o.share_raw_pct, color="#4C72B0", height=0.65)
for yi, (n, p, nd, pd_) in enumerate(zip(o.mentions_raw, o.share_raw_pct, o.mentions_dedup, o.share_dedup_pct)):
    ax.text(p + 0.5, yi, f"{n} ({p:.1f}%)   [dedup: {nd}, {pd_:.1f}%]", va="center", fontsize=8)
ax.set_yticks(yy); ax.set_yticklabels(o.index); ax.set_xlim(0, 75)
ax.set_xlabel(f"Share of purpose mentions (%), raw tokens, N = {pr.sum()} mentions in {len(df)} records")
ax.set_title("(a) Stated purposes, share of mentions (descending)", loc="left")
ax = axes[1]
im = ax.imshow(pst.values, cmap="Blues", aspect="auto", vmin=0, vmax=100)
ax.set_xticks(range(pst.shape[1])); ax.set_xticklabels([c.replace(" (visibility improvement)", "\n(visibility)") for c in pst.columns], rotation=40, ha="right", fontsize=7.5)
ax.set_yticks(range(pst.shape[0])); ax.set_yticklabels([f"{a} (n={st_counts[ABBR_STATE[a]]})" for a in pst.index], fontsize=7.5)
for i_ in range(pst.shape[0]):
    for j_ in range(pst.shape[1]):
        v = pst.values[i_, j_]
        if v > 0:
            ax.text(j_, i_, f"{v:.0f}", ha="center", va="center", fontsize=6.5, color="white" if v > 60 else "black")
plt.colorbar(im, ax=ax, label="% of state's records citing purpose")
ax.set_title("(b) Purpose by state (% of records)", loc="left")
plt.tight_layout(); plt.savefig(IMG / "fig4_purposes.png"); plt.close()

# ---------------------------------------------------------------------------
# 6. Agent x apparatus
# ---------------------------------------------------------------------------
def light_norm_agent(a):
    a = re.sub(r"hygroscopic( aerosols| agents| materials)?", "hygroscopic agents", a)
    a = a.replace("liquid propane", "propane")
    if "perchlorate" in a or a == "silver iodide, ammonium, sodium perchlorate, acetone":
        a = "silver iodide, ammonium iodide, perchlorates, acetone"
    if a == "dry ice":
        a = "dry ice (solid co2)"
    return a

df["agent_ln"] = df.agent.map(light_norm_agent)
APP_LAB = {"ground": "Ground", "airborne": "Airborne", "ground, airborne": "Ground + airborne", "not reported": "Not reported"}
df["app_lab"] = df.apparatus.map(APP_LAB)
ct_full = pd.crosstab(df.agent_ln, df.app_lab)[list(APP_LAB.values())]
ct_full["Total"] = ct_full.sum(1); ct_full = ct_full.sort_values("Total", ascending=False)
ct_full.to_csv(OUT / "T6_agent_by_apparatus_full.csv")
TOPN = 10
top = ct_full.head(TOPN).copy()
rest = ct_full.iloc[TOPN:].sum(); rest.name = f"all other agents ({len(ct_full) - TOPN} formulations)"
top = pd.concat([top, rest.to_frame().T])
top.to_csv(OUT / "T6b_agent_by_apparatus_top10.csv")
col_tot = ct_full[list(APP_LAB.values())].sum()
lead = {c: ct_full[c].idxmax() for c in ["Ground", "Airborne", "Ground + airborne"]}
assert all("silver iodide" in v for v in lead.values())
df["has_agi"] = df.agent.str.contains("silver iodide")


def family(a):
    if a == "silver iodide": return "AgI only"
    if "silver iodide" in a and any(k in a for k in ["sodium iodide", "ammonium iodide", "cesium iodide", "potassium iodide", "silver chloride"]):
        return "AgI + iodide/halide complexing salts"
    if "silver iodide" in a and any(k in a for k in ["hygroscopic", "calcium chloride", "sodium chloride", "potassium chloride"]):
        return "AgI + hygroscopic salts"
    if "silver iodide" in a: return "AgI + glaciogenic/other (dry ice, propane, acetone)"
    if a in ("calcium chloride",): return "Hygroscopic only"
    if a in ("dry ice", "carbon dioxide"): return "CO2 / dry ice only"
    if a == "ionized air": return "Ionized air"
    return "Other non-conventional (water, SO2/CaCO3, NH4I)"

df["agent_family"] = df.agent.map(family)
fam = pd.crosstab(df.agent_family, df.app_lab)
fam = fam[[c for c in APP_LAB.values() if c in fam.columns]]
fam.to_csv(OUT / "T6c_agent_family_by_apparatus.csv")
fam3 = fam[["Ground", "Airborne", "Ground + airborne"]]
fam3 = fam3[fam3.sum(1) > 0]
chi2, pchi, dof, exp = stats.chi2_contingency(fam3)
cramer = np.sqrt(chi2 / (fam3.values.sum() * (min(fam3.shape) - 1)))
# Monte-Carlo p (sparse cells)
rng_c = np.random.default_rng(SEED + 3)
sub = df[df.app_lab.isin(["Ground", "Airborne", "Ground + airborne"])]
obs = chi2; cnt = 0; NP = 2000
for _ in range(NP):
    perm = rng_c.permutation(sub.app_lab.values)
    c2 = stats.chi2_contingency(pd.crosstab(sub.agent_family.values, perm))[0]
    cnt += c2 >= obs
# conditional percentages: airborne use (airborne or both) by family, West only
WEST = ["california", "colorado", "utah", "idaho", "nevada", "wyoming", "montana", "oregon"]
west = df[df.state.isin(WEST) & (df.apparatus != "not reported")].copy()
west["air"] = west.apparatus.str.contains("airborne").astype(int)
cond = west.groupby("agent_family").air.agg(["sum", "count"])
cond["pct_airborne_use"] = 100 * cond["sum"] / cond["count"]
cond.to_csv(OUT / "T6d_west_airborne_use_by_family.csv")
# Project-cluster bootstrap CIs for airborne-use % by family (West). A GEE/logit is
# completely separated here (hygroscopic-salt records are 100% airborne, complexing-salt ~1%).
rng_g = np.random.default_rng(SEED + 4)
wp = {p: g for p, g in west.groupby("proj_key")}; wkeys = list(wp)
fams_w = cond.index.tolist(); bt = {f_: [] for f_ in fams_w}
for _ in range(1000):
    bs = pd.concat([wp[k] for k in rng_g.choice(wkeys, len(wkeys))])
    gg = bs.groupby("agent_family").air.mean()
    for f_ in fams_w:
        if f_ in gg: bt[f_].append(100 * gg[f_])
cond["boot_ci_lo"] = [np.percentile(bt[f_], 2.5) for f_ in fams_w]
cond["boot_ci_hi"] = [np.percentile(bt[f_], 97.5) for f_ in fams_w]
cond["n_projects"] = west.groupby("agent_family").proj_key.nunique()
cond.to_csv(OUT / "T6d_west_airborne_use_by_family.csv")
gee_res = {"note": "GEE completely separated; reporting project-bootstrap CIs instead",
           "table": cond.round(2).reset_index().to_dict(orient="records")}
S["agent_apparatus"] = {
    "apparatus_totals": col_tot.astype(int).to_dict(),
    "pct_ground_any": r(100 * df.apparatus.str.contains("ground").mean(), 2),
    "pct_airborne_any": r(100 * df.apparatus.str.contains("airborne").mean(), 2),
    "pct_both": r(100 * (df.apparatus == "ground, airborne").mean(), 2),
    "n_agent_formulations_raw": int(df.agent.nunique()), "n_agent_formulations_light_norm": int(df.agent_ln.nunique()),
    "top10_agents": top.Total.astype(int).to_dict(),
    "leading_agent_per_apparatus": lead,
    "leading_agent_counts": {c: int(ct_full.loc[lead[c], c]) for c in lead},
    "pct_records_with_agi": r(100 * df.has_agi.mean(), 2), "n_records_with_agi": int(df.has_agi.sum()),
    "pct_agi_only": r(100 * (df.agent == "silver iodide").mean(), 2),
    "agi_share_by_apparatus": {c: r(100 * df.loc[df.app_lab == c, "has_agi"].mean(), 2) for c in APP_LAB.values()},
    "family_chi2": float(chi2), "family_dof": int(dof), "family_p_asymptotic": float(pchi),
    "family_p_permutation": float((cnt + 1) / (NP + 1)), "cramers_v": float(cramer),
    "family_counts": df.agent_family.value_counts().astype(int).to_dict(),
    "west_airborne_use_by_family": cond.pct_airborne_use.round(2).to_dict(),
    "west_gee": gee_res,
}

fig, ax = plt.subplots(figsize=(9.5, 6))
cols = ["Ground", "Airborne", "Ground + airborne", "Not reported"]
M = top[cols].values
im = ax.imshow(M, cmap="viridis", aspect="auto", norm=Normalize(0, M.max()))
for i_ in range(M.shape[0]):
    for j_ in range(M.shape[1]):
        v = int(M[i_, j_])
        ax.text(j_, i_, str(v), ha="center", va="center", fontsize=9, color="white" if v < 0.55 * M.max() else "black")
ax.set_xticks(range(len(cols))); ax.set_xticklabels([f"{c}\n(n={int(col_tot[c])})" for c in cols])
ax.set_yticks(range(len(top))); ax.set_yticklabels([f"{a}  (n={int(n)})" for a, n in zip(top.index, top.Total)])
cb = plt.colorbar(im, ax=ax); cb.set_label("Number of records (linear scale)")
ax.set_title("Seeding agent (as reported, light normalisation) × deployment apparatus", loc="left")
for sp in ax.spines.values(): sp.set_visible(False)
plt.savefig(IMG / "fig5_agent_apparatus_heatmap.png"); plt.close()

# ---------------------------------------------------------------------------
# 7. Robustness across units of analysis
# ---------------------------------------------------------------------------
def headline(d):
    pt = d.purpose.str.split(", ").map(lambda t: list(dict.fromkeys(t)))
    toks = [t for ts in pt for t in ts]
    toks = pd.Series(toks).value_counts()
    return {"top3_state_share": 100 * d.state.isin(top3).mean(),
            "agi_share": 100 * d.agent.str.contains("silver iodide").mean(),
            "ground_only_share": 100 * (d.apparatus == "ground").mean(),
            "any_airborne_share": 100 * d.apparatus.str.contains("airborne").mean(),
            "snow_precip_mention_share": 100 * (toks.get("augment snowpack", 0) + toks.get("increase precipitation", 0)) / toks.sum(),
            "winter_share": 100 * d.season.str.contains("winter").mean(),
            "n": len(d)}

units = {"records": df,
         "project_state_years": df.sort_values("filename").drop_duplicates(["proj_key", "year"]),
         "one_record_per_project": df.sort_values("year").groupby("proj_key").tail(1),
         "excl_nonconventional_agents": df[~df.agent.isin(["ionized air", "water", "sulfur dioxide, calcium carbonate"])],
         "excl_2025": df[df.year < 2025],
         "excl_validation_failures": df[rec_all]}
rob = pd.DataFrame({k: headline(v) for k, v in units.items()}).T
rob.to_csv(OUT / "T7_robustness_units.csv")
dev = (rob.drop(columns="n") - rob.loc["records"].drop("n")).abs()
S["robustness"] = {"table": rob.round(2).to_dict(orient="index"),
                   "max_abs_change_pp_excl_project_unit": r(dev.drop(index="one_record_per_project").values.max(), 2),
                   "max_abs_change_pp_all": r(dev.values.max(), 2),
                   "max_change_by_metric_pp": dev.max().round(2).to_dict()}

# ---------------------------------------------------------------------------
# 8. Extras: operators, persistence, seasonality, duration
# ---------------------------------------------------------------------------
op = df.operator_affiliation.value_counts()
op.to_csv(OUT / "T8_operators.csv")
pers = df.groupby("proj_key").year.agg(["min", "max", "nunique"])
pers["span"] = pers["max"] - pers["min"] + 1
pers.to_csv(OUT / "T8b_project_persistence.csv")
seas = df.season.str.split(", ").explode().value_counts()
startm = df.start.dt.month.value_counts().reindex(range(1, 13), fill_value=0)
durd = dur[(dur >= 0) & (dur <= 366)]
S["extras"] = {"n_operators": int(op.size), "top5_operators": op.head(5).astype(int).to_dict(),
               "top5_operator_share_pct": r(100 * op.head(5).sum() / len(df), 2),
               "projects_active_10plus_years": int((pers["nunique"] >= 10).sum()),
               "median_years_active": float(pers["nunique"].median()), "n_projects": int(len(pers)),
               "season_mentions": seas.astype(int).to_dict(),
               "pct_records_with_winter": r(100 * df.season.str.contains("winter").mean(), 2),
               "start_month_counts": {int(k): int(v) for k, v in startm.items()},
               "median_duration_days": float(durd.median()),
               "duration_iqr": [float(durd.quantile(.25)), float(durd.quantile(.75))],
               "control_area_reported_pct": r(100 * (~df.control_area_missing).mean(), 2)}

fig, axes = plt.subplots(1, 3, figsize=(13, 4))
ax = axes[0]; o = op.head(10)[::-1]
ax.barh(range(len(o)), o.values, color="#4C72B0")
ax.set_yticks(range(len(o))); ax.set_yticklabels(o.index, fontsize=7.5)
for i_, v in enumerate(o.values): ax.text(v + 2, i_, str(v), va="center", fontsize=7)
ax.set_xlabel("Records"); ax.set_title(f"(a) Top-10 operators ({op.size} total)", loc="left")
ax = axes[1]
ax.hist(pers["nunique"], bins=np.arange(0.5, pers["nunique"].max() + 1.5), color="#55A868", ec="white")
ax.set_xlabel("Distinct years with a record"); ax.set_ylabel("Projects (normalised name × state)")
ax.set_title(f"(b) Project persistence (n={len(pers)})", loc="left")
ax = axes[2]
mn = ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"]
ax.bar(range(1, 13), startm.values, color="#C44E52")
ax.set_xticks(range(1, 13)); ax.set_xticklabels(mn); ax.set_ylabel("Records")
ax.set_title("(c) Month of operational start", loc="left")
plt.tight_layout(); plt.savefig(IMG / "fig6_operators_persistence_seasonality.png"); plt.close()

# Robustness figure
fig, ax = plt.subplots(figsize=(10, 4))
mets = ["top3_state_share", "agi_share", "ground_only_share", "any_airborne_share", "snow_precip_mention_share", "winter_share"]
xx = np.arange(len(mets)); wdt = 0.13
for j, (u, c) in enumerate(zip(rob.index, plt.get_cmap("Set2").colors)):
    ax.bar(xx + (j - 2.5) * wdt, rob.loc[u, mets].astype(float), width=wdt, color=c, label=f"{u} (n={int(rob.loc[u,'n'])})")
ax.set_xticks(xx); ax.set_xticklabels([m.replace("_", " ") for m in mets], rotation=15)
ax.set_ylabel("%"); ax.legend(fontsize=7, frameon=False, ncol=2, loc="upper left", bbox_to_anchor=(0, 1.22))
ax.set_ylim(0, 105)
plt.tight_layout(); plt.savefig(IMG / "fig7_robustness_units.png"); plt.close()

# ---------------------------------------------------------------------------
def to_py(o):
    if isinstance(o, dict): return {str(k): to_py(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [to_py(v) for v in o]
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, (np.floating, float)): return None if np.isnan(o) else round(float(o), 4)
    if isinstance(o, np.bool_): return bool(o)
    return o

json.dump(to_py(S), open(OUT / "summary.json", "w"), indent=1)
df.drop(columns=["start", "end"]).to_csv(OUT / "records_normalised.csv", index=False)
print("done")
