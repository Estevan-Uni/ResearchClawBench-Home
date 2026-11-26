"""Mechanism analyses on trained models (inference-time perturbations):
 (a) targeted silencing of each presynaptic input type onto T4 (or T5) subtypes,
 (b) input-current decomposition onto the central T4c / T5c cell (PD vs ND edges),
 (c) TmY parallel pathway: connectome inputs + silencing of all T4/T5 outputs.
Outputs in outputs/mechanism/."""
import sys, os, time, json
sys.path.insert(0, "/workspace/code")
import numpy as np, pandas as pd
import fastsim as fs, stimuli as S, metrics as M, ablations as A

OUT = "/workspace/outputs/mechanism"; os.makedirs(OUT, exist_ok=True)
DT = 1 / 200
SPEEDS = (4.8, 13, 25)
N_MODELS = int(os.environ.get("N_MECH_MODELS", 15))
T = fs.topo()
types = fs.central_type_order(); ti = {t: i for i, t in enumerate(types)}
tp = pd.read_csv("/workspace/outputs/connectome_type_pairs.csv")

# --------------------------------------------------------------- (c) inputs ---
TMY = ["TmY3", "TmY4", "TmY5a", "TmY13", "TmY18", "TmY14", "TmY15"]
rows = []
for t in TMY + ["T4c", "T5c"]:
    sub = tp[tp.tgt == t].sort_values("n_syn", ascending=False)
    for _, r in sub.iterrows():
        rows.append(dict(target=t, source=r.src, n_syn=r.n_syn, sign=r.sign,
                         source_is_T4T5=r.src in M.T4 + M.T5))
inp = pd.DataFrame(rows); inp.to_csv(f"{OUT}/tmy_t4t5_inputs.csv", index=False)
summ = inp.groupby("target").apply(lambda d: pd.Series(dict(
    n_inputs=len(d), total_syn=d.n_syn.sum(), syn_from_T4T5=d[d.source_is_T4T5].n_syn.sum(),
    T4T5_sources=",".join(d[d.source_is_T4T5].source)))).reset_index()
summ["frac_from_T4T5"] = summ.syn_from_T4T5 / summ.total_syn
summ.to_csv(f"{OUT}/tmy_input_summary.csv", index=False)
print(summ, flush=True)

# --------------------------------------------------------------- helpers -------
def mask_for(src_types, tgt_types=None):
    m = np.ones(T.n_edges)
    sel = np.isin(T.edge_src_type, src_types)
    if tgt_types is not None:
        sel &= np.isin(T.edge_tgt_type, tgt_types)
    m[sel] = 0
    return m


def run_edges(sim, Vg=None):
    Vg = sim.steady_state(DT, 1.0) if Vg is None else Vg
    er, edf, ecfg = S.simulate_edges(sim, SPEEDS, DT, V_grey=Vg)
    m = S.edge_metrics(er, edf, ecfg)
    return m


T4_inputs = tp[tp.tgt == "T4c"].sort_values("n_syn", ascending=False)
T4_inputs = [s for s in T4_inputs[T4_inputs.n_syn >= 1].src if not s.startswith("T4")]
T5_inputs = tp[tp.tgt == "T5c"].sort_values("n_syn", ascending=False)
T5_inputs = [s for s in T5_inputs[T5_inputs.n_syn >= 1].src if not s.startswith("T5")]
json.dump(dict(T4_inputs=T4_inputs, T5_inputs=T5_inputs), open(f"{OUT}/silenced_inputs.json", "w"))
print("T4 inputs", T4_inputs, "\nT5 inputs", T5_inputs, flush=True)

conds = [("intact", None, None)]
conds += [(f"T4-{s}", [s], M.T4) for s in T4_inputs]
conds += [(f"T5-{s}", [s], M.T5) for s in T5_inputs]
conds += [("noT4T5out", M.T4 + M.T5, None), ("noT4out", M.T4, None), ("noT5out", M.T5, None)]

csv = f"{OUT}/silencing.csv"
rows = pd.read_csv(csv).to_dict("records") if os.path.exists(csv) else []
done = {(r["condition"], r["model"]) for r in rows}
for i in range(N_MODELS):
    P = fs.model_params(i)
    for name, src, tgt in conds:
        if (name, i) in done:
            continue
        t0 = time.time()
        sim = fs.FastSim(P, edge_mask=None if src is None else mask_for(src, tgt))
        m = run_edges(sim)
        on, off = list(m["intensity"]).index(1), list(m["intensity"]).index(0)
        pk = m["peak"]  # (samples, neuron)
        row = dict(condition=name, model=i)
        for t in types:
            row[f"dsi_on_{t}"] = m["dsi"][on, ti[t]]; row[f"dsi_off_{t}"] = m["dsi"][off, ti[t]]
            row[f"pd_on_{t}"] = m["pd"][on, ti[t]]; row[f"pd_off_{t}"] = m["pd"][off, ti[t]]
            row[f"peakmax_{t}"] = float(np.nanmax(pk[:, ti[t]]))
        rows.append(row)
        pd.DataFrame(rows).to_csv(csv, index=False)
        print(i, name, f"{time.time()-t0:.0f}s T4c {row['dsi_on_T4c']:.2f} T5c {row['dsi_off_T5c']:.2f}", flush=True)

# --------------------------------------------------------------- (b) currents --
stim, edf, ecfg = S.edge_stimuli((19,), DT)
res = {}
for tgt_type, inten in [("T4c", 1), ("T5c", 0)]:
    node = int(T.central_cells_index[ti[tgt_type]])
    sel = np.where((edf.intensity.values == inten) & np.isin(edf.angle.values, [90, 270]))[0]
    allc = []
    for i in range(10):
        P = fs.model_params(i)
        sim = fs.FastSim(P)
        Vg = sim.steady_state(DT, 1.0)
        v, cur, in_edges = sim.simulate(stim[sel], dt=DT, V0=Vg, record=np.array([node]), currents_onto=node)
        src_types = T.edge_src_type[in_edges]
        ut = list(dict.fromkeys(src_types))
        c_type = np.stack([np.nansum(cur[:, :, src_types == s], -1) for s in ut], -1)  # (2,T,n_src)
        allc.append(dict(types=ut, cur=c_type, v=v[:, :, 0]))
    ut_all = sorted(set(sum([a["types"] for a in allc], [])))
    C = np.zeros((10, len(sel), stim.shape[1], len(ut_all)))
    for k, a in enumerate(allc):
        for j, s in enumerate(a["types"]):
            C[k, :, :, ut_all.index(s)] = a["cur"][:, :, j]
    Vv = np.stack([a["v"] for a in allc])
    np.savez_compressed(f"{OUT}/currents_{tgt_type}.npz", C=C, V=Vv, src_types=np.array(ut_all),
                        angles=edf.angle.values[sel], time=np.arange(stim.shape[1]) * DT - 1.0)
    print("currents", tgt_type, "done", flush=True)
