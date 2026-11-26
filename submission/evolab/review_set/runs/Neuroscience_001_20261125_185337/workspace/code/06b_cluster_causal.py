"""Causal follow-up of the cluster<->mechanism coupling: silence Mi9 and/or Mi4 inputs
onto T4 in models with inverted vs correct T4c tuning; measure T4c DSI and PD."""
import sys, os
sys.path.insert(0, "/workspace/code")
import numpy as np, pandas as pd
import fastsim as fs, stimuli as S, metrics as M

OUT = "/workspace/outputs/clusters"
T = fs.topo(); types = fs.central_type_order(); ti = {t: i for i, t in enumerate(types)}
pm = pd.read_csv(f"{OUT}/per_model_T4c.csv")
inv = pm[pm.T4c_class == "inverted"].model.tolist()
cor = pm[pm.T4c_class == "correct"].model.tolist()[:6]
rows = []
for cls, models in [("inverted", inv), ("correct", cor)]:
    for i in models:
        P = fs.model_params(i)
        for cond, src in [("intact", []), ("-Mi9", ["Mi9"]), ("-Mi4", ["Mi4"]), ("-Mi4-Mi9", ["Mi4", "Mi9"])]:
            m = np.ones(T.n_edges)
            m[np.isin(T.edge_src_type, src) & np.isin(T.edge_tgt_type, M.T4)] = 0
            sim = fs.FastSim(P, edge_mask=m)
            er, edf, ecfg = S.simulate_edges(sim, (4.8, 13, 25), 1 / 200)
            mm = S.edge_metrics(er, edf, ecfg)
            on = list(mm["intensity"]).index(1)
            rows.append(dict(cls=cls, model=i, condition=cond, T4c_dsi=mm["dsi"][on, ti["T4c"]],
                             T4c_pd_deg=np.rad2deg(mm["pd"][on, ti["T4c"]]) % 360,
                             T4c_peak=float(np.nanmax(mm["peak"][:, ti["T4c"]])),
                             Mi9_fri=pm.set_index("model").Mi9_fri[i], Mi4_fri=pm.set_index("model").Mi4_fri[i]))
            print(rows[-1], flush=True)
            pd.DataFrame(rows).to_csv(f"{OUT}/causal_Mi4Mi9_T4c.csv", index=False)
