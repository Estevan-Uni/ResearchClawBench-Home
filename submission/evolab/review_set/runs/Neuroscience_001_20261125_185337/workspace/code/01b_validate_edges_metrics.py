"""Validate (i) torch sparse simulator + trimmed moving-edge pipeline against native flyvis
moving_edge_responses and (ii) DSI/PD/FRI computed by native flyvis functions on native
vs fast responses, for models 000-002 (speed 19 deg/s, 12 directions x ON/OFF)."""
import sys, json, time
sys.path.insert(0, "/workspace/code")
import numpy as np
import fastsim as fs, stimuli as S
from flyvis import NetworkView
from flyvis.analysis.stimulus_responses import moving_edge_responses, flash_responses
from flyvis.analysis import moving_bar_responses as MB, flash_responses as FR

types = fs.central_type_order()
out = {}
for i in [0, 1, 2]:
    nv = NetworkView(fs.MODEL_NAMES[i])
    t0 = time.time()
    me = moving_edge_responses(nv, speeds=(19,), dt=1 / 200)
    tn = time.time() - t0
    P = fs.model_params(i); sim = fs.FastSim(P)
    t0 = time.time()
    er, edf, ecfg = S.simulate_edges(sim, (19,), 1 / 200)
    tf = time.time() - t0
    nat = me["responses"].values[0]
    ok = np.isfinite(er)
    err = np.abs(er[ok] - nat[ok]).max()
    dsi_n = MB.direction_selectivity_index(me).squeeze()
    pd_n = MB.preferred_direction(me).squeeze()
    m = S.edge_metrics(er, edf, ecfg)
    dn = np.asarray(dsi_n.transpose('intensity', 'neuron').values)
    pn = np.asarray(pd_n.transpose('intensity', 'neuron').values)
    t45 = [types.index(t) for t in ["T4a", "T4b", "T4c", "T4d", "T5a", "T5b", "T5c", "T5d"]]
    fr = flash_responses(nv, dt=1 / 200)
    fri_n = np.asarray(FR.flash_response_index(fr, radius=6).values).squeeze()
    frf, fdf, fcfg = S.simulate_flashes(sim)
    fri_f = S.fri(frf, fdf, fcfg)
    out[f"{i:03d}"] = dict(native_time_s=tn, fast_time_s=tf, frac_simulated=float(ok.mean()),
                           edge_max_abs_err=float(err), dsi_max_abs_diff=float(np.abs(dn - m["dsi"]).max()),
                           pd_max_abs_diff=float(np.abs(np.angle(np.exp(1j * (pn - m["pd"])))).max()),
                           fri_max_abs_diff=float(np.abs(fri_n - fri_f).max()),
                           native_dsi_T4_on=dn[1, t45[:4]].round(3).tolist(), native_dsi_T5_off=dn[0, t45[4:]].round(3).tolist(),
                           fast_dsi_T4_on=m["dsi"][1, t45[:4]].round(3).tolist(), fast_dsi_T5_off=m["dsi"][0, t45[4:]].round(3).tolist())
    print(i, out[f"{i:03d}"], flush=True)
    json.dump(out, open("/workspace/outputs/01b_validation_edges.json", "w"), indent=1)
