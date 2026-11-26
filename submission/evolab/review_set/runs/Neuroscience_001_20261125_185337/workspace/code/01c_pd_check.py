"""PD discrepancy between native and fast pipelines vs DSI (angle ill-defined when DSI~0)."""
import sys, json; sys.path.insert(0, "/workspace/code")
import numpy as np, fastsim as fs, stimuli as S
from flyvis import NetworkView
from flyvis.analysis.stimulus_responses import moving_edge_responses
from flyvis.analysis import moving_bar_responses as MB
out = {}
for i in [0, 1, 2]:
    me = moving_edge_responses(NetworkView(fs.MODEL_NAMES[i]), speeds=(19,), dt=1 / 200)
    sim = fs.FastSim(fs.model_params(i)); er, edf, ecfg = S.simulate_edges(sim, (19,), 1 / 200)
    m = S.edge_metrics(er, edf, ecfg)
    dn = np.asarray(MB.direction_selectivity_index(me).squeeze().transpose('intensity', 'neuron').values)
    pn = np.asarray(MB.preferred_direction(me).squeeze().transpose('intensity', 'neuron').values)
    d = np.abs(np.angle(np.exp(1j * (pn - m["pd"]))))
    out[i] = dict(max_pd_diff_dsi_gt_0p01=float(d[dn > 0.01].max()), max_pd_diff_all=float(d.max()),
                  dsi_at_max_diff=float(dn.ravel()[d.argmax()]))
print(out); json.dump(out, open("/workspace/outputs/01c_pd_check.json", "w"), indent=1)
