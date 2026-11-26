"""Summarize the classical gap: per-layer MPS log-fidelity decay vs experiment; chi needed (log-log extrapolation, marked EXTRAPOLATED)."""
import os, json, numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
O = lambda f: os.path.join(ROOT, 'outputs', f)
M = pd.read_csv(O('mps_N56_depth.csv')); P = pd.read_csv(O('pooled_estimates.csv'))
mb = P[(P.family == 'N56_depths') & (P.method == 'MB')].set_index('d').F
R = json.load(open(O('model_results.json'))); lam_exp = R['depth_regressions']['N56_MB_d8-20']['lam']
rows = []
for (g, chi), gg in M.groupby(['geometry', 'chi']):
    m = gg.groupby('d').logF.mean()
    late = m[(m.index >= 12) & (m.index <= 24)]
    slope = -np.polyfit(late.index, late.values, 1)[0]
    row = dict(geometry=g, chi=chi, decay_per_layer_d12_24=slope)
    for d in [8, 12, 16, 20, 24]:
        row[f'logF_d{d}'] = m.get(d, np.nan); row[f'gap_e_folds_d{d}'] = np.log(mb.get(d, np.nan)) - m.get(d, np.nan)
    rows.append(row)
T = pd.DataFrame(rows); T['exp_decay_per_layer'] = lam_exp
# chi needed: fit logF(d) = a + b log2(chi) per geometry,d using chi>=32, extrapolate to ln F_exp
need = []
for g in T.geometry.unique():
    for d in [8, 12, 16, 20, 24]:
        sub = M[(M.geometry == g) & (M.d == d)].groupby('chi').logF.mean()
        sub = sub[sub.index >= 16]
        if len(sub) < 2: continue
        b, a = np.polyfit(np.log2(sub.index), sub.values, 1)
        target = np.log(mb.get(d, np.nan))
        if sub.values.max() >= target:
            chi_req = float(sub.index[np.argmax(sub.values >= target)]); extrap = False
        else:
            chi_req = 2 ** ((target - a) / b) if b > 0 else np.inf; extrap = True
        need.append(dict(geometry=g, d=d, lnF_exp=target, slope_per_chi_doubling=b, log2_chi_required=np.log2(chi_req), extrapolated=extrap,
                         mps_memory_bytes_log10=np.log10(56 * 2 * chi_req ** 2 * 16)))
NE = pd.DataFrame(need)
T.to_csv(O('classical_gap_summary.csv'), index=False); NE.to_csv(O('mps_chi_required.csv'), index=False)
pd.set_option('display.width', 250)
print(T.round(3).to_string()); print(NE.round(3).to_string())
