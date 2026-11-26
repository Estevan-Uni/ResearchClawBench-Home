import json, os, numpy as np, pandas as pd
O = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'outputs')
def load(f):
    d = pd.DataFrame([json.loads(l) for l in open(os.path.join(O, f))]); return d.drop_duplicates(['config', 'idx']).set_index(['config', 'idx'])
def cp_at(tr, t):
    tr = np.array([x[:2] for x in tr], float); i = np.searchsorted(tr[:, 0], t, side='right') - 1
    return tr[max(i, 0), 1]
L = load('res_LNS2.jsonl'); rows = []
for name, f in [('LNS2+RL W=8', 'res_HYB_rate_W8.jsonl'), ('LNS2 seed1', 'res_LNS2_seed1.jsonl'), ('W=full', 'res_ABL_HYB_rate_Wfull.jsonl'), ('heuristic', 'res_ABL_HYB_rate_W8_heur.jsonl'), ('patience', 'res_ABL_HYB_pat_W8.jsonl')]:
    H = load(f); J = H.join(L, rsuffix='_l', how='inner')
    J = J[J.family != 'random_large']
    out = {'variant': name, 'n': len(J)}
    for fr in [0.025, 0.05, 0.1, 0.25, 0.5, 1.0]:
        r = [np.log((cp_at(a, fr * b) + 1) / (cp_at(c, fr * b) + 1)) for a, c, b in zip(J.trace, J.trace_l, J.budget)]
        out[f'{fr:g}B'] = round(float(np.exp(np.mean(r))), 3)
    rows.append(out)
T = pd.DataFrame(rows); print(T.to_string()); T.to_csv(os.path.join(O, 'table_anytime_ratio.csv'), index=False)
