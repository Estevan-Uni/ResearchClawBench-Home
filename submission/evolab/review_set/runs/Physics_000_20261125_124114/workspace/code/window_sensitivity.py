"""Sensitivity of LJ stability windows to the reference energy e_b in
Delta = (E - N e_b)/N^(2/3) (e_b plays the role of the chemical potential of the
depositing species).  -> ../outputs/window_sensitivity.json"""
import json, numpy as np
import icoshell as ic
SC = json.load(open('../outputs/scans_LJ.json'))
SC['c03']['results'].update(SC['c03b']['results'])
out = {}
for eb in [-6.0, -7.0, -8.61, -10.0]:
    out[str(eb)] = {}
    for core in ['c01', 'c11', 'c02', 'c12', 'c03', 'c22', 'c13']:
        res = SC[core]['results']
        g = np.array(list(res.values())[0]['grid'])
        M = []
        names = list(res.keys())
        for r in res.values():
            E = np.array(r['E']); N = r['N']; ok = np.array(r['shift']) < 3
            M.append(np.where(ok, (E - eb * N) / N ** (2 / 3), np.inf))
        w = np.argmin(np.array(M), 0)
        seg = []
        for d, i in zip(g, w):
            if seg and seg[-1][0] == names[i]:
                seg[-1][2] = float(d)
            else:
                seg.append([names[i], float(d), float(d)])
        out[str(eb)][core] = seg
json.dump(out, open('../outputs/window_sensitivity.json', 'w'), indent=1)
for eb, v in out.items():
    print(eb, {k: [(s[0], s[1], s[2]) for s in seg] for k, seg in v.items() if k in ('c01', 'c12', 'c13', 'c03')})
