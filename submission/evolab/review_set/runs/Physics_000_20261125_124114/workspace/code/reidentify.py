"""Re-identify final (as-grown and annealed) structures of all runs with
recall (coverage), precision and F1 against every candidate template.
-> adds 'reid' to ../outputs/growth/reid_<run>.json"""
import json, glob, os, numpy as np
import md_growth as md
from run_growth_exp import exp_cfg
cache = {}
for fn in sorted(glob.glob('../outputs/growth/*_s*.json')):
    out = fn.replace('/growth/', '/growth/reid_').replace('reid_reid_', 'reid_')
    if os.path.basename(fn).startswith('reid_') or (os.path.exists(out) and os.path.getmtime(out) > os.path.getmtime(fn)):
        continue
    r = json.load(open(fn))
    cfg = exp_cfg(r['name'])
    key = r['name']
    if key not in cache:
        cache[key] = md.Identifier(cfg['seed_path'], cfg['seed_d'], r['d_new'], cfg['cands'])
    idf = cache[key]
    lay = 'radial' if cfg.get('layer') == 'radial' else np.arange(r['nseed'], len(r['D']))
    cen = cfg.get('id_centre') or 0
    res = {}
    for st in ['X_grown', 'X_anneal']:
        X = np.array(r[st])
        res[st] = idf.identify(X, cen, lay)
    json.dump(res, open(out, 'w'))
    print(os.path.basename(fn), {k: round(v, 2) for k, v in res['X_anneal']['f1'].items()})
