"""Choose the layer bands for rel-att and grad-att on the CALIBRATION split only (disjoint images from eval).
Criterion: mean over calib questions of the attention ratio inside the pseudo-GT evidence box
(mean map value inside box / mean map value over the image), averaged over a contiguous 5-layer window.
usage: python calibrate_layers.py MODEL"""
import sys, os, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_textvqa import layer_ratios, grounding

model = sys.argv[1]; WIN = 5
gt = grounding()
R = layer_ratios(model, 'calib', gt)
out = {'n': int(len(R['rel']))}
for k in ['raw', 'rel', 'grad']:
    curve = np.nanmean(np.log(np.clip(R[k], 1e-6, None)), 0)  # geometric mean of ratios (robust)
    w = np.array([curve[i:i + WIN].mean() for i in range(len(curve) - WIN + 1)])
    s = int(np.argmax(w[2:])) + 2  # skip the first two layers (degenerate maps)
    out[k] = dict(curve_geo=np.exp(curve).tolist(), best_start=s, band=[s, s + WIN], band_ratio=float(np.exp(w[s])))
    print(k, 'best band', [s, s + WIN], 'geo-mean ratio %.2f' % np.exp(w[s]),
          'per-layer', np.round(np.exp(curve), 2).tolist())
json.dump(out, open(f'/workspace/outputs/analysis/calib_layers_{model}.json', 'w'), indent=1)
