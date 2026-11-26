"""Encoder dissociation probe on Imagenette (10 classes x N val images).
For each image we extract (a) understanding features: SigLIP -> und. adaptor, mean-pooled (2048-d)
and (b) generation features: VQ codes -> gen_embed -> gen. adaptor, mean-pooled (2048-d),
plus spatial variants (4x4 avg-pooled grids). Then 5-fold CV:
  * linear probe (logistic regression) for semantic class  -> high-level semantics
  * ridge regression to an 8x8 RGB thumbnail (R^2)          -> low-level / spatial detail
Outputs /workspace/outputs/probes/encoder_probe.json"""
import os, sys, glob, json, numpy as np, PIL.Image, time
from janus_common import load, torch
import torch.nn.functional as F
N = int(sys.argv[1]) if len(sys.argv) > 1 else 12
R = "/workspace/outputs/ext/imagenette/imagenette2-160/val"
NAMES = {"n01440764": "tench", "n02102040": "springer", "n02979186": "cassette player", "n03000684": "chainsaw",
         "n03028079": "church", "n03394916": "French horn", "n03417042": "garbage truck", "n03425413": "gas pump",
         "n03445777": "golf ball", "n03888257": "parachute"}
proc, model = load(); ip = proc.image_processor
X = {k: [] for k in ["und_mean", "gen_mean", "und_grid", "gen_grid", "siglip_raw_mean"]}; Y = []; TH = []; PS = []
t0 = time.time()
with torch.inference_mode():
    for ci, w in enumerate(sorted(NAMES)):
        files = sorted(glob.glob(f"{R}/{w}/*.JPEG"))[:N]
        assert files, w
        for f in files:
            im = PIL.Image.open(f).convert("RGB")
            px = ip([im], return_tensors="pt").pixel_values          # [1,3,384,384]
            sig = model.vision_model(px); und = model.aligner(sig)[0]
            _, _, info = model.gen_vision_model.encode(px); codes = info[2].view(-1)
            gen = model.prepare_gen_img_embeds(codes)
            rec = model.gen_vision_model.decode_code(codes.view(1, -1), shape=[1, 8, 24, 24])
            PS.append(float(10 * np.log10(4 / F.mse_loss(rec, px).item())))  # range [-1,1] -> peak-to-peak 2
            g = lambda e: F.adaptive_avg_pool2d(e.view(24, 24, -1).permute(2, 0, 1)[None], 4)[0].flatten().numpy()
            X["und_mean"].append(und.mean(0).numpy()); X["gen_mean"].append(gen.mean(0).numpy())
            X["und_grid"].append(g(und)); X["gen_grid"].append(g(gen)); X["siglip_raw_mean"].append(sig[0].mean(0).numpy())
            Y.append(ci)
            TH.append(np.asarray(im.resize((8, 8), PIL.Image.BOX), dtype=np.float32).flatten() / 255)
        print(w, NAMES[w], len(Y), f"{time.time()-t0:.0f}s", flush=True)
Y = np.array(Y); TH = np.array(TH)
np.savez_compressed("/workspace/outputs/probes/encoder_feats.npz", Y=Y, TH=TH, PS=np.array(PS), **{k: np.array(v) for k, v in X.items()})
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score
cv = StratifiedKFold(5, shuffle=True, random_state=0); out = {"n_images": int(len(Y)), "n_classes": 10, "vq_psnr_mean": float(np.mean(PS))}
for k, v in X.items():
    v = np.array(v)
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=5000))
    pred = cross_val_predict(clf, v, Y, cv=cv); acc = float((pred == Y).mean())
    reg = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-1, 5, 13)))
    thp = cross_val_predict(reg, v, TH, cv=cv); r2 = float(r2_score(TH, thp, multioutput="variance_weighted"))
    cm = np.zeros((10, 10), int)
    for a, b in zip(Y, pred): cm[a, b] += 1
    out[k] = {"probe_acc": acc, "thumb_r2": r2, "confusion": cm.tolist()}
    print(k, acc, r2, flush=True)
out["class_names"] = [NAMES[w] for w in sorted(NAMES)]
json.dump(out, open("/workspace/outputs/probes/encoder_probe.json", "w"), indent=1)
