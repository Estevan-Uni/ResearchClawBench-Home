"""Analysis of saved encoder features (from probe_encoders.py): 5-fold linear probe + thumbnail ridge R^2."""
import json, numpy as np
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score
d = np.load("/workspace/outputs/probes/encoder_feats.npz"); Y, TH, PS = d["Y"], d["TH"], d["PS"]
NAMES = ["tench", "springer", "cassette player", "chainsaw", "church", "French horn", "garbage truck", "gas pump", "golf ball", "parachute"]
out = {"n_images": int(len(Y)), "n_classes": 10, "vq_psnr_mean": float(PS.mean()), "vq_psnr_std": float(PS.std()), "class_names": NAMES}
for k in ["siglip_raw_mean", "und_mean", "und_grid", "gen_mean", "gen_grid"]:
    v = d[k]; accs = []; r2s = []
    for seed in range(3):
        cv = StratifiedKFold(5, shuffle=True, random_state=seed)
        pred = cross_val_predict(make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=5000)), v, Y, cv=cv)
        accs.append(float((pred == Y).mean()))
        thp = cross_val_predict(make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-1, 5, 13))), v, TH, cv=KFold(5, shuffle=True, random_state=seed))
        r2s.append(float(r2_score(TH, thp, multioutput="variance_weighted")))
        if seed == 0:
            cm = np.zeros((10, 10), int)
            for a, b in zip(Y, pred): cm[a, b] += 1
    out[k] = {"dim": int(v.shape[1]), "probe_acc": float(np.mean(accs)), "probe_acc_std": float(np.std(accs)),
              "thumb_r2": float(np.mean(r2s)), "thumb_r2_std": float(np.std(r2s)), "confusion": cm.tolist()}
    print(f"{k:16s} dim={v.shape[1]:5d} acc={np.mean(accs):.3f}±{np.std(accs):.3f}  thumbR2={np.mean(r2s):.3f}±{np.std(r2s):.3f}")
print("VQ PSNR", PS.mean(), PS.std())
json.dump(out, open("/workspace/outputs/probes/encoder_probe.json", "w"), indent=1)
