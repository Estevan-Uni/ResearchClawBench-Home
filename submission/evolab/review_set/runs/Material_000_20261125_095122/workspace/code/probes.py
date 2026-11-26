"""Linear probes on frozen embeddings: pretrained (masked-node) vs random-init encoder."""
import sys; sys.path.insert(0, '/workspace/code')
import numpy as np, pandas as pd, torch
torch.set_num_threads(1)
from sklearn.linear_model import RidgeCV, LogisticRegression
from sklearn.model_selection import cross_val_predict, StratifiedKFold, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from common import *
from model import *
from control_labels import has_motif
D = load_cache(); gs = D['pretrain']; items = featurize(gs)
desc = pd.read_csv('outputs/descriptors_pretrain.csv')
targets = {'frac_mag': desc.frac_mag.values, 'mean_bond': desc.mean_bond.values, 'n_atoms': desc.n_atoms.values.astype(float),
           'n_components': desc.n_components.values.astype(float)}
cls_targets = {'motif (M-X-M\' bridge)': np.array([has_motif(i) for i in items]).astype(int), 'provided label y': desc.y.values}
def emb(enc):
    m = Classifier(enc); m.eval(); out = []
    with torch.no_grad():
        for k in range(0, len(items), 500): out.append(m.embed(collate(items[k:k + 500])).numpy())
    return np.vstack(out)
encs = {}
for name in sys.argv[1].split(','):
    torch.manual_seed(0); e = Encoder()
    if name != 'random_init': e.load_state_dict(torch.load(f'outputs/encoder_{name}.pt'))
    encs[name] = emb(e)
rows = []
for name, Z in encs.items():
    for t, yv in targets.items():
        p = cross_val_predict(make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 13))), Z, yv, cv=KFold(5, shuffle=True, random_state=0))
        rows.append(dict(encoder=name, target=t, metric='R2', value=1 - ((yv - p) ** 2).sum() / ((yv - yv.mean()) ** 2).sum()))
    for t, yv in cls_targets.items():
        p = cross_val_predict(make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=3000)), Z, yv, cv=StratifiedKFold(5, shuffle=True, random_state=0), method='predict_proba')[:, 1]
        rows.append(dict(encoder=name, target=t, metric='AUC', value=auc(yv, p)))
R = pd.DataFrame(rows); print(R.to_string()); R.to_csv('outputs/probes.csv', index=False)
