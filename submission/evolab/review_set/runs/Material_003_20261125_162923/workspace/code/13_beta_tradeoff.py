"""beta trade-off: compare VAE runs at matched epoch (8) plus the final main run."""
import pandas as pd, os
cols = ['val_rec_acid', 'val_rec_epox', 'val_kl', 'head_r2', 'head_mae_K', 'exact_acid', 'exact_epox', 'fg_rec_acid', 'fg_rec_epox',
        'fg_near_acid', 'fg_near_epox', 'fg_prior_acid', 'fg_prior_epox']
rows = []
for tag, beta in [('b002', 0.02), ('b01', 0.1), ('b05', 0.5)]:
    f = f'outputs/vae_{tag}_log.csv'
    if not os.path.exists(f): continue
    L = pd.read_csv(f)
    e = L[L.epoch == min(8, L.epoch.max())].iloc[0]
    rows.append(dict(run=f'beta={beta} @ epoch {int(e.epoch)}', **{c: e[c] for c in cols}))
L = pd.read_csv('outputs/vae_b01_log.csv'); e = L.iloc[-1]
rows.append(dict(run=f'beta=0.1 @ epoch {int(e.epoch)} (final, used)', **{c: e[c] for c in cols}))
T = pd.DataFrame(rows); T.to_csv('outputs/vae_beta_tradeoff.csv', index=False)
print(T.round(3).to_markdown(index=False))
