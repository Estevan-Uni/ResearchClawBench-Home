"""Step 2: ANN meta-model (4-layer fully connected network, Adam) mapping the identified parameters to the
fixed-length ECAT response [V(tau), dT(tau), t_end].  Per-epoch training/validation MSE is logged.
Usage: python step2_ann.py --db core3000 --tag core [--small baseline20] [--epochs 1500]"""
import sys, os, json, time, argparse, pickle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from sklearn.neural_network import MLPRegressor

ap = argparse.ArgumentParser()
ap.add_argument('--db', default='core3000'); ap.add_argument('--tag', default='core')
ap.add_argument('--small', default=''); ap.add_argument('--epochs', type=int, default=1500)
ap.add_argument('--hidden', default='128,128,128'); ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--ntest', type=int, default=300); ap.add_argument('--nval', type=int, default=300)
a = ap.parse_args()
O = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'outputs')
HID = tuple(int(h) for h in a.hidden.split(','))


from metamodel import MetaModel


def train(U, Y, Uv, Yv, mm, epochs, seed, lr0=2e-3, lr_min=2e-5, batch=64):
    net = MLPRegressor(hidden_layer_sizes=HID, activation='tanh', solver='adam', learning_rate_init=lr0,
                       batch_size=min(batch, len(U)), random_state=seed, alpha=1e-6, max_iter=1, shuffle=True)
    Z, Zv = mm.enc(Y), mm.enc(Yv)
    hist = dict(epoch=[], train_mse=[], val_mse=[], lr=[])
    for ep in range(epochs):
        net.partial_fit(U, Z)
        lr = lr_min + 0.5 * (lr0 - lr_min) * (1 + np.cos(np.pi * (ep + 1) / epochs))  # cosine decay
        net._optimizer.learning_rate = lr  # NOTE: overridden by sklearn's Adam (recomputed from learning_rate_init each step) -> effective lr constant
        if ep % 5 == 0 or ep == epochs - 1:
            hist['epoch'].append(ep + 1)
            hist['train_mse'].append(float(np.mean((net.predict(U) - Z) ** 2)))
            hist['val_mse'].append(float(np.mean((net.predict(Uv) - Zv) ** 2)))
            hist['lr'].append(lr)
    mm.net = net
    return hist


def metrics(mm, U, V, T, te):
    Vp, Tp, tp = mm.predict(U)
    Z, Zp = (mm._Y(V, T, te) - mm.mu) / mm.sd, (mm._Y(Vp, Tp, tp) - mm.mu) / mm.sd  # unweighted standardised
    r2 = lambda y, yp: float(1 - np.sum((y - yp) ** 2) / np.sum((y - y.mean()) ** 2))
    return dict(mse_normalised=float(np.mean((Z - Zp) ** 2)),
                mse_V_V2=float(np.mean((V - Vp) ** 2)), mse_T_K2=float(np.mean((T - Tp) ** 2)),
                rmse_V_mV=float(1e3 * np.sqrt(np.mean((V - Vp) ** 2))),
                rmse_V_mV_p95=float(np.percentile(1e3 * np.sqrt(np.mean((V - Vp) ** 2, 1)), 95)),
                rmse_T_K=float(np.sqrt(np.mean((T - Tp) ** 2))),
                rmse_T_K_p95=float(np.percentile(np.sqrt(np.mean((T - Tp) ** 2, 1)), 95)),
                tend_mape_pct=float(100 * np.mean(np.abs(tp - te) / te)),
                tend_maxape_pct=float(100 * np.max(np.abs(tp - te) / te)),
                R2_V=r2(V, Vp), R2_T=r2(T, Tp), R2_tend=r2(te, tp))


if __name__ == '__main__':
    d = np.load(os.path.join(O, f'db_{a.db}.npz'))
    ok = d['ok']; U, V, T, te = d['U'][ok], d['V'][ok], d['T'][ok], d['tend'][ok]
    rng = np.random.default_rng(a.seed); idx = rng.permutation(len(U))
    te_i, va_i, tr_i = idx[:a.ntest], idx[a.ntest:a.ntest + a.nval], idx[a.ntest + a.nval:]
    nV, nT = V.shape[1], T.shape[1]
    res = dict(db=a.db, hidden=HID, n_weight_layers=len(HID) + 1, activation='tanh', optimizer='Adam',
               epochs=a.epochs, n_train=int(len(tr_i)), n_val=int(len(va_i)), n_test=int(len(te_i)),
               n_inputs=int(U.shape[1]), n_outputs=int(nV + nT + 1))
    mm = MetaModel(nV, nT); mm.fit_scaler(mm._Y(V[tr_i], T[tr_i], te[tr_i]))
    t0 = time.time()
    hist = train(U[tr_i], mm._Y(V[tr_i], T[tr_i], te[tr_i]), U[va_i], mm._Y(V[va_i], T[va_i], te[va_i]), mm,
                 a.epochs, a.seed)
    res['train_time_s'] = time.time() - t0
    res['history_first_last'] = dict(train_mse_initial=hist['train_mse'][0], train_mse_final=hist['train_mse'][-1],
                                     val_mse_initial=hist['val_mse'][0], val_mse_final=hist['val_mse'][-1])
    res['test'] = metrics(mm, U[te_i], V[te_i], T[te_i], te[te_i])
    res['train'] = metrics(mm, U[tr_i], V[tr_i], T[tr_i], te[tr_i])
    # evaluation time of the ANN (batch of 1 and batch of 200)
    t0 = time.time(); [mm.predict(U[te_i[:1]]) for _ in range(200)]; res['ann_time_single_s'] = (time.time() - t0) / 200
    t0 = time.time(); [mm.predict(U[te_i[:200]]) for _ in range(20)]; res['ann_time_per_sample_batch200_s'] = (time.time() - t0) / 20 / 200
    dbinfo = json.load(open(os.path.join(O, f'db_{a.db}.json')))
    res['physics_time_per_sim_s'] = dbinfo['mean_time_per_sim_s']
    res['speedup_single'] = res['physics_time_per_sim_s'] / res['ann_time_single_s']
    res['speedup_batch'] = res['physics_time_per_sim_s'] / res['ann_time_per_sample_batch200_s']
    hists = dict(full=hist)
    if a.small:   # small-data check: same architecture trained on the small LHS baseline only
        ds = np.load(os.path.join(O, f'db_{a.small}.npz')); oks = ds['ok']
        Us, Vs, Ts, ts = ds['U'][oks], ds['V'][oks], ds['T'][oks], ds['tend'][oks]
        ms = MetaModel(nV, nT); ms.fit_scaler(ms._Y(Vs, Ts, ts))
        hs = train(Us, ms._Y(Vs, Ts, ts), U[va_i], ms._Y(V[va_i], T[va_i], te[va_i]), ms, a.epochs, a.seed, batch=8)
        res['small_data'] = dict(n_train=int(len(Us)), test=metrics(ms, U[te_i], V[te_i], T[te_i], te[te_i]),
                                 train=metrics(ms, Us, Vs, Ts, ts))
        # report the small model's test error in the full model's normalisation for comparability
        Vp, Tp, tp = ms.predict(U[te_i])
        res['small_data']['test_mse_in_full_normalisation'] = float(np.mean(((mm._Y(V[te_i], T[te_i], te[te_i])) - mm._Y(Vp, Tp, tp)) ** 2 / mm.sd ** 2))
        hists['small'] = hs
    pickle.dump(dict(mm=mm, split=dict(test=te_i, val=va_i, train=tr_i)), open(os.path.join(O, f'ann_{a.tag}.pkl'), 'wb'))
    json.dump(hists, open(os.path.join(O, f'ann_{a.tag}_history.json'), 'w'))
    json.dump(res, open(os.path.join(O, f'ann_{a.tag}_metrics.json'), 'w'), indent=2)
    print(json.dumps(res, indent=2))
