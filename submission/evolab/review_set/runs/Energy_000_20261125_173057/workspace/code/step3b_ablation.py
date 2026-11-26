"""Step 3b: identifiability ablation of MMGA on the 5 synthetic truths:
 (i) measurement-noise level (2 mV/0.1 K, 5 mV/0.3 K, 10 mV/0.5 K),
 (ii) information content of the objective (V+T+capacity vs V+capacity only, i.e. no temperature sensor).
Truth curves (refined PyBaMM) are reused from step3_curves.pkl; new noise realisations are drawn."""
import sys, os, json, pickle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from multiprocessing import Pool
from ecat_model import CORE, unit_to_phys, features
from ga import ga
from step3_identify import objective, distinct_topk, init, work, SIG_V, SIG_Q

O = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'outputs')
def objV(V, T, te, tgt):
    return np.mean((V - tgt[0]) ** 2, -1) / SIG_V ** 2 + ((te - tgt[2]) / tgt[2]) ** 2 / SIG_Q ** 2

CONFIGS = [('VT_noise2mV', 0.002, 0.1, objective), ('VT_noise5mV', 0.005, 0.3, objective),
           ('VT_noise10mV', 0.010, 0.5, objective), ('Vonly_noise2mV', 0.002, 0.1, objV)]

if __name__ == '__main__':
    S = json.load(open(os.path.join(O, 'step3_synthetic.json'))); CV = pickle.load(open(os.path.join(O, 'step3_curves.pkl'), 'rb'))
    mm = pickle.load(open(os.path.join(O, 'ann_core.pkl'), 'rb'))['mm']
    pool = Pool(2, initializer=init); rng = np.random.default_rng(77); out = []
    for name, nv, nt, fob in CONFIGS:
        for c, s in enumerate(S):
            t, V, T = CV[c]['truth']; x_t = np.array(s['x_true'])
            tgt = features(dict(t=t, V=V + nv * rng.standard_normal(len(V)), T=T + nt * rng.standard_normal(len(T))))
            allP, allF, seedX = [], [], []
            for sd in range(5):
                P, F, h = ga(lambda U: fob(*mm.predict(U), tgt), 4, 200, 100, seed=sd); allP.append(P); allF.append(F)
                seedX.append(unit_to_phys(P[:1], CORE)[0])
            Pc, Fc = distinct_topk(np.vstack(allP), np.concatenate(allF), 10)
            outs = pool.map(work, list(Pc))
            Jp = np.array([fob(o[0], o[1], o[2], tgt) if o is not None else 1e9 for o in outs]); b = int(np.argmin(Jp))
            x = unit_to_phys(Pc[b][None], CORE)[0]; seedX = np.array(seedX)
            out.append(dict(config=name, case=c, x_true=x_t.tolist(), x_id=x.tolist(),
                            rel_err_pct={n: float(100 * (x[j] - x_t[j]) / x_t[j]) for j, n in enumerate(CORE)},
                            seed_cv_pct={n: float(100 * seedX[:, j].std() / seedX[:, j].mean()) for j, n in enumerate(CORE)},
                            J_phys=float(Jp[b])))
            print(name, c, {k: round(v, 2) for k, v in out[-1]['rel_err_pct'].items()}, flush=True)
            json.dump(out, open(os.path.join(O, 'step3b_ablation.json'), 'w'), indent=1)
    pool.close()
