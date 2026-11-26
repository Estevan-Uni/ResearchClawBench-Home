"""'Virtual MD + GP calibration' oracle used to score generated vitrimers.
(1) MD surrogate: GP with Tanimoto kernel on binary Morgan FP (r=2, 2048) of the repeat unit, trained on
    all 8,424 MD Tg values (independent of the VAE parameters, but same data -> in-silico consistency only).
(2) GP calibration (step 2) maps MD Tg -> experimental-scale Tg with uncertainty."""
import sys, pickle, numpy as np
sys.path.insert(0, 'code')
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
RDLogger.DisableLog('rdApp.*')
import importlib
ru_mod = importlib.import_module('01_repeat_units')
from gp_lib import minmax_kernel

_gen = AllChem.GetMorganGenerator(radius=2, fpSize=2048)


def bitfp(smiles):
    return np.stack([_gen.GetFingerprintAsNumPy(Chem.MolFromSmiles(s)).astype(np.float32) for s in smiles])


def tanimoto(A, B):
    inter = A @ B.T
    return inter / (A.sum(1)[:, None] + B.sum(1)[None] - inter + 1e-9)


class Oracle:
    def __init__(self, path='outputs/oracle.pkl', calpath='outputs/gp_calibration.pkl'):
        d = pickle.load(open(path, 'rb'))
        self.__dict__.update(d)
        c = pickle.load(open(calpath, 'rb'))
        self.cal = c['models'][c['best']]; self.cal_fp = c['fp_norm_train']; self.desc_norm = c['desc_norm']

    def md(self, rus):
        X = bitfp(rus)
        Ks = self.s * tanimoto(X, self.Xtr)
        mu = self.m + Ks @ self.alpha
        return mu, Ks.max(1) / self.s

    def calibrate(self, rus, md, md_std=None):
        cfp = np.stack([ru_mod.count_fp(s) for s in rus]); cfp = cfp / cfp.sum(1, keepdims=True)
        Kv = minmax_kernel(cfp, self.cal_fp)
        desc = np.stack([ru_mod.intensive_desc(s)[0] for s in rus])
        Xte = {'Kfp': Kv, 'md_te': ((np.asarray(md) - 400) / 100)[:, None], 'desc_te': (desc - self.desc_norm[0]) / self.desc_norm[1]}
        std = np.full(len(rus), 24.6) if md_std is None else md_std
        mu, vf, nv = self.cal.predict(Xte, std)
        return np.asarray(md) + mu, np.sqrt(vf + nv)

    def score_pairs(self, acids, epoxs):
        rus = [ru_mod.build_repeat_unit(a, e) for a, e in zip(acids, epoxs)]
        ok = np.array([r is not None for r in rus])
        md = np.full(len(rus), np.nan); cal = np.full(len(rus), np.nan); sig = np.full(len(rus), np.nan); sim = np.full(len(rus), np.nan)
        if ok.any():
            r_ok = [r for r in rus if r is not None]
            m, s = self.md(r_ok); md[ok] = m; sim[ok] = s
            c, sg = self.calibrate(r_ok, m); cal[ok] = c; sig[ok] = sg
        return dict(ru=rus, ok=ok, md=md, cal=cal, sigma_cal=sig, maxsim_lib=sim)
