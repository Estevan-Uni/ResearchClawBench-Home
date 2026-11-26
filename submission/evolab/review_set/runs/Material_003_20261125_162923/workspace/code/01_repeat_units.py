"""Step 1: build vitrimer repeat units (beta-hydroxy ester) from acid/epoxide pairs,
and compute fingerprints / size-intensive descriptors for calibration + vitrimer sets.

Repeat unit:  *-O-C(=O)-R_acid-C(=O)-O-CH2-CH(OH)-R_epox-CH(OH)-CH2-*
Epoxide ring opened at less-substituted carbon (CH2) by carboxylate.
"""
import numpy as np, pandas as pd, pickle, sys
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
RDLogger.DisableLog('rdApp.*')

ACID = Chem.MolFromSmarts('[CX3](=O)[OX2H1]')
OXI = Chem.MolFromSmarts('[CH2]1[CH1,CH0]O1')   # ring CH2 then substituted C, then O


def build_repeat_unit(acid_smi, epox_smi):
    a = Chem.MolFromSmiles(acid_smi); e = Chem.MolFromSmiles(epox_smi)
    if a is None or e is None:
        return None
    am = a.GetSubstructMatches(ACID); em = e.GetSubstructMatches(OXI)
    if len(am) != 2 or len(em) != 2:
        return None
    rw = Chem.RWMol(Chem.CombineMols(a, e))
    off = a.GetNumAtoms()
    # acid hydroxyl O indices
    acid_O = [m[2] for m in am]
    ep = [(m[0] + off, m[1] + off, m[2] + off) for m in em]  # CH2, C, O
    # open both epoxides: break CH2-O bond, O stays on substituted C as OH
    for ch2, c, o in ep:
        rw.RemoveBond(ch2, o)
    # ester bond: acid O(H) #1 -> CH2 of epoxide #1 (chain), acid O #2 & epoxide CH2 #2 become stars
    rw.AddBond(acid_O[0], ep[0][0], Chem.BondType.SINGLE)
    s1 = rw.AddAtom(Chem.Atom(0)); rw.AddBond(acid_O[1], s1, Chem.BondType.SINGLE)
    s2 = rw.AddAtom(Chem.Atom(0)); rw.AddBond(ep[1][0], s2, Chem.BondType.SINGLE)
    m = rw.GetMol()
    for at in m.GetAtoms():
        at.SetNoImplicit(False)
    try:
        Chem.SanitizeMol(m)
        smi = Chem.MolToSmiles(m)
        return smi if Chem.MolFromSmiles(smi) is not None else None
    except Exception:
        return None


def mol_for_fp(smi):
    """Replace * by H-capped atoms for descriptor calculation (stars -> removed)."""
    m = Chem.MolFromSmiles(smi)
    rw = Chem.RWMol(m)
    stars = [a.GetIdx() for a in m.GetAtoms() if a.GetAtomicNum() == 0]
    for i in sorted(stars, reverse=True):
        rw.RemoveAtom(i)
    m2 = rw.GetMol()
    for a in m2.GetAtoms():
        a.SetNoImplicit(False)
    Chem.SanitizeMol(m2)
    return m2


def count_fp(smi, radius=2, nbits=2048):
    m = Chem.MolFromSmiles(smi)
    gen = AllChem.GetMorganGenerator(radius=radius, fpSize=nbits)
    return gen.GetCountFingerprintAsNumPy(m).astype(np.float32)


DESC = ['FractionCSP3', 'NumRotatableBonds', 'NumAromaticRings', 'NumHDonors', 'NumHAcceptors',
        'TPSA', 'RingCount', 'NumAliphaticRings', 'MolWt', 'HeavyAtomCount', 'NHOHCount', 'NOCount']


def intensive_desc(smi):
    m = mol_for_fp(smi)
    n = m.GetNumHeavyAtoms()
    d = {k: getattr(Descriptors, k)(m) for k in DESC}
    out = [d['FractionCSP3'], d['NumRotatableBonds'] / n, d['NumAromaticRings'] / n * 6,
           d['NumHDonors'] / n, d['NumHAcceptors'] / n, d['TPSA'] / n, d['RingCount'] / n * 6,
           d['NumAliphaticRings'] / n * 6, d['MolWt'] / n, d['NHOHCount'] / n, d['NOCount'] / n,
           sum(a.GetIsAromatic() for a in m.GetAtoms()) / n,
           sum(a.GetAtomicNum() == 7 for a in m.GetAtoms()) / n,
           sum(a.GetAtomicNum() in (9, 17, 35, 53) for a in m.GetAtoms()) / n,
           sum(a.GetAtomicNum() == 16 for a in m.GetAtoms()) / n]
    return np.array(out, dtype=np.float32), n


if __name__ == '__main__':
    cal = pd.read_csv('data/tg_calibration.csv')
    vit = pd.read_csv('data/tg_vitrimer_MD.csv')
    ru = [build_repeat_unit(a, e) for a, e in zip(vit.acid, vit.epoxide)]
    vit['repeat_unit'] = ru
    print('repeat units valid:', vit.repeat_unit.notna().sum(), '/', len(vit))
    print(vit.repeat_unit.iloc[0])
    vit.to_csv('outputs/vitrimer_repeat_units.csv', index=False)
    cal_fp = np.stack([count_fp(s) for s in cal.smiles])
    vit_fp = np.stack([count_fp(s) for s in vit.repeat_unit])
    cal_d = [intensive_desc(s) for s in cal.smiles]
    vit_d = [intensive_desc(s) for s in vit.repeat_unit]
    feats = dict(cal_fp=cal_fp, vit_fp=vit_fp,
                 cal_desc=np.stack([x[0] for x in cal_d]), vit_desc=np.stack([x[0] for x in vit_d]),
                 cal_nheavy=np.array([x[1] for x in cal_d]), vit_nheavy=np.array([x[1] for x in vit_d]))
    np.savez_compressed('outputs/features.npz', **feats)
    print('heavy atoms cal %.1f+-%.1f  vit %.1f+-%.1f' % (feats['cal_nheavy'].mean(), feats['cal_nheavy'].std(),
                                                         feats['vit_nheavy'].mean(), feats['vit_nheavy'].std()))
