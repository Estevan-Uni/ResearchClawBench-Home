"""Structure-quality metrics (numpy)."""
import numpy as np


def kabsch(P, Q, w=None):
    """Return R,t minimising ||(P R^T + t) - Q||, P,Q (N,3)."""
    if w is None:
        w = np.ones(len(P))
    w = w / w.sum()
    mp, mq = (P * w[:, None]).sum(0), (Q * w[:, None]).sum(0)
    H = ((P - mp) * w[:, None]).T @ (Q - mq)
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    return R, mq - mp @ R.T


def apply(R, t, X):
    return X @ R.T + t


def rmsd(A, B):
    return float(np.sqrt(((A - B) ** 2).sum(-1).mean()))


def lddt(pred, ref, mask_pairs=None, cutoff=15.0, thresholds=(0.5, 1, 2, 4)):
    """Per-structure lDDT over all pairs (or a boolean pair mask) within cutoff in reference."""
    dr = np.linalg.norm(ref[:, None] - ref[None], axis=-1)
    dp = np.linalg.norm(pred[:, None] - pred[None], axis=-1)
    m = (dr < cutoff) & ~np.eye(len(ref), dtype=bool)
    if mask_pairs is not None:
        m &= mask_pairs
    dd = np.abs(dr - dp)[m]
    return float(np.mean([(dd < t).mean() for t in thresholds]))


def contacts(X, idx_a, idx_b, cut=4.5):
    d = np.linalg.norm(X[idx_a][:, None] - X[idx_b][None], axis=-1)
    return d < cut


def evaluate(pred, f, ref=None):
    """pred (N,3); f numpy feature dict (with coords = reference). Returns metric dict."""
    ref = f['coords'] if ref is None else ref
    mol = f['atom_moltype']
    prot = np.where(mol == 0)[0]
    lig = np.where(mol == 3)[0]
    ca = f['token_center_atom'][f['moltype'] == 0]
    out = {}
    R, t = kabsch(pred[ca], ref[ca])
    P = apply(R, t, pred)
    out['prot_ca_rmsd'] = rmsd(P[ca], ref[ca])
    out['prot_heavy_rmsd'] = rmsd(P[prot], ref[prot])
    out['lig_rmsd_protaligned'] = rmsd(P[lig], ref[lig])
    # pocket-aligned ligand RMSD: align on CA of residues with any heavy atom within 10 A of ligand
    dmin = np.linalg.norm(ref[prot][:, None] - ref[lig][None], axis=-1).min(1)
    pocket_tok = np.unique(f['atom_to_token'][prot[dmin < 10.0]])
    pocket_ca = f['token_center_atom'][pocket_tok]
    R2, t2 = kabsch(pred[pocket_ca], ref[pocket_ca])
    P2 = apply(R2, t2, pred)
    out['lig_rmsd_pocketaligned'] = rmsd(P2[lig], ref[lig])
    out['pocket_ca_rmsd'] = rmsd(P2[pocket_ca], ref[pocket_ca])
    # ligand internal RMSD (shape only)
    R3, t3 = kabsch(pred[lig], ref[lig])
    out['lig_internal_rmsd'] = rmsd(apply(R3, t3, pred[lig]), ref[lig])
    out['lddt_ca'] = lddt(pred[ca], ref[ca])
    out['lddt_all'] = lddt(pred, ref)
    # interface lDDT: only protein-ligand pairs
    ism = np.zeros((len(ref), len(ref)), bool)
    ism[np.ix_(prot, lig)] = True; ism[np.ix_(lig, prot)] = True
    out['lddt_interface'] = lddt(pred, ref, ism, cutoff=6.0)
    # fraction of native protein-ligand heavy-atom contacts (residue level)
    a2t = f['atom_to_token']
    def res_contacts(X):
        c = contacts(X, prot, lig, 4.5)
        pairs = set()
        for i, j in zip(*np.where(c)):
            pairs.add((a2t[prot[i]], j))
        return pairs
    nat, prd = res_contacts(ref), res_contacts(pred)
    out['fnat'] = len(nat & prd) / max(len(nat), 1)
    out['fnonnat'] = len(prd - nat) / max(len(prd), 1)
    # clashes: heavy atoms of different tokens closer than 1.5 A, excluding bonded neighbours (<2.0 A in ref)
    D = np.linalg.norm(pred[:, None] - pred[None], axis=-1)
    Dr = np.linalg.norm(ref[:, None] - ref[None], axis=-1)
    diff_tok = a2t[:, None] != a2t[None]
    cl = (D < 1.5) & diff_tok & (Dr > 2.0)
    out['n_clashes'] = int(np.triu(cl, 1).sum())
    # ligand bond lengths
    lb = f['lig_bonds']
    if len(lb):
        bp = np.linalg.norm(pred[lb[:, 0]] - pred[lb[:, 1]], axis=-1)
        br = np.linalg.norm(ref[lb[:, 0]] - ref[lb[:, 1]], axis=-1)
        out['lig_bond_mae'] = float(np.abs(bp - br).mean())
    # protein CA chirality (sign of N-CA-C-CB improper) agreement
    names = f['atom_names']
    chi_ok, n = 0, 0
    for tk in np.unique(a2t[prot]):
        at = np.where(a2t == tk)[0]
        nm = {names[i]: i for i in at}
        if all(k in nm for k in ('N', 'CA', 'C', 'CB')):
            def vol(X):
                a, b, c, d = X[nm['CA']], X[nm['N']], X[nm['C']], X[nm['CB']]
                return np.dot(b - a, np.cross(c - a, d - a))
            chi_ok += np.sign(vol(pred)) == np.sign(vol(ref)); n += 1
    out['ca_chirality_ok'] = chi_ok / max(n, 1)
    # protein backbone peptide-bond lengths C(i)-N(i+1)
    return out


def ligand_dock_baseline(f, ref_lig_conf, n=50, seed=0):
    """Baseline: rigid ETKDG reference conformer of the ligand placed with random orientation at a
    random protein-surface point (native protein kept). Returns list of pocket-aligned ligand RMSDs."""
    rng = np.random.default_rng(seed)
    ref = f['coords']
    mol = f['atom_moltype']
    prot = np.where(mol == 0)[0]; lig = np.where(mol == 3)[0]
    cen = ref[prot].mean(0)
    rg = np.sqrt(((ref[prot] - cen) ** 2).sum(-1).mean())
    res = []
    for _ in range(n):
        q = rng.normal(size=4); q /= np.linalg.norm(q)
        a, b, c, d = q
        R = np.array([[a*a+b*b-c*c-d*d, 2*(b*c-a*d), 2*(b*d+a*c)],
                      [2*(b*c+a*d), a*a-b*b+c*c-d*d, 2*(c*d-a*b)],
                      [2*(b*d-a*c), 2*(c*d+a*b), a*a-b*b-c*c+d*d]])
        u = rng.normal(size=3); u /= np.linalg.norm(u)
        X = (ref_lig_conf - ref_lig_conf.mean(0)) @ R.T + cen + u * (rg + 5.0)
        res.append(rmsd(X, ref[lig]))
    return res
