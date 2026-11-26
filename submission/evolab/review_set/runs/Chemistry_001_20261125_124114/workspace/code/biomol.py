"""Unified biomolecular input representation (AlphaFold3-style tokenisation).

* Standard protein residues and nucleotides -> one token per residue.
* Ligand / non-standard molecules        -> one token per heavy atom.
Every token owns >=1 atoms; every atom carries reference-conformer features
(ref_pos, element, charge, atom-name characters, ref_space_uid) that are
computed independently of the experimental structure (RDKit ETKDG).
"""
import numpy as np

PROTEIN_RES = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
               'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL']
RNA_RES = ['A', 'G', 'C', 'U']
DNA_RES = ['DA', 'DG', 'DC', 'DT']
# 32 residue classes, as in AF3: 20 aa + unk, 4 RNA + unk, 4 DNA + unk, gap
RESTYPES = PROTEIN_RES + ['UNK'] + RNA_RES + ['N'] + DNA_RES + ['DN', 'GAP']
RES2IDX = {r: i for i, r in enumerate(RESTYPES)}
MOLTYPES = ['protein', 'rna', 'dna', 'ligand']
ELEMENTS = ['H', 'C', 'N', 'O', 'S', 'P', 'F', 'CL', 'BR', 'I', 'SE', 'MG', 'ZN', 'FE', 'CA', 'NA', 'K', 'X']
ELEM2IDX = {e: i for i, e in enumerate(ELEMENTS)}
ONE2THREE = {'A': 'ALA', 'R': 'ARG', 'N': 'ASN', 'D': 'ASP', 'C': 'CYS', 'Q': 'GLN', 'E': 'GLU',
             'G': 'GLY', 'H': 'HIS', 'I': 'ILE', 'L': 'LEU', 'K': 'LYS', 'M': 'MET', 'F': 'PHE',
             'P': 'PRO', 'S': 'SER', 'T': 'THR', 'W': 'TRP', 'Y': 'TYR', 'V': 'VAL'}
THREE2ONE = {v: k for k, v in ONE2THREE.items()}
MOL_WEIGHT = {'protein': 1.0, 'rna': 5.0, 'dna': 5.0, 'ligand': 10.0}  # AF3 loss weights


# --------------------------------------------------------------------------- parsing
def parse_pdb(path, keep_h=False):
    """Return list of residues: dict(chain, resseq, resname, atoms=[(name, elem, charge, xyz)])."""
    residues, key2res = [], {}
    for line in open(path):
        if not line.startswith(('ATOM', 'HETATM')):
            continue
        if line.startswith('ENDMDL'):
            break
        name = line[12:16].strip()
        resname = line[17:20].strip()
        chain = line[21]
        resseq = int(line[22:26])
        icode = line[26]
        xyz = np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])
        elem_field = line[76:80].strip()
        elem = ''.join(c for c in elem_field if c.isalpha()).upper() or name[0]
        charge = 0
        if '+' in elem_field or '-' in elem_field:
            digits = ''.join(c for c in elem_field if c.isdigit()) or '1'
            charge = int(digits) * (1 if '+' in elem_field else -1)
        if elem == 'H' and not keep_h:
            continue
        key = (chain, resseq, icode)
        if key not in key2res:
            key2res[key] = dict(chain=chain, resseq=resseq, resname=resname, atoms=[])
            residues.append(key2res[key])
        key2res[key]['atoms'].append((name, elem, charge, xyz))
    return residues


def parse_sdf(path, keep_h=False):
    """Minimal V2000 SDF parser -> elements, charges, coords, bonds (heavy atoms only by default)."""
    lines = open(path).read().splitlines()
    n_at, n_b = int(lines[3][0:3]), int(lines[3][3:6])
    elems, coords, charges = [], [], []
    chg_map = {1: 3, 2: 2, 3: 1, 5: -1, 6: -2, 7: -3}
    for l in lines[4:4 + n_at]:
        coords.append([float(l[0:10]), float(l[10:20]), float(l[20:30])])
        elems.append(l[31:34].strip().upper())
        ccode = int(l[36:39]) if len(l) >= 39 and l[36:39].strip() else 0
        charges.append(chg_map.get(ccode, 0))
    bonds = []
    for l in lines[4 + n_at:4 + n_at + n_b]:
        bonds.append((int(l[0:3]) - 1, int(l[3:6]) - 1, int(l[6:9])))
    for l in lines[4 + n_at + n_b:]:  # M  CHG lines override
        if l.startswith('M  CHG'):
            f = l.split()
            for k in range(int(f[2])):
                charges[int(f[3 + 2 * k]) - 1] = int(f[4 + 2 * k])
    keep = [i for i, e in enumerate(elems) if keep_h or e != 'H']
    nh = {i: 0 for i in keep}
    for a, b, o in bonds:
        if elems[a] == 'H' and b in nh: nh[b] += 1
        if elems[b] == 'H' and a in nh: nh[a] += 1
    remap = {o: n for n, o in enumerate(keep)}
    bonds = [(remap[a], remap[b], o) for a, b, o in bonds if a in remap and b in remap]
    return dict(elements=[elems[i] for i in keep], charges=[charges[i] for i in keep],
                coords=np.array([coords[i] for i in keep]), bonds=bonds, n_h=[nh[i] for i in keep],
                all_elements=elems, all_coords=np.array(coords))


# ------------------------------------------------------------ reference conformers
# RDKit could not be installed in the sandbox (network), so reference conformers are generated
# with two structure-agnostic procedures:
#  * polymer residues: coordinates of a *different* instance of the same residue type
#    (randomly rotated) -> realistic covalent geometry, but not the residue's own pose;
#  * ligands: torsion-randomised distance-geometry embedding that only keeps local covalent
#    geometry (1-2/1-3 distances, chiral/planar volumes, rigid sp2-sp2 torsions).

def _rand_rot(rng):
    q = rng.normal(size=4); q /= np.linalg.norm(q)
    a, b, c, d = q
    return np.array([[a*a+b*b-c*c-d*d, 2*(b*c-a*d), 2*(b*d+a*c)],
                     [2*(b*c+a*d), a*a-b*b+c*c-d*d, 2*(c*d-a*b)],
                     [2*(b*d-a*c), 2*(c*d+a*b), a*a-b*b-c*c+d*d]])


def polymer_templates(residues, seed=0):
    """For each residue return {atom_name: ref_xyz} taken from the next other instance of the same
    residue type (cyclic). Residue types occurring once fall back to their own coordinates."""
    rng = np.random.default_rng(seed)
    by_type = {}
    for i, r in enumerate(residues):
        by_type.setdefault(r['resname'], []).append(i)
    out, src = [], []
    for i, r in enumerate(residues):
        lst = by_type[r['resname']]
        j = lst[(lst.index(i) + 1) % len(lst)]
        tmpl = {nm: xyz for nm, el, ch, xyz in residues[j]['atoms']}
        X = np.stack(list(tmpl.values())); X = (X - X.mean(0)) @ _rand_rot(rng).T
        out.append(dict(zip(tmpl.keys(), X))); src.append(j)
    return out, src


def _bond_graph(n, bonds):
    adj = [[] for _ in range(n)]
    for a, b, o in bonds:
        adj[a].append(b); adj[b].append(a)
    D = np.full((n, n), 99, int)
    for s in range(n):  # BFS topological distances
        D[s, s] = 0; fr = [s]
        while fr:
            nf = []
            for u in fr:
                for v in adj[u]:
                    if D[s, v] == 99:
                        D[s, v] = D[s, u] + 1; nf.append(v)
            fr = nf
    return adj, D


def ligand_reference_conformer(sdf, seed=0, iters=1500, cache_dir='/workspace/outputs/refconf_cache'):
    """Torsion-randomised conformer from the ligand graph; returns (pos (n,3), info dict)."""
    import torch, hashlib, os
    key = hashlib.md5((str(sdf['elements']) + str(sdf['bonds']) + str(np.round(sdf['coords'], 3).tolist())
                       + f'{seed}_{iters}').encode()).hexdigest()
    cp = f'{cache_dir}/{key}.npz'
    if os.path.exists(cp):
        d = np.load(cp)
        return d['pos'], dict(final_loss=float(d['loss']), n_restraints=int(d['nr']))
    with torch.enable_grad():
        pos, info = _ligand_reference_conformer(sdf, seed, iters)
    os.makedirs(cache_dir, exist_ok=True)
    np.savez(cp, pos=pos, loss=info['final_loss'], nr=info['n_restraints'])
    return pos, info


def _ligand_reference_conformer(sdf, seed, iters):
    import torch
    X0 = sdf['coords']; n = len(X0)
    elems = sdf['elements']
    adj, D = _bond_graph(n, sdf['bonds'])
    # hybridisation proxy from all-atom (incl. H) neighbour counts
    nall = len(sdf['all_elements'])
    heavy_idx = [i for i, e in enumerate(sdf['all_elements']) if e != 'H']
    deg_all = np.zeros(nall, int)
    # recompute total valence neighbours from the full SDF bond block
    dbl = set()
    for a, b, o in sdf['bonds']:
        if o == 2:
            dbl.add(a); dbl.add(b)
    n_h = np.array(sdf.get('n_h', [0] * n))
    tot = np.array([len(adj[i]) for i in range(n)]) + n_h
    sp2 = np.array([(e == 'C' and t == 3) or (e == 'O' and i in dbl) or i in dbl for i, (e, t) in enumerate(zip(elems, tot))])
    for i, e in enumerate(elems):  # amide / guanidinium N: 3 neighbours, bonded to sp2 C
        if e == 'N' and tot[i] == 3 and any(sp2[j] and elems[j] == 'C' for j in adj[i]):
            sp2[i] = True
    dist0 = np.linalg.norm(X0[:, None] - X0[None], axis=-1)
    keep = (D == 1) | (D == 2)
    rigid = np.zeros((n, n), bool)
    for a, b, o in sdf['bonds']:
        if sp2[a] and sp2[b]:
            for i in adj[a] + [a]:
                for j in adj[b] + [b]:
                    if i != j:
                        rigid[i, j] = rigid[j, i] = True
    keep |= rigid & (D == 3)
    iu = np.where(np.triu(keep, 1)); tgt = dist0[iu]
    far3 = np.where(np.triu(D == 3, 1) & ~rigid); far4 = np.where(np.triu(D >= 4, 1))
    centers = [(i, sorted(adj[i])[:3]) for i in range(n) if len(adj[i]) >= 3]
    def vol(X, c, nb):
        a, b, d = X[nb[0]] - X[c], X[nb[1]] - X[c], X[nb[2]] - X[c]
        return (a * torch.cross(b, d, dim=-1)).sum(-1)
    X0t = torch.tensor(X0, dtype=torch.float64)
    vt = torch.stack([vol(X0t, c, nb) for c, nb in centers]) if centers else None
    # chain-growth initialisation along a BFS tree: bond length from input, ~110 deg angles,
    # uniformly random dihedrals (=> flexible torsions randomised, no global pose information)
    rng0 = np.random.default_rng(seed + 1000)
    Xi = np.zeros((n, 3)); placed = np.zeros(n, bool); parent = -np.ones(n, int)
    placed[0] = True; order = [0]
    while order:
        u = order.pop(0)
        for v in adj[u]:
            if placed[v]:
                continue
            L = dist0[u, v]
            p = parent[u]
            if p < 0:
                dvec = rng0.normal(size=3)
            else:
                b = Xi[u] - Xi[p]; b /= np.linalg.norm(b)
                perp = np.cross(b, rng0.normal(size=3)); perp /= np.linalg.norm(perp)
                ang = np.deg2rad(180 - 110)
                dvec = np.cos(ang) * b + np.sin(ang) * perp
            Xi[v] = Xi[u] + L * dvec / np.linalg.norm(dvec)
            placed[v] = True; parent[v] = u; order.append(v)
    X = torch.tensor(Xi, dtype=torch.float64).requires_grad_()
    opt = torch.optim.Adam([X], lr=0.05)
    iu_t = [torch.tensor(v) for v in iu]; f3 = [torch.tensor(v) for v in far3]; f4 = [torch.tensor(v) for v in far4]
    tgt_t = torch.tensor(tgt)
    for it in range(iters):
        d = (X[iu_t[0]] - X[iu_t[1]]).norm(dim=-1)
        L = ((d - tgt_t) ** 2).sum()
        d3 = (X[f3[0]] - X[f3[1]]).norm(dim=-1); d4 = (X[f4[0]] - X[f4[1]]).norm(dim=-1)
        L = L + (torch.relu(2.6 - d3) ** 2).sum() + (torch.relu(3.2 - d4) ** 2).sum()
        if centers:
            v = torch.stack([vol(X, c, nb) for c, nb in centers])
            L = L + ((v - vt) ** 2).sum()
        opt.zero_grad(); L.backward(); opt.step()
        if it == 2 * iters // 3:
            for gp in opt.param_groups: gp['lr'] = 0.02
    pos = X.detach().numpy()
    rng = np.random.default_rng(seed)
    pos = (pos - pos.mean(0)) @ _rand_rot(rng).T
    return pos, dict(final_loss=float(L.detach()), n_restraints=len(tgt))


# --------------------------------------------------------------------- tokenisation
def build_complex(chains, ref_seed=0):
    """chains: list of dict(kind in MOLTYPES, and either residues=[parsed residues] (polymer) or
    sdf=parsed sdf (ligand)). Returns feature dict of numpy arrays + ground-truth coords."""
    tok, atm = [], []
    tok_bonds = []
    for ci, ch in enumerate(chains):
        kind = ch['kind']
        if kind in ('protein', 'rna', 'dna'):
            if 'templates' in ch:
                tmpls = ch['templates']
            else:
                tmpls, _ = polymer_templates(ch['residues'], seed=ref_seed)
            for ri, res in enumerate(ch['residues']):
                ref = tmpls[ri]
                t = len(tok)
                tok.append(dict(restype=RES2IDX.get(res['resname'], RES2IDX['UNK']), mol=kind,
                                resid=ri, asym=ci, entity=ci, sym=0))
                for (nm, el, chg, xyz) in res['atoms']:
                    if nm == 'OXT':
                        continue
                    atm.append(dict(token=t, name=nm, elem=el, charge=chg, xyz=xyz,
                                    ref=ref.get(nm, np.zeros(3)), ref_mask=float(nm in ref),
                                    uid=t, mol=kind))
        else:  # ligand: one token per heavy atom
            sdf = ch['sdf']
            refpos, _ = ligand_reference_conformer(sdf, seed=ref_seed)
            t0 = len(tok)
            uid = 10000 + ci
            for k, (el, chg, xyz) in enumerate(zip(sdf['elements'], sdf['charges'], sdf['coords'])):
                tok.append(dict(restype=RES2IDX['UNK'], mol='ligand', resid=0, asym=ci, entity=ci, sym=0))
                atm.append(dict(token=t0 + k, name=f'{el}{k}', elem=el, charge=chg, xyz=xyz,
                                ref=refpos[k], ref_mask=1.0, uid=uid, mol='ligand'))
            for a, b, o in sdf['bonds']:
                tok_bonds.append((t0 + a, t0 + b))
            ch['_lig_bonds_atoms'] = [(a, b) for a, b, o in sdf['bonds']]
    T, N = len(tok), len(atm)
    f = {}
    f['restype'] = np.array([t['restype'] for t in tok])
    f['moltype'] = np.array([MOLTYPES.index(t['mol']) for t in tok])
    f['residue_index'] = np.array([t['resid'] for t in tok])
    f['token_index'] = np.arange(T)
    f['asym_id'] = np.array([t['asym'] for t in tok])
    f['entity_id'] = np.array([t['entity'] for t in tok])
    f['sym_id'] = np.array([t['sym'] for t in tok])
    tb = np.zeros((T, T), np.float32)
    for a, b in tok_bonds:
        tb[a, b] = tb[b, a] = 1
    f['token_bonds'] = tb
    f['atom_to_token'] = np.array([a['token'] for a in atm])
    f['ref_pos'] = np.stack([a['ref'] for a in atm]).astype(np.float32)
    f['ref_mask'] = np.array([a['ref_mask'] for a in atm], np.float32)
    f['ref_element'] = np.array([ELEM2IDX.get(a['elem'], ELEM2IDX['X']) for a in atm])
    f['ref_charge'] = np.array([a['charge'] for a in atm], np.float32)
    chars = np.zeros((N, 4), np.int64)
    for i, a in enumerate(atm):
        for j, c in enumerate(a['name'][:4]):
            chars[i, j] = min(ord(c) - 32, 63)
    f['ref_atom_name_chars'] = chars
    f['ref_space_uid'] = np.array([a['uid'] for a in atm])
    f['atom_moltype'] = np.array([MOLTYPES.index(a['mol']) for a in atm])
    f['atom_weight'] = np.array([MOL_WEIGHT[a['mol']] for a in atm], np.float32)
    f['coords'] = np.stack([a['xyz'] for a in atm]).astype(np.float32)
    f['atom_names'] = np.array([a['name'] for a in atm])
    # centre atom per token (CA / C1' / ligand atom) and distogram atom (CB / CA for Gly)
    center = np.zeros(T, np.int64)
    rep = np.zeros(T, np.int64)
    for i, a in enumerate(atm):
        t = a['token']
        m = tok[t]['mol']
        if m == 'ligand' or (m == 'protein' and a['name'] == 'CA') or (m in ('rna', 'dna') and a['name'] == "C1'"):
            center[t] = i
        if m == 'ligand' or (m == 'protein' and (a['name'] == 'CB' or (a['name'] == 'CA' and tok[t]['restype'] == RES2IDX['GLY']))) \
                or (m in ('rna', 'dna') and a['name'] == "C4"):
            rep[t] = i
    f['token_center_atom'] = center
    f['token_rep_atom'] = rep
    # ligand bonds at atom level (for bond loss / bond-length metric)
    lb = []
    lig_atoms = np.where(f['atom_moltype'] == 3)[0]
    offset = lig_atoms.min() if len(lig_atoms) else 0
    for ch in chains:
        if ch['kind'] == 'ligand':
            lb += [(offset + a, offset + b) for a, b in ch['_lig_bonds_atoms']]
    f['lig_bonds'] = np.array(lb, np.int64).reshape(-1, 2)
    # all intra-residue polymer covalent bonds from reference conformer distance (<1.9A) + peptide bonds
    return f


def load_2l3r(root='/workspace/data/sample/2l3r', ref_seed=0):
    prot = parse_pdb(f'{root}/2l3r_protein.pdb')
    lig = parse_sdf(f'{root}/2l3r_ligand.sdf')
    return build_complex([dict(kind='protein', residues=prot), dict(kind='ligand', sdf=lig)], ref_seed=ref_seed)


