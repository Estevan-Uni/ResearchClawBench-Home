"""Smoke test of the unified framework on a synthetic multi-entity input:
protein (UHRF1 TTD sequence) + dsDNA (2 x 6 nt) + RNA (5 nt) + the H3K9me3 ligand graph.
Nucleotide reference conformers are unavailable offline (no RDKit/CCD), so nucleotide atoms carry
ref_mask=0 - the network must still tokenise, embed and denoise them.  The untrained network is used;
the point is to verify shapes, tokenisation rules and the sampler, not accuracy."""
import sys, os, json, time
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(__file__))
from biomol import parse_pdb, parse_sdf, build_complex, MOLTYPES
from model import AF3Lite
from common import to_torch, sample

NUC_BACKBONE = ["P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "C1'"]
BASES = {'A': ['N9', 'C8', 'N7', 'C5', 'C6', 'N6', 'N1', 'C2', 'N3', 'C4'],
         'G': ['N9', 'C8', 'N7', 'C5', 'C6', 'O6', 'N1', 'C2', 'N2', 'N3', 'C4'],
         'C': ['N1', 'C2', 'O2', 'N3', 'C4', 'N4', 'C5', 'C6'],
         'T': ['N1', 'C2', 'O2', 'N3', 'C4', 'O4', 'C5', 'C7', 'C6'],
         'U': ['N1', 'C2', 'O2', 'N3', 'C4', 'O4', 'C5', 'C6']}


def nuc_chain(seq, kind):
    res = []
    for k, b in enumerate(seq):
        names = NUC_BACKBONE + (["O2'"] if kind == 'rna' else []) + BASES[b]
        rn = b if kind == 'rna' else 'D' + b
        res.append(dict(chain='N', resseq=k, resname=rn, atoms=[(n, n[0], 0, np.zeros(3)) for n in names]))
    tm = [{} for _ in res]  # no reference conformer -> ref_mask = 0
    return dict(kind=kind, residues=res, templates=tm)


prot = parse_pdb('/workspace/data/sample/2l3r/2l3r_protein.pdb')
lig = parse_sdf('/workspace/data/sample/2l3r/2l3r_ligand.sdf')
chains = [dict(kind='protein', residues=prot), nuc_chain('GATTAC', 'dna'), nuc_chain('GTAATC', 'dna'),
          nuc_chain('GCAUC', 'rna'), dict(kind='ligand', sdf=lig)]
f = build_complex(chains)
summary = dict(n_tokens=int(len(f['restype'])), n_atoms=int(len(f['atom_to_token'])),
               tokens_per_moltype={m: int((f['moltype'] == i).sum()) for i, m in enumerate(MOLTYPES)},
               atoms_per_moltype={m: int((f['atom_moltype'] == i).sum()) for i, m in enumerate(MOLTYPES)},
               n_chains=int(f['asym_id'].max() + 1), n_token_bonds=int(f['token_bonds'].sum() / 2))
ft = to_torch(f)
model = AF3Lite(n_pairformer=2).eval()
t0 = time.time()
with torch.no_grad():
    tr = model.trunk(ft)
    x = sample(model, ft, tr, n_samples=2, n_steps=5)
summary.update(trunk_single_shape=list(tr['s'].shape), trunk_pair_shape=list(tr['z'].shape),
               distogram_shape=list(tr['disto'].shape), sample_shape=list(x.shape),
               finite=bool(torch.isfinite(x).all()), seconds=time.time() - t0)
print(json.dumps(summary, indent=1))
json.dump(summary, open('/workspace/outputs/demo_multientity.json', 'w'), indent=1)
