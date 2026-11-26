"""Step 11: near-room-temperature (T* = 300 K) design case study: model candidate from latent BO, then
explicit chemical-intuition edits toward commercially accessible monomers, each re-scored with the same
virtual-MD + GP-calibration oracle and the latent GP. Also draws the candidate grid."""
import sys, numpy as np, pandas as pd, pickle
sys.path.insert(0, 'code')
from latent_lib import *
from oracle_lib import Oracle
from rdkit.Chem import Draw

model, D = load('b01')
meta = pickle.load(open('outputs/chemspace_meta.pkl', 'rb'))
gp = TorchGP(meta['gp_X'], meta['gp_y'], iters=0, init=meta['gp_p'])
orc = Oracle()
B = pd.read_csv('outputs/bo_designs.csv')
lib_a, lib_e = set(D['acid']), set(D['epox'])
can = lambda s: Chem.MolToSmiles(Chem.MolFromSmiles(s))
RDGE = 'c1cc(OCC2CO2)cc(OCC2CO2)c1'                       # resorcinol diglycidyl ether (commercial)
DGEBA = 'CC(C)(c1ccc(OCC2CO2)cc1)c1ccc(OCC2CO2)cc1'       # bisphenol-A diglycidyl ether (commercial)
cases = [
    ('M0 model candidate (BO, T*=300 K)', 'CN(CCCCCOCC(=O)O)CCOCC(=O)O', 'CC(C)NC(=O)COC(=O)c1cc(OCC2CO2)cc(OCC2CO2)c1',
     'BO design; tertiary amine in diacid = internal transesterification catalyst'),
    ('E1 simplify epoxide -> RDGE', 'CN(CCCCCOCC(=O)O)CCOCC(=O)O', RDGE,
     'drop isopropyl-amide ester arm; one-step glycidylation of resorcinol (commercial)'),
    ('E2 commercial amine diacid + RDGE', 'CN(CCC(=O)O)CCC(=O)O', RDGE,
     '3,3\'-(methylimino)dipropionic acid keeps the catalytic tertiary amine, shorter spacer'),
    ('E3 PEG-type diacid + RDGE', 'O=C(O)COCCOCCOCC(=O)O', RDGE,
     '3,6,9-trioxaundecanedioic acid (commercial), flexible ether spacer'),
    ('E4 sebacic acid + RDGE', 'O=C(O)CCCCCCCCC(=O)O', RDGE, 'bio-based C10 diacid'),
    ('R1 reference: sebacic acid + DGEBA', 'O=C(O)CCCCCCCCC(=O)O', DGEBA, 'classic Leibler-type epoxy/acid vitrimer chemistry'),
    ('R2 reference: adipic acid + DGEBA', 'O=C(O)CCCCC(=O)O', DGEBA, 'shorter diacid, stiffer network'),
]
rows = []
for name, a, e, why in cases:
    a, e = can(a), can(e)
    r = orc.score_pairs([a], [e])
    z = np.c_[encode(model, [a], 'acid')[0], encode(model, [e], 'epox')[0]]
    mu, sd = gp.predict(z, noise=True)
    rows.append(dict(case=name, acid=a, epoxide=e, rationale=why, fg_ok=fg_check(a, 'acid') and fg_check(e, 'epox'),
                     oracle_md=r['md'][0], oracle_cal=r['cal'][0], oracle_sigma=r['sigma_cal'][0], latent_gp=mu[0], latent_sd=sd[0],
                     head_md=head(model, z)[0], maxsim_lib=r['maxsim_lib'][0], acid_in_lib=a in lib_a, epox_in_lib=e in lib_e,
                     repeat_unit=r['ru'][0]))
R = pd.DataFrame(rows); R.to_csv('outputs/rt_design_edits.csv', index=False)
print(R[['case', 'oracle_md', 'oracle_cal', 'oracle_sigma', 'latent_gp', 'head_md', 'maxsim_lib', 'acid_in_lib', 'epox_in_lib']].round(1).to_string())

# ---- candidate grid: best per objective + RT case study
sel = []
for o, how in [('max', 'max'), ('min', 'min'), ('target373', 373), ('target300', 300)]:
    q = B[B.objective == o]
    q = q[(q.reenc_mu - q.oracle_cal).abs() < 25]          # re-encode consistency filter
    if how == 'max': q = q.sort_values('oracle_cal', ascending=False)
    elif how == 'min': q = q.sort_values('oracle_cal')
    else: q = q.assign(err=(q.oracle_cal - how).abs()).sort_values('err')
    for _, r in q.head(2).iterrows():
        sel.append((f'{o}: Tg {r.oracle_cal:.0f} +/- {r.oracle_sigma:.0f} K', r.acid, r.epoxide))
for _, r in R.iterrows():
    if r.case[:2] in ('M0', 'E1', 'E2', 'E3'):
        sel.append((f'{r.case[:2]}: Tg {r.oracle_cal:.0f} +/- {r.oracle_sigma:.0f} K', r.acid, r.epoxide))
mols, legs = [], []
for lab, a, e in sel:
    mols += [Chem.MolFromSmiles(a), Chem.MolFromSmiles(e)]; legs += [lab + ' (acid)', '(epoxide)']
img = Draw.MolsToGridImage(mols, molsPerRow=4, subImgSize=(340, 250), legends=legs)
img.save('report/images/fig11_candidates.png')
pd.DataFrame(sel, columns=['label', 'acid', 'epoxide']).to_csv('outputs/candidate_grid.csv', index=False)
