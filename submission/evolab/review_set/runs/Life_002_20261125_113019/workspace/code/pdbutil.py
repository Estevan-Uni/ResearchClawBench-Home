"""Minimal PDB / mmCIF parser: protein-only, model 1, writes clean PDB."""
import numpy as np, gzip, os
AA3 = {'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C','GLN':'Q','GLU':'E','GLY':'G','HIS':'H','ILE':'I',
       'LEU':'L','LYS':'K','MET':'M','PHE':'F','PRO':'P','SER':'S','THR':'T','TRP':'W','TYR':'Y','VAL':'V','MSE':'M','SEC':'U','UNK':'X'}

def _open(p):
    return gzip.open(p,'rt') if p.endswith('.gz') else open(p)

def read_pdb_atoms(path):
    atoms=[]; model=0
    for l in _open(path):
        if l.startswith('MODEL'): model+=1
        if l.startswith('ENDMDL') and model>=1: break
        if l.startswith(('ATOM','HETATM')):
            res=l[17:20].strip()
            if res not in AA3: continue
            if l[16] not in ' A': continue
            atoms.append(dict(name=l[12:16].strip(), res=res, chain=l[21], resi=l[22:27], x=float(l[30:38]), y=float(l[38:46]), z=float(l[46:54]), elem=l[76:78].strip() or l[12:14].strip()[0]))
    return atoms

def read_cif_atoms(path):
    cols=[]; atoms=[]; inloop=False; first_model=None
    for l in _open(path):
        if l.startswith('_atom_site.'):
            cols.append(l.strip().split('.',1)[1]); inloop=True; continue
        if inloop and cols and l.startswith(('ATOM','HETATM')):
            f=l.split(); d=dict(zip(cols,f))
            m=d.get('pdbx_PDB_model_num','1')
            if first_model is None: first_model=m
            if m!=first_model: continue
            res=d['label_comp_id']
            if res not in AA3: continue
            if d.get('label_alt_id','.') not in '.A': continue
            ch=d.get('auth_asym_id',d['label_asym_id'])
            atoms.append(dict(name=d['label_atom_id'].strip('"'), res=res, chain=ch, resi=d.get('auth_seq_id','0')+(d.get('pdbx_PDB_ins_code','?') if d.get('pdbx_PDB_ins_code','?') not in '?.' else ''),
                              x=float(d['Cartn_x']), y=float(d['Cartn_y']), z=float(d['Cartn_z']), elem=d.get('type_symbol','C')))
        elif inloop and cols and not l.startswith(('ATOM','HETATM')) and atoms:
            break
    return atoms

def read_atoms(path):
    return read_cif_atoms(path) if '.cif' in path else read_pdb_atoms(path)

def write_clean_pdb(atoms, out, min_res=20):
    """protein-only; single-char chain IDs required for PDB. Drop chains with < min_res CA."""
    from collections import OrderedDict
    chains=OrderedDict()
    for a in atoms: chains.setdefault(a['chain'],[]).append(a)
    keep=[c for c,v in chains.items() if sum(1 for a in v if a['name']=='CA')>=min_res]
    ids='ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789'
    mapping={}
    used=set(c for c in keep if len(c)==1)
    pool=[i for i in ids if i not in used]
    for c in keep:
        mapping[c]= c if len(c)==1 else pool.pop(0)
    n=1
    with open(out,'w') as f:
        for c in keep:
            for a in chains[c]:
                nm=a['name']; nm_f=(' '+nm) if len(nm)<4 else nm
                resi=a['resi'].strip(); ins=''
                if resi and not resi[-1].isdigit(): ins=resi[-1]; resi=resi[:-1]
                f.write('ATOM  %5d %-4s %3s %1s%4d%1s   %8.3f%8.3f%8.3f  1.00  0.00          %2s\n'%(n%100000,nm_f,a['res'],mapping[c],int(resi),ins or ' ',a['x'],a['y'],a['z'],a['elem']))
                n+=1
            f.write('TER\n')
        f.write('END\n')
    return mapping

def ca_chains(path):
    """returns dict chain -> (seq, coords Nx3) from clean pdb"""
    out={}
    for a in read_pdb_atoms(path):
        if a['name']!='CA': continue
        s,c=out.setdefault(a['chain'],([],[]))
        s.append(AA3[a['res']]); c.append((a['x'],a['y'],a['z']))
    return {k:(''.join(v[0]),np.array(v[1])) for k,v in out.items()}
