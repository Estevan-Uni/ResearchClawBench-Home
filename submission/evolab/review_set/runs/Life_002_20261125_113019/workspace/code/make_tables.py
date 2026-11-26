"""Emit markdown tables used in the report (all numbers from outputs/*.json / .m8)."""
import json, numpy as np, pandas as pd, sys
sys.path.insert(0,'code'); from parse_foldseek import load_m8
CT=json.load(open('outputs/chain_tables.json'))['tables']
US={t:json.load(open(f'outputs/usalign/7XG4_vs_{t}.json')) for t in ['6N40','9N3Y','8RFJ','8S35','7XG4']}
L=[]
f3=lambda v:', '.join(f'{x:.3f}' for x in v)
def mat(U): return '['+'; '.join(f3(r) for r in U)+']'
# 1 requested pair
rows=load_m8('outputs/pair/7XG4_vs_6N40.m8')
L.append('### T1. Requested pair 7XG4 (query, 9 chains) vs 6N40 (target, 1 chain): Foldseek-Multimer records\n')
L.append('| assignment | query chain | target chain | aln len | fident | chain TM (q) | chain TM (t) | E-value | complex qTM | complex tTM | complex U (query→target, rows) | complex t (Å) |')
L.append('|---|---|---|---|---|---|---|---|---|---|---|---|')
for r in sorted(rows,key=lambda r:-r['complexqtmscore']):
    U=r['cU'].T; t=-U@r['cT']
    L.append(f"| {r['complexassignid']} | {r['qch']} | A | {r['alnlen']} | {r['fident']:.3f} | {r['qtmscore']:.3f} | {r['ttmscore']:.3f} | {r['evalue']:.2g} | {r['complexqtmscore']:.4f} | {r['complexttmscore']:.4f} | {mat(U)} | {f3(t)} |")
u=US['6N40']
L.append(f"\nUS-align -mm 1 -ter 0 on the same pair: chain map {', '.join(a+'→'+b for a,b in u['pairs'])}; aligned length {u['alnlen']}, RMSD {u['rmsd']:.2f} Å, seq. id. {u['seqid']:.3f}; qTM = {u['qtm']:.4f}, tTM = {u['ttm']:.4f}; U = {mat(u['u'])}, t = ({f3(u['t'])}) Å; wall time {u['time']:.1f} s.\n")
# 2 per-target chain tables
for tc in ['9N3Y','8RFJ','8S35']:
    d=CT[tc]; u=US[tc]; up=dict(u['pairs'])
    L.append(f"### Chain-correspondence table: 7XG4 → {tc} ({d['tier']})\n")
    L.append(f"Foldseek-Multimer complex qTM = {d['complexqTM']:.4f}, tTM = {d['complexTM_t']:.4f}; complex U (query→target) = {mat(d['complexU_q2t'])}, t = ({f3(d['complext_q2t'])}) Å. "
             f"US-align: qTM = {u['qtm']:.4f}, tTM = {u['ttm']:.4f}, U = {mat(u['u'])}, t = ({f3(u['t'])}) Å.\n")
    L.append('| q chain | t chain (FS-M) | t chain (US-align) | q len | aln len | fident | chain TM (q) | chain TM (t) | per-pair U (query→target, rows) | per-pair t (Å) | Δangle to complex U (°) | RMS disp. vs complex (Å) |')
    L.append('|---|---|---|---|---|---|---|---|---|---|---|---|')
    for p in d['pairs']:
        L.append(f"| {p['qchain']} | {p['tchain']} | {up.get(p['qchain'],'–')} | {p['qlen']} | {p['alnlen']} | {p['fident']:.3f} | {p['chainTM_q']:.3f} | {p['chainTM_t']:.3f} | {mat(p['U_q2t'])} | {f3(p['t_q2t'])} | {p['dAngle_deg']:.2f} | {p['dispRMS']:.2f} |")
    extra=[q for q in up if q not in {p['qchain'] for p in d['pairs']}]
    for q in extra: L.append(f"| {q} | – (not aligned) | {up[q]} | | | | | | | | | |")
    L.append(f"\nPaired-chain sequence identity: mean {d['fident_mean']:.3f}, min {d['fident_min']:.3f}, max {d['fident_max']:.3f} over {d['nchain_pairs']} chain pairs.\n")
open('outputs/tables.md','w').write('\n'.join(L)); print('\n'.join(L))
