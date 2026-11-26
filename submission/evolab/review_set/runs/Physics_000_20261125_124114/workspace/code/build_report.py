"""Generate all report tables from ../outputs and fill report_template.md -> ../report/report.md"""
import json, os, re, numpy as np
import icoshell as ic

O = '../outputs/'
L = lambda f: json.load(open(O + f))
T = {}


def md(rows, head):
    s = '| ' + ' | '.join(head) + ' |\n|' + '---|' * len(head) + '\n'
    for r in rows:
        s += '| ' + ' | '.join(str(x) for x in r) + ' |\n'
    return s


f3 = lambda x: '—' if x is None else ('%.3f' % x)

# lattice table
lt = L('lattice_table_audit.json')
rows = [['(%d,%d)' % tuple(r['hk']), r['T'], r['N'], r['cls'], 'yes' if r['chiral'] else 'no'] for r in lt['table']]
half = (len(rows) + 1) // 2
rows2 = [a + b for a, b in zip(rows[:half], rows[half:] + [[''] * 5])]
T['TABLE_LATTICE'] = md(rows2, ['(h,k)', 'T', 'N', 'class', 'chiral'] * 2)
T['SEQ_CHECK'] = 'Mackay path cumulative sizes %s (dataset `mackay_sequence` match: %s); zig-zag path %s (dataset `new_sequence_b5` match: %s).' % (
    lt['sequences']['mackay'], lt['dataset_sequences_match']['mackay'], lt['sequences']['zigzag'], lt['dataset_sequences_match']['b5'])
rows = []
for a in lt['audit']:
    if 'label' in a:
        rows.append([a['item'], 'N=%d %s' % (a['N'], a['hk'] or ''), a['label'], a['A'], a['B'], a['C']])
    else:
        fmt = lambda c: '; '.join('%s→%s %.3f' % (tuple(x[0]), tuple(x[1]), x[2]) for x in c[:3]) or 'no NN step'
        rows.append([a['item'], 'range %.2f–%.2f' % tuple(a['range']), '—',
                     ('✓ ' if a['ok_A'] else '✗ ') + fmt(a['A']), ('✓ ' if a['ok_B'] else '✗ ') + fmt(a['B']),
                     ('✓ ' if a['ok_C'] else '✗ ') + fmt(a['C'])])
T['TABLE_AUDIT'] = md(rows, ['dataset item', 'object', 'dataset label', 'conv. A (adopted)', 'conv. B', 'conv. C'])
T['AUDIT_SCORE'] = 'A: %d/%d, B: %d/%d, C: %d/%d' % (lt['score']['A'], lt['n_items'], lt['score']['B'], lt['n_items'],
                                                     lt['score']['C'], lt['n_items'])
# geometry checks
g = L('geometry_checks.json')
T['LJ_CHECK'] = 'LJ$_{13}$ = %.4f, LJ$_{55}$ = %.4f, LJ$_{147}$ = %.4f ε (literature −44.3268, −279.2485, −876.4612); max central-difference gradient relative error %.1e' % (
    g['LJ13']['E'], g['LJ55']['E'], g['LJ147']['E'], g['grad_rel_err_max'])
# mismatch table
mt = L('mismatch_table.json')
rows = []
for r in mt['table']:
    rows.append(['(%d,%d)→(%d,%d)' % (*r['frm'], *r['to']), r['step'], '%.3f' % r['cos'], f3(r['formula']), f3(r['limit']),
                 f3(r['LJ']), f3(r['Morse10']), f3(r['harm'])])
T['TABLE_DSTAR'] = md(rows, ['step', 'type', 'cos θ', 'δ* formula', 'large-T limit', 'δ* LJ', 'δ* Morse ρ=10', 'δ* harmonic'])
T['MAE'] = ', '.join('%s %.4f (n=%d)' % (k, v['MAE'], v['n']) for k, v in mt['mae'].items())
rows = []
jo = L('joint_seeds.json')
for n, v in jo.items():
    p = [tuple(x) for x in v['path']]
    rows.append([' → '.join('(%d,%d)' % q for q in p), ', '.join('%.3f' % d for d in v['deltas']),
                 ', '.join('%.3f' % d for d in v['deltas_vertex_contact']), '%.3f' % v['E']])
T['TABLE_JOINT'] = md(rows, ['path', 'δ (joint NM optimum)', 'δ (vertex-contact formula)', 'E_min [ε]'])
# stability windows
w = L('stability_windows.json')
SC = L('scans_LJ.json')
rows = []
for core, seg in w.items():
    p = SC[core]['core_path']
    rows.append(['%s (N=%d)' % (tuple(p[-1]), sum(ic.N_of(*q) for q in p)),
                 '; '.join('%s %s: %.2f–%.2f' % (s[0], ic.shell_class(*eval(s[0])), s[1], s[2]) for s in seg)])
T['TABLE_WINDOWS'] = md(rows, ['core', 'lowest-Δ outer shell in δ window (LJ grid 0.02)'])
# design
ds = L('design.json')
rows = [[p['core'], p['shell'], '%.3f' % p['delta'], p['name'], '(%d,%d)' % tuple(p['shell_hk']), p['cls'], '%.3f' % p['dstar'],
         ('%s %s' % (p['window_name'], ic.shell_class(*p['window']))) if p.get('window') else '— (δ beyond scan)']
        for p in ds['pairs'] if p['delta'] < 0.45]
T['TABLE_PAIRS'] = md(rows, ['core', 'shell', 'δ (radii)', 'nearest-δ* rule', 'shell (h,k)', 'class', 'δ* of chosen step', 'LJ stability-window rule'])
rows = [[' → '.join(s['seq']), s['name'], ' → '.join('(%d,%d)' % tuple(q) for q in s['path'][1:]), ' → '.join(s['classes'][1:]),
         ', '.join('%.3f/%.3f' % (r['delta'], r['dstar']) for r in s['rows'])] for s in ds['sequences']]
T['TABLE_SEQS'] = md(rows, ['species sequence', 'predicted cluster', 'path', 'classes', 'δ actual / δ* per step'])
au = ds['audit']['pairs']
T['PAIR_AUDIT'] = '; '.join('%s–%s listed %.3f vs radii %.3f' % (a['a'], a['b'], a['listed'], a['from_radii']) for a in au) + \
    '; (K/Na from radii = %.3f)' % au[0]['NaK']
# gupta
G = L('gupta_results.json')
rows = []
for c in G['cases']:
    rows.append([c['mode'], '%s%d@%s%d' % (c['core'], c['Ncore'], c['shell'], c['Ns']), '(%d,%d)' % tuple(c['outer']),
                 ic.shell_class(*c['outer']), c['step'], '%.3f' % c['Delta'], '%.3f' % c['eps_shell'], '%.1f' % c['shift']])
T['TABLE_GUPTA'] = md(rows, ['cross set', 'cluster', 'outer', 'class', 'step', 'Δ (eV)', 'ε_shell (eV/atom)', 'max shift (°)'])
ag = L('gupta_vs_rule.json')
T['GUPTA_AGREE_W'] = ', '.join('%s: Δ %d/%d, ε_shell %d/%d' % (m, sum(a['okw_Delta'] for a in ag if a['mode'] == m),
                                                             sum(1 for a in ag if a['mode'] == m),
                                                             sum(a['okw_eps'] for a in ag if a['mode'] == m),
                                                             sum(1 for a in ag if a['mode'] == m)) for m in ['avg', 'lit'])
rows = [['%s%d@%s' % (a['core'], a['Ncore'], a['shell']), '%.3f' % a['delta'], a['mode'], str(a['rule']), str(a['window']),
         str(a['best_Delta']), str(a['best_eps'])] for a in ag]
T['TABLE_RULES'] = md(rows, ['system', 'δ (radii)', 'cross set', 'nearest-δ* rule', 'LJ window rule', 'Gupta best Δ', 'Gupta best ε_shell'])
T['GUPTA_AGREE'] = ', '.join('%s: Δ %d/%d, ε_shell %d/%d' % (m, sum(a['ok_Delta'] for a in ag if a['mode'] == m),
                                                             sum(1 for a in ag if a['mode'] == m),
                                                             sum(a['ok_eps'] for a in ag if a['mode'] == m),
                                                             sum(1 for a in ag if a['mode'] == m)) for m in ['avg', 'lit'])
T['GUPTA_PURE'] = ', '.join('%s %.3f' % (k, v) for k, v in G['pure'].items())
# lattice model
lmj = L('lattice_model.json')
T['KEL'] = 'K = %.1f ε per shell atom (median of %d quadratic fits), λ = βK/2 = %.1f at T* = 0.22' % (
    lmj['K_median'], len(lmj['K']), lmj['lam_thermo'])
rows = []
for k, v in lmj['runs'].items():
    st = v['stats']; tot = sum(st.values())
    tp = v['top_paths'][0]
    rows.append([k, *['%.2f' % (st[t] / tot) for t in ['conservative', 'mismatch-driven', 'reverse', 'random']],
                 ' '.join('(%d,%d)' % tuple(q) for q in tp[0]) + ' [%d/500]' % tp[1]])
T['TABLE_E4'] = md(rows, ['run', 'conservative', 'mismatch-driven', 'reverse', 'random', 'most frequent path'])
# growth
gs = L('growth_summary.json')
st = L('growth_stats.json')
rows = []
for n in sorted(gs):
    s = gs[n]
    ss = st.get(n, {})
    fam = n.split('_')[0]
    for stage in ['id_grown', 'id_anneal']:
        best = ss.get(stage + '_best', [])
        from collections import Counter
        cnt = Counter(best)
        comp = ss.get(stage)
        comp_s = '%d/%d (%.2f, CI %.2f–%.2f)' % (comp['k'], comp['n'], *comp['ci']) if comp else '—'
        rows.append([n, stage.replace('id_', ''), s['n'], ', '.join('%s×%d' % (k, v) for k, v in cnt.most_common()),
                     comp_s, ', '.join('%s×%d' % (k, v) for k, v in ss.get(stage + '_classes', {}).items())])
T['TABLE_GROWTH'] = md(rows, ['experiment', 'stage', 'runs', 'best-matching shell (count)', 'target completed (Wilson 95% CI)', 'class'])
rows = []
for n in sorted(gs):
    for r in gs[n]['runs']:
        a = r['id_anneal']
        rows.append([n, r['seed'], (str(tuple(a['best'])) if a['resolved'] else 'unresolved (best %s)' % str(tuple(a['best']))), '%.2f' % a['excess'],
                     ', '.join('%s %.2f/%.2f' % (k, v, a['prec'][k]) for k, v in a['cov'].items()),
                     ', '.join('%s %+.2f' % (k, v) for k, v in a['chi'].items()) or '—',
                     '%.1f' % (r['E_anneal'] - r['ideal'][list(r['ideal'])[0]]['E']) if n.startswith('E1') else '—'])
T['TABLE_RUNS'] = md(rows, ['experiment', 'seed', 'best shell (F1≥0.6)', 'F1 of best', 'recall/precision per template (annealed)', 'chirality index', 'E−E(ideal 55) [ε]'])
mv = L('md_vs_model_steps.json')
rows = [[c['name'], '(%d,%d)' % tuple(c['seed']), c['delta'], c['n'],
         ', '.join('%s %d' % (k, v) for k, v in c['counts'].items()),
         ', '.join('%s %.2f' % (k[:4], v) for k, v in c['model_thermo'].items()),
         ', '.join('%s %.2f' % (k[:4], v) for k, v in c['model_soft'].items()),
         '%s (%s)' % (tuple(c['window_shell']), max(c['model_window'], key=c['model_window'].get))] for c in mv]
T['TABLE_MDMODEL'] = md(rows, ['MD experiment', 'seed shell', 'δ', 'runs', 'MD outcomes', 'elastic model (λ, p_r=0.1)', 'elastic model (λ/30)', 'T=0 stability-map winner'])

tpl = open('report_template.md').read()
tpl = tpl.replace('GROWTH_SECTION', open('growth_text.md').read())
tpl = tpl.replace('DISCUSSION_SECTION', open('discussion_text.md').read())
for key, fn in [('E2_DELTA0_TEXT', 'txt_e2_delta0.md'), ('E2B_TEXT', 'txt_e2b.md'), ('E3C_TEXT', 'txt_e3c.md'),
                ('DISC_DELTA0', 'txt_disc_delta0.md'), ('E2B_DELTA0', 'txt_e2b_d0.md'), ('E2B_S1_RESULT', 'txt_e2b_s1.md'), ('SLOW_E2_TEXT', 'txt_slow_e2.md'), ('GUPTA_DISCUSSION', 'txt_gupta.md')]:
    if os.path.exists(fn):
        tpl = tpl.replace(key, open(fn).read().strip())
for k, v in T.items():
    tpl = tpl.replace('{{%s}}' % k, v)
missing = re.findall(r'\{\{(\w+)\}\}', tpl)
if missing:
    print('MISSING', missing)
open('../report/report.md', 'w').write(tpl)
json.dump({k: v for k, v in T.items()}, open(O + 'report_tables.json', 'w'), indent=1)
print('report written', len(tpl))
