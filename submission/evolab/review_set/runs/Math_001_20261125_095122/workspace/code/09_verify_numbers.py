"""Re-derive the numbers quoted in report.md from saved outputs."""
import json, pickle, numpy as np
from common import *

O = '/workspace/outputs'
R = json.load(open(f'{O}/reference.json')); D = pickle.load(open(f'{O}/lasso_hist.pkl', 'rb')); H = D['H']
TB = json.load(open(f'{O}/lasso_tables.json')); TH = json.load(open(f'{O}/theory.json'))
AD = json.load(open(f'{O}/admm_linear.json')); KJ = json.load(open(f'{O}/kappa.json'))
lines = []
def p(*a):
    s = ' '.join(str(x) for x in a); print(s); lines.append(s)

f, r = R['facts'], R['ref']
p('facts L', f['L'], 'lam_max', f['lam_max'], 'F0', f['F0'], 'resid', f['resid_true'], 'null', f['null_dim'])
p('ref Fs', r['Fs'], 'kkt_on', r['kkt_on'], 'kkt_off', r['kkt_off'], 'slack', r['min_offsupport_slack'], 'dgap', r['dgap'],
  'fpr', r['fpr'], 'muS', r['mu_S'], 'LS', r['L_S'], 'sqrt', r['sqrt_muS_LS'], 'TP/FP/FN', r['TP'], r['FP'], r['FN'],
  'FPmax', r['FP_maxabs'], 'FNval', r['FN_true_vals'], 'relerr', r['relerr'])
p('scan relerr', {k: round(v['relerr'] * 100, 2) for k, v in R['scan'].items()}, 'FP', {k: v['FP'] for k, v in R['scan'].items()})
p('dF0', f['F0'] - r['Fs'], 'cert rel', r['dgap'] / (f['F0'] - r['Fs']))
for m in ['PG', 'FISTA', 'FISTA+grad-restart', 'VOS-AG-II', 'VOS-AG-II+restart', 'VOS-AG-II monotone', 'ADMM']:
    p('phase', m, TB['phases'][m])
p('pred', AD['pred'])
p('mv to 1e-10', {m: v['mv_rel1e10'] for m, v in TB['table2'].items()})
h = H['VOS-AG-II+restart']; p('VOS restart first k gap<=1e-14:', first_hit(h['k'], h['gap'], 1e-14))
p('VOS-I final rel gap', H['VOS-AG-I']['gap'][-1] / (f['F0'] - r['Fs']), 'at 400', H['VOS-AG-I']['gap'][400] / (f['F0'] - r['Fs']),
  'nnz final', H['VOS-AG-I']['nnz'][-1], 'min nnz k>=10', H['VOS-AG-I']['nnz'][10:].min())
p('FISTA dstar at 400', H['FISTA']['dstar'][400], 'VOS-R dstar final', H['VOS-AG-II+restart']['dstar'][-1], 'ADMM', H['ADMM']['dstar'][-1])
p('PG gap at 100', H['PG']['gap'][100])
sw = D['sweep']; p('sweep it_1e10', {round(b, 3): v['it_1e10'] for b, v in sw.items()})
p('best beta', D['best_beta'])
p('admm rates', {round(float(b), 3): (round(v['rate_V'], 4) if v['rate_V'] == v['rate_V'] else None, round(v.get('rho_T', np.nan) ** 2, 4), v['monotone'], v['max_increase']) for b, v in AD['res'].items()})
# theory
for mu, c in TH['continuous'].items():
    p('cont', mu, 'maxlogE+t', c['max_logE_plus_t'], 'maxrelres', c['max_rel_resid'], 'nfev', c['nfev'], 'tmask', c['tmask'],
      'gf final', c['gap_gf'][-1] / c['gap_gf'][0])
for mu, d in TH['discrete'].items():
    p('disc', mu, 'kkt', d['kkt'], {k: (d[k]['max_ratio'], d[k]['min_ratio'], d[k]['rate_last'], d[k]['max_margin']) for k in d if 'VOS' in k}, '1-sqrtq', d['one_minus_sqrtq'])
p('vos1 tightness', json.load(open(f'{O}/vos1_tightness.json')))
eq = TH['equivalence']; rd = eq['lasso_vos2_vs_fista_reldiff']; dd = eq['lasso_vos2_vs_fista_diff_over_err']
p('equiv nest', eq['EN_mu0.1_vos2_vs_nesterov_maxrel'], 'fista rd', {k: rd[k] for k in [2, 50, 100, 200, 300, 400]},
  'diff/err range k>=50', min(dd[50:]), max(dd[50:]))
for k, v in TH['overshoot'].items():
    g = np.array(v['gap']); idx = np.nonzero(g <= 1e-10)[0]
    p('overshoot', k, 'increases', v['n_increase'], 'first k gap<=1e-10', int(idx[0]) if len(idx) else None)
p('kappa', KJ)
p('slopes', json.load(open(f'{O}/kappa_slopes.json')))
# gamma_k vs 4L/(k+1)^2
gam = 100.0; rat = []
for k in range(300):
    if k in (10, 100, 299): rat.append((k, gam / (4 * 100 / (k + 1) ** 2)))
    gam = gam / (1 + alpha_II(100.0, 0.0, gam))
p('gamma_k / (4L/(k+1)^2)', rat)
# null-space share of initial ridge gap for mu=1e-3, seed-1 x0
A, b, xt, _ = load_data(); x0 = np.random.default_rng(1).standard_normal(A.shape[1])
U, s, Vt = np.linalg.svd(A, full_matrices=True)
PN = Vt[1000:].T @ (Vt[1000:] @ x0)
P = Problem(A, b, mu=1e-3, smax=10.0); P.set_ref(reference(P))
p('null-space share mu=1e-3: mu/2||P_N x0||^2 =', 0.5e-3 * PN @ PN, ' initial gap', P.gap(x0))
p('restart counts', json.load(open(f'{O}/restart_counts.json')))
p('robustness', json.load(open(f'{O}/lasso_robustness.json')))
p('k_stat', {m: TB['phases'][m]['k_stat'] for m in TB['phases']})
p('timings', D['T'])
open(f'{O}/verify_numbers.txt', 'w').write('\n'.join(lines))
