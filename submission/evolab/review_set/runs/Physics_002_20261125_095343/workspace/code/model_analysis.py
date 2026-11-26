"""TRB fits, gate-count error model with Monte-Carlo uncertainty, depth regressions, mirror inference,
XEB-vs-MB consistency. Writes outputs/*.csv and json."""
import os, sys, json, pickle, numpy as np, pandas as pd
from scipy.optimize import curve_fit, minimize_scalar
sys.path.insert(0, os.path.dirname(__file__))
from estimators import xeb_terms, pooled_pt_mle, boot_mean
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = lambda f: os.path.join(ROOT, 'outputs', f)
rng = np.random.default_rng(2024)
inst = pd.read_csv(OUT('per_instance_estimates.csv'))
P = pd.read_csv(OUT('pooled_estimates.csv'))
res = {}

# ---- pooled PT-MLE (replaces biased per-circuit average)
recs = pickle.load(open(OUT('data_cache.pkl'), 'rb'))
pm = []
for key in sorted({(r['family'], r['N'], r['d']) for r in recs if r['type'] == 'XEB' and 'ideal_probs' in r}):
    rr = [r for r in recs if (r['family'], r['N'], r['d']) == key and r['type'] == 'XEB']
    cx = [xeb_terms(r['counts'], r['ideal_probs'], r['N']) for r in rr]
    F, s = pooled_pt_mle([a for a, b in cx], [b for a, b in cx])
    # circuit bootstrap
    bs = []
    for b in range(300):
        idx = rng.integers(0, len(cx), len(cx))
        bs.append(pooled_pt_mle([cx[i][0] for i in idx], [cx[i][1] for i in idx])[0])
    pm.append(dict(family=key[0], N=key[1], d=key[2], method='PTMLE_pooled', n_circ=len(cx), shots=20, F=F,
                   sigma_boot=np.std(bs, ddof=1), sigma_naive=s))
P = pd.concat([P[P.method != 'PTMLE_pooled'], pd.DataFrame(pm)], ignore_index=True)
P.to_csv(OUT('pooled_estimates.csv'), index=False)

# ---- TRB: per-qubit bit survival P(d)=1/2+B p^d, pooled over duplicate-N datasets, circuit bootstrap within depth
trb = inst[inst.method == 'TRB_bitsurv']
f_trb = lambda d, B, p: 0.5 + B * p ** d
trb_rows, trb_boot = [], {}
for N, g in trb.groupby('N'):
    def fit(gg):
        m = gg.groupby('d').F.mean(); s = gg.groupby('d').F.std(ddof=1) / np.sqrt(gg.groupby('d').size())
        popt, _ = curve_fit(f_trb, m.index.values, m.values, p0=[0.5, 0.999], sigma=np.maximum(s.values, 1e-4), maxfev=20000)
        return popt
    B, p = fit(g)
    bs = []
    groups = {d: gd for d, gd in g.groupby('d')}
    for b in range(800):
        gg = pd.concat([gd.iloc[rng.integers(0, len(gd), len(gd))] for gd in groups.values()])
        try: bs.append(fit(gg))
        except Exception: pass
    bs = np.array(bs)
    emem = 0.75 * (1 - bs[:, 1]); espam = 0.5 - bs[:, 0]
    trb_boot[int(N)] = 0.75 * (1 - bs[:, 1])
    trb_rows.append(dict(N=N, n_datasets=g.family.nunique(), n_circ=len(g), B=B, p=p, eps_mem=0.75 * (1 - p),
                         eps_mem_se=emem.std(ddof=1), eps_spam=0.5 - B, eps_spam_se=espam.std(ddof=1)))
TRB = pd.DataFrame(trb_rows); TRB.to_csv(OUT('trb_fits.csv'), index=False)
print(TRB.to_string())
# SPAM: inverse-variance mean over N (per-qubit), clipped >=0
w = 1 / TRB.eps_spam_se ** 2
eps_spam = float((TRB.eps_spam * w).sum() / w.sum()); eps_spam_se = float(1 / np.sqrt(w.sum()))
res['eps_spam'] = eps_spam; res['eps_spam_se'] = eps_spam_se
emem = dict(zip(TRB.N.astype(int), TRB.eps_mem)); emem_se = dict(zip(TRB.N.astype(int), TRB.eps_mem_se))
emem_boot_dict = trb_boot

def model_lnF(N, d, e2q, es=None, em=None):
    es = eps_spam if es is None else es; em = emem[N] if em is None else em
    return N * np.log1p(-es) + d * (N / 2 * np.log1p(-e2q) + N * np.log1p(-em))

# ---- experimental points: one table of (dataset, N, d, F, sigma)
def pick(fam, meth):
    q = P[(P.family == fam) & (P.method == meth)]
    return q[['N', 'd', 'F', 'sigma_boot']].rename(columns={'sigma_boot': 'sigma'}).assign(family=fam, method=meth)
calib = pick('N_scan_depth12', 'linXEB'); calib = calib[calib.N <= 40]
held = pd.concat([pick('N40_verification', 'linXEB'), pick('N40_verification', 'MB'), pick('N_scan_depth12', 'MB'),
                  pick('N56_depths', 'MB')], ignore_index=True)
allpts = pd.concat([calib, held], ignore_index=True)

def fit_e2q(pts):
    chi = lambda e: (((pts.F - np.exp([model_lnF(int(n), int(d), e) for n, d in zip(pts.N, pts.d)])) / pts.sigma) ** 2).sum()
    r = minimize_scalar(chi, bounds=(1e-5, 2e-2), method='bounded'); e = r.x
    h = 1e-6; curv = (chi(e + h) - 2 * chi(e) + chi(e - h)) / h ** 2
    return e, np.sqrt(2 / curv), chi(e)
e2q, e2q_se, chi_cal = fit_e2q(calib)
e2q_all, e2q_all_se, chi_all = fit_e2q(allpts)
res.update(e2q_calib=e2q, e2q_calib_se=e2q_se, chi2_calib=chi_cal, dof_calib=len(calib) - 1,
           e2q_all=e2q_all, e2q_all_se=e2q_all_se, chi2_all=chi_all, dof_all=len(allpts) - 1)
print('e2q calib', e2q, e2q_se, 'all', e2q_all, e2q_all_se)

# ---- Monte-Carlo band (joint over e2q, eps_mem(N) bootstrap, SPAM)
NMC = 5000
E2 = rng.normal(e2q, e2q_se, NMC); ES = rng.normal(eps_spam, eps_spam_se, NMC)
EM = {N: rng.choice(v, NMC) for N, v in emem_boot_dict.items()}
def model_band(N, d):
    v = np.exp(N * np.log1p(-ES) + d * (N / 2 * np.log1p(-E2) + N * np.log1p(-EM[N])))
    return v.mean(), v.std(ddof=1), np.percentile(v, [16, 84, 2.5, 97.5])
grid = []
for N in sorted(emem):
    for d in range(0, 31):
        m, s, q = model_band(N, d)
        grid.append(dict(N=N, d=d, F_model=np.exp(model_lnF(N, d, e2q)), mc_mean=m, mc_sd=s, q16=q[0], q84=q[1], q025=q[2], q975=q[3],
                         F_model_e2q_m30=np.exp(model_lnF(N, d, 0.7 * e2q)), F_model_e2q_p30=np.exp(model_lnF(N, d, 1.3 * e2q)),
                         F_model_allfit=np.exp(model_lnF(N, d, e2q_all))))
G = pd.DataFrame(grid); G.to_csv(OUT('model_grid.csv'), index=False)

# ---- coverage & out-of-sample chi2
cov = []
for _, rw in allpts.iterrows():
    m, s, q = model_band(int(rw.N), int(rw.d))
    stot = np.sqrt(s ** 2 + rw.sigma ** 2)
    cov.append(dict(family=rw.family, method=rw.method, N=rw.N, d=rw.d, F=rw.F, sigma=rw.sigma, F_model=np.exp(model_lnF(int(rw.N), int(rw.d), e2q)),
                    model_sd=s, z=(rw.F - np.exp(model_lnF(int(rw.N), int(rw.d), e2q))) / stot, held_out=rw.family + rw.method != 'N_scan_depth12linXEB'))
C = pd.DataFrame(cov); C.to_csv(OUT('model_coverage.csv'), index=False)
H = C[C.held_out]
res.update(n_heldout=len(H), frac_1sigma=float((H.z.abs() < 1).mean()), frac_2sigma=float((H.z.abs() < 2).mean()),
           chi2_heldout=float((H.z ** 2).sum()), frac_1sigma_all=float((C.z.abs() < 1).mean()), frac_2sigma_all=float((C.z.abs() < 2).mean()))
for (fam, meth), g in H.groupby(['family', 'method']):
    res[f'chi2_{fam}_{meth}'] = [float((g.z ** 2).sum()), len(g)]
# error budget per layer at N=56, N=40
for N in [40, 56]:
    l2q = -(N / 2) * np.log1p(-e2q); lmem = -N * np.log1p(-emem[N])
    res[f'budget_N{N}'] = dict(lambda_2q=l2q, lambda_mem=lmem, frac_2q=l2q / (l2q + lmem), lambda_total=l2q + lmem,
                              spam_lnA=N * np.log1p(-eps_spam))

# ---- depth regressions ln F = ln A - lambda d (weighted, delta-method sigma on lnF)
def depth_fit(pts, dpred=None):
    x = pts.d.values.astype(float); y = np.log(pts.F.values); s = pts.sigma.values / pts.F.values
    W = np.diag(1 / s ** 2); X = np.vstack([np.ones_like(x), -x]).T
    cov = np.linalg.inv(X.T @ W @ X); beta = cov @ X.T @ W @ y
    chi2 = float(((y - X @ beta) / s) ** 2 @ np.ones_like(y))
    out = dict(lnA=beta[0], lnA_se=np.sqrt(cov[0, 0]), lam=beta[1], lam_se=np.sqrt(cov[1, 1]), cov01=cov[0, 1], chi2=chi2, dof=len(x) - 2)
    if dpred is not None:
        v = np.array([1, -dpred]); lp = v @ beta; lps = np.sqrt(v @ cov @ v)
        out.update(pred_d=dpred, pred_F=np.exp(lp), pred_F_se=np.exp(lp) * lps)
    return out
mb56 = pick('N56_depths', 'MB'); mb56_noD24 = mb56[mb56.d <= 20]
regs = {'N56_MB_d8-20': depth_fit(mb56_noD24, 12), 'N56_MB_d8-24': depth_fit(mb56, 12),
        'N56_MB_d8-20_excl12': depth_fit(mb56_noD24[mb56_noD24.d != 12], 12),
        'N40_MB': depth_fit(pick('N40_verification', 'MB'), 12), 'N40_linXEB': depth_fit(pick('N40_verification', 'linXEB'), 12),
        'N40_logXEB': depth_fit(pick('N40_verification', 'logXEB'), 12)}
r56 = regs['N56_MB_d8-20']; v = np.array([1, -24.]); bb = np.array([r56['lnA'], r56['lam']])
cv = np.array([[r56['lnA_se'] ** 2, r56['cov01']], [r56['cov01'], r56['lam_se'] ** 2]])
regs['N56_MB_d8-20']['extrap_d24_F'] = float(np.exp(v @ bb)); regs['N56_MB_d8-20']['extrap_d24_F_se'] = float(np.exp(v @ bb) * np.sqrt(v @ cv @ v))
res['depth_regressions'] = regs
print(json.dumps(regs, indent=1, default=float))

# ---- mirror inference at N56: d24 MB as mirror of depth-12 circuit
row24 = mb56[mb56.d == 24].iloc[0]; S, sS = row24.F, row24.sigma
F12_naive = np.sqrt(S); F12_naive_se = sS / (2 * np.sqrt(S))
A_reg = np.exp(r56['lnA']); A_reg_se = A_reg * r56['lnA_se']
A_spam = (1 - eps_spam) ** 56; A_spam_se = 56 * eps_spam_se * A_spam
def sq(A, sA): f = np.sqrt(A * S); return f, f * np.sqrt((sA / (2 * A)) ** 2 + (sS / (2 * S)) ** 2)
F12_Areg = sq(A_reg, A_reg_se); F12_Aspam = sq(A_spam, A_spam_se)
d12 = mb56[mb56.d == 12].iloc[0]
m12, s12, _ = model_band(56, 12)
mir = dict(S24=S, S24_se=sS, F12_sqrt=F12_naive, F12_sqrt_se=F12_naive_se, A_reg=A_reg, A_reg_se=A_reg_se, F12_sqrtAreg=F12_Areg[0], F12_sqrtAreg_se=F12_Areg[1],
           A_spam=A_spam, F12_sqrtAspam=F12_Aspam[0], F12_sqrtAspam_se=F12_Aspam[1], direct_MB12=d12.F, direct_MB12_se=d12.sigma,
           reg_d12=r56['pred_F'], reg_d12_se=r56['pred_F_se'], reg_excl12_d12=regs['N56_MB_d8-20_excl12']['pred_F'], reg_excl12_d12_se=regs['N56_MB_d8-20_excl12']['pred_F_se'],
           model_d12=np.exp(model_lnF(56, 12, e2q)), model_d12_se=s12,
           model_d24=np.exp(model_lnF(56, 24, e2q)), model_d24_se=model_band(56, 24)[1])
for k, (f, s) in {'sqrt': (F12_naive, F12_naive_se), 'sqrtAreg': F12_Areg, 'sqrtAspam': F12_Aspam}.items():
    mir[f'z_{k}_vs_direct'] = (f - d12.F) / np.hypot(s, d12.sigma)
    mir[f'z_{k}_vs_reg'] = (f - r56['pred_F']) / np.hypot(s, r56['pred_F_se'])
    mir[f'z_{k}_vs_model'] = (f - mir['model_d12']) / np.hypot(s, s12)
mir['z_d24_vs_extrap'] = (S - regs['N56_MB_d8-20']['extrap_d24_F']) / np.hypot(sS, regs['N56_MB_d8-20']['extrap_d24_F_se'])
mir['z_d24_vs_model'] = (S - mir['model_d24']) / np.hypot(sS, mir['model_d24_se'])
res['mirror'] = mir
print(json.dumps(mir, indent=1, default=float))

# ---- XEB vs MB consistency at matched (family,N,d)
lx = P[P.method == 'linXEB'].set_index(['family', 'N', 'd']); lg = P[P.method == 'logXEB'].set_index(['family', 'N', 'd'])
mbp = P[P.method == 'MB'].set_index(['family', 'N', 'd'])
J = lx[['F', 'sigma_boot']].join(mbp[['F', 'sigma_boot']], rsuffix='_MB', how='inner').join(lg[['F', 'sigma_boot']], rsuffix='_log')
J['z_lin_MB'] = (J.F - J.F_MB) / np.hypot(J.sigma_boot, J.sigma_boot_MB)
J['z_log_MB'] = (J.F_log - J.F_MB) / np.hypot(J.sigma_boot_log, J.sigma_boot_MB)
J['z_lin_log'] = (J.F - J.F_log) / np.hypot(J.sigma_boot, J.sigma_boot_log)  # NB correlated -> conservative
w = 1 / J.sigma_boot ** 2
slope = float((w * J.F * J.F_MB).sum() / (w * J.F_MB ** 2).sum()); slope_se = float(1 / np.sqrt((w * J.F_MB ** 2).sum()))
res['xeb_vs_mb'] = dict(n=len(J), chi2_lin=float((J.z_lin_MB ** 2).sum()), chi2_log=float((J.z_log_MB ** 2).sum()), slope_lin_on_MB=slope, slope_se=slope_se)
J.reset_index().to_csv(OUT('xeb_vs_mb.csv'), index=False)
# correlation of per-instance lin vs log XEB
pi = inst[inst.method.isin(['linXEB', 'logXEB'])].pivot_table(index=['family', 'N', 'd', 'r'], columns='method', values='F')
res['corr_lin_log_instance'] = float(pi.corr().iloc[0, 1])
# per-instance: is there resolvable instance-to-instance variation? chi2 of per-instance linXEB around pooled mean
vr = P[P.method.isin(['linXEB', 'logXEB', 'MB'])].groupby('method').var_ratio.agg(['mean', 'median', 'min', 'max'])
res['var_ratio'] = vr.to_dict()

# ---- fidelity summary table per (family,N,d)
rows = []
for (fam, N, d), g in P.groupby(['family', 'N', 'd']):
    row = dict(family=fam, N=N, d=d)
    for _, rw in g.iterrows():
        row[f'F_{rw.method}'] = rw.F; row[f'sig_{rw.method}'] = rw.sigma_boot
    m, s, q = model_band(int(N), int(d))
    row.update(F_model=np.exp(model_lnF(int(N), int(d), e2q)), sig_model=s, F_model_allfit=np.exp(model_lnF(int(N), int(d), e2q_all)))
    # combined: inverse-variance of linXEB and MB (independent samples)
    est = [(row.get('F_linXEB'), row.get('sig_linXEB')), (row.get('F_MB'), row.get('sig_MB'))]
    est = [(f, s) for f, s in est if f is not None and not np.isnan(f)]
    ww = np.array([1 / s ** 2 for f, s in est]); ff = np.array([f for f, s in est])
    row['F_combined'] = float((ww * ff).sum() / ww.sum()); row['sig_combined'] = float(1 / np.sqrt(ww.sum()))
    if fam == 'N56_depths':
        rg = regs['N56_MB_d8-20']; vv = np.array([1, -d]); b2 = np.array([rg['lnA'], rg['lam']])
        row['F_depthreg'] = float(np.exp(vv @ b2)); row['sig_depthreg'] = float(np.exp(vv @ b2) * np.sqrt(vv @ cv @ vv))
    rows.append(row)
S_ = pd.DataFrame(rows)
S_ = pd.concat([S_, pd.DataFrame([dict(family='N56_depths', N=56, d=12, F_mirror_inferred_sqrtS24=F12_naive, sig_mirror_inferred_sqrtS24=F12_naive_se,
                                       F_mirror_inferred_sqrtAS24=F12_Areg[0], sig_mirror_inferred_sqrtAS24=F12_Areg[1], note='inferred from d=24 MB (mirror of depth-12)')])], ignore_index=True)
S_.to_csv(OUT('fidelity_summary.csv'), index=False)
json.dump(res, open(OUT('model_results.json'), 'w'), indent=1, default=float)
print(json.dumps({k: v for k, v in res.items() if k not in ('depth_regressions', 'mirror')}, indent=1, default=float))
