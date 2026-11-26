"""Generate all markdown tables in the report from CSVs in outputs/."""
import pandas as pd
O = '/workspace/outputs/'; out = []
def md(df, fl=3): return df.to_markdown(floatfmt=f'.{fl}f') if hasattr(df, 'to_markdown') else df.round(fl).to_string()
ov = pd.read_csv(O + 'rpe_overview_summary.csv', index_col=0); out.append('## T_overview\n' + md(ov))
m = pd.read_csv(O + 'rpe_metrics.csv')
t = m[(m.k == 30) | (m.method == 'all_features')].drop(columns=['k', 'dpt_rho_all_graph_cyc_cells']).set_index('method')
out.append('## T_rpe_k30\n' + md(t))
for met in ['dpt_rho_all', 'acc', 'nmi']:
    out.append(f'## T_rpe_kcurve_{met}\n' + md(m[m.method != 'all_features'].pivot(index='method', columns='k', values=met), 2))
ci = pd.read_csv(O + 'rpe_bootstrap_ci_k30.csv'); ci['CI95'] = ci.apply(lambda r: f"{r['mean']:.3f} [{r.lo:.3f}, {r.hi:.3f}]", axis=1)
out.append('## T_boot\n' + md(ci.pivot(index='method', columns='cells', values='CI95')))
out.append('## T_paired\n' + md(pd.read_csv(O + 'rpe_paired_diff_k30.csv').set_index('method')))
out.append('## T_diag\n' + md(pd.read_csv(O + 'rpe_seed_diagnostics.csv').set_index('variant')))
st = pd.read_csv(O + 'rpe_stability.csv').groupby('method').jaccard.agg(['mean', 'std']); out.append('## T_stab\n' + md(st))
nz = pd.read_csv(O + 'rpe_noise.csv')
out.append('## T_rpe_noise_dpt\n' + md(nz.pivot_table(index='method', columns=['kind', 'level'], values='dpt_rho_all'), 2))
sim = pd.read_csv(O + 'sim_results.csv')
cl = sim[sim.factor == 'clean']
for met in ['precision', 'acc', 'nmi', 'dpt_rho']:
    out.append(f'## T_sim_clean_{met}\n' + md(cl.groupby(['method', 'topology'])[met].mean().unstack(), 2))
out.append('## T_sim_clean_conf\n' + md(cl.groupby(['method', 'topology'])['conf_frac'].mean().unstack(), 2))
sw = pd.concat([pd.concat([cl.assign(factor=f), sim[sim.factor == f]]) for f in ['snr', 'library', 'dropout']])
for met in ['precision', 'dpt_rho', 'acc']:
    out.append(f'## T_sweep_{met}\n' + md(sw.groupby(['method', 'factor', 'level'])[met].mean().unstack(['factor', 'level']), 2))
# degradation: clean minus worst level, averaged
deg = sw.groupby(['method', 'factor', 'level'])['dpt_rho'].mean().unstack('level'); deg = (deg[0] - deg[3]).unstack('factor')
out.append('## T_sweep_drop_dpt (clean - hardest level)\n' + md(deg, 2))
out.append('## T_zero_frac\n' + md(sim.groupby(['factor', 'level']).zero_frac.mean().to_frame(), 3))
out.append(f'## reps\n{sim.rep.nunique()} reps; datasets={len(sim.groupby(["rep","topology","factor","level"]))}')
open(O + 'tables.md', 'w').write('\n\n'.join(out)); print('ok')
