"""Regenerate all tables and figures from outputs/res_<METHOD>.jsonl (exact filenames, no globbing)."""
import os, sys, json, numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy.stats import wilcoxon
sys.path.insert(0, os.path.dirname(__file__))
from data_io import FAMILIES
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'outputs')
IMG = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'report', 'images')
MAIN = ['LNS2+RL', 'LNS2', 'LaCAM', 'PIBT', 'PP', 'MARL']
FILES = {'LNS2+RL': 'res_HYB_rate_W8.jsonl', 'LNS2': 'res_LNS2.jsonl', 'LaCAM': 'res_LaCAM.jsonl',
         'PIBT': 'res_PIBT.jsonl', 'PP': 'res_PP.jsonl', 'MARL': 'res_MARL.jsonl', 'LNS2 (seed 1)': 'res_LNS2_seed1.jsonl'}
ABL = {'LNS2+RL (rate, W=8)': 'res_HYB_rate_W8.jsonl', 'patience k=10': 'res_ABL_HYB_pat_W8.jsonl',
       'never switch (MARL only)': 'res_ABL_HYB_never_W8.jsonl', 'bandit': 'res_ABL_HYB_bandit_W8.jsonl',
       'W=full horizon': 'res_ABL_HYB_rate_Wfull.jsonl', 'W=16': 'res_ABL_HYB_rate_W16.jsonl',
       'heuristic policy': 'res_ABL_HYB_rate_W8_heur.jsonl', 'BC-only policy': 'res_ABL_HYB_rate_W8_bc.jsonl',
       'LNS2 (seed 1)': 'res_LNS2_seed1.jsonl', 'LNS2': 'res_LNS2.jsonl'}
COL = {'LNS2+RL': '#d62728', 'LNS2': '#1f77b4', 'LaCAM': '#2ca02c', 'PIBT': '#98df8a', 'PP': '#9467bd', 'MARL': '#ff7f0e',
       'LNS2 (seed 1)': '#aec7e8'}

def load(fn):
    p = os.path.join(OUT, fn)
    if not os.path.exists(p): return pd.DataFrame()
    d = pd.DataFrame([json.loads(l) for l in open(p)])
    return d.drop_duplicates(subset=['config', 'idx'], keep='first').reset_index(drop=True)

def short(cfg, fam, n):
    return f'{fam}-{n}'

def main():
    R = {m: load(f) for m, f in FILES.items()}
    R = {m: d for m, d in R.items() if len(d)}
    lines = []
    P = lambda *a: lines.append(' '.join(str(x) for x in a))
    # ---------- success table ----------
    rows = []
    for m, d in R.items():
        for (fam, n), g in d.groupby(['family', 'agents']):
            rows.append(dict(method=m, family=fam, agents=n, n=len(g), succ=g['success'].mean(),
                             cp=g['cp'].mean() if 'cp' in g else np.nan,
                             valid_all=bool(g.get('valid', pd.Series([True])).fillna(True).all())))
    S = pd.DataFrame(rows)
    S.to_csv(os.path.join(OUT, 'table_success.csv'), index=False)
    piv = S.pivot_table(index=['family', 'agents'], columns='method', values='succ')
    pivn = S.pivot_table(index=['family', 'agents'], columns='method', values='n')
    order = [m for m in MAIN + ['LNS2 (seed 1)'] if m in piv.columns]
    piv = piv.reindex(columns=order); pivn = pivn.reindex(columns=order)
    piv = piv.reindex(sorted(piv.index, key=lambda x: (FAMILIES.index(x[0]), x[1])))
    P('## Success rate per configuration (fraction of instances solved)')
    P('| family | agents | ' + ' | '.join(order) + ' |'); P('|' + '---|' * (2 + len(order)))
    for idx, r in piv.iterrows():
        P(f'| {idx[0]} | {idx[1]} | ' + ' | '.join('-' if pd.isna(r[m]) else f'{r[m]:.2f} ({int(pivn.loc[idx, m])})' for m in order) + ' |')
    # overall success
    P('\n## Overall success (instances solved / run)')
    for m, d in R.items():
        P(f'- {m}: {int(d.success.sum())}/{len(d)} = {d.success.mean():.3f}; valid solutions: '
          f'{int(d.get("valid", pd.Series(dtype=bool)).fillna(False).sum())}')
    # ---------- paired comparison hybrid vs LNS2 ----------
    if 'LNS2+RL' in R and 'LNS2' in R:
        H = R['LNS2+RL'].set_index(['config', 'idx']); L = R['LNS2'].set_index(['config', 'idx'])
        J = H.join(L, suffixes=('_h', '_l'), how='inner')
        P(f'\n## LNS2+RL vs LNS2 (paired, {len(J)} instances)')
        P(f'- success: LNS2+RL {J.success_h.sum()} vs LNS2 {J.success_l.sum()}; only-hybrid {((J.success_h) & (~J.success_l)).sum()}, only-LNS2 {((~J.success_h) & (J.success_l)).sum()}')
        rng = np.random.default_rng(0); diff = (J.success_h.astype(float) - J.success_l.astype(float)).values
        bs = [rng.choice(diff, len(diff)).mean() for _ in range(5000)]
        P(f'- success-rate difference {diff.mean():+.3f}, 95% bootstrap CI [{np.percentile(bs, 2.5):+.3f}, {np.percentile(bs, 97.5):+.3f}]')
        ratio = (J.cp_h + 1) / (J.cp_l + 1)
        P(f'- geometric mean (CP+1) ratio hybrid/LNS2: {np.exp(np.log(ratio).mean()):.3f}; hybrid lower CP on {(J.cp_h < J.cp_l).mean():.2f}, tie {(J.cp_h == J.cp_l).mean():.2f}')
        try: P(f'- Wilcoxon signed-rank on final CP: p = {wilcoxon(J.cp_h, J.cp_l).pvalue:.4g}')
        except Exception as e: P('- wilcoxon n/a', e)
        fr = []
        for fam, g in J.groupby('family_h'):
            rr = (g.cp_h + 1) / (g.cp_l + 1)
            try: p = wilcoxon(g.cp_h, g.cp_l).pvalue
            except Exception: p = np.nan
            fr.append(dict(family=fam, n=len(g), gm_ratio=np.exp(np.log(rr).mean()), cp_h=g.cp_h.mean(), cp_l=g.cp_l.mean(),
                           red=100 * (1 - g.cp_h.sum() / max(1, g.cp_l.sum())), succ_h=g.success_h.mean(), succ_l=g.success_l.mean(),
                           win=(g.cp_h < g.cp_l).mean(), p=p))
        FR = pd.DataFrame(fr); FR['order'] = FR.family.map(FAMILIES.index); FR = FR.sort_values('order')
        FR.to_csv(os.path.join(OUT, 'table_family_cp.csv'), index=False)
        P('\n| family | n | success hybrid | success LNS2 | mean CP hybrid | mean CP LNS2 | CP reduction % | GM (CP+1) ratio | hybrid wins | Wilcoxon p |')
        P('|---|---|---|---|---|---|---|---|---|---|')
        for _, r in FR.iterrows():
            P(f'| {r.family} | {r.n} | {r.succ_h:.2f} | {r.succ_l:.2f} | {r.cp_h:.1f} | {r.cp_l:.1f} | {r.red:+.1f} | {r.gm_ratio:.3f} | {r.win:.2f} | {r.p:.3g} |')
        # per-config CP
        P('\n| config | n | mean CP hybrid | mean CP LNS2 | CP reduction % |'); P('|---|---|---|---|---|')
        for (fam, n), g in sorted(J.groupby(['family_h', 'agents_h']), key=lambda x: (FAMILIES.index(x[0][0]), x[0][1])):
            P(f'| {fam}-{n} | {len(g)} | {g.cp_h.mean():.1f} | {g.cp_l.mean():.1f} | {100 * (1 - g.cp_h.sum() / max(1, g.cp_l.sum())):+.1f} |')
        # SoC
        both = J[J.success_h & J.success_l]
        P(f'\n- SoC on {len(both)} commonly solved instances: hybrid/LNS2 mean ratio {np.mean(both.soc_h / both.soc_l) if len(both) else float("nan"):.4f}')
        allsoc = (J.soc_h / J.sum_sp_h); allsocl = (J.soc_l / J.sum_sp_l)
        P(f'- SoC / sum of shortest paths over all instances (final, possibly colliding plan): hybrid {allsoc.mean():.3f}, LNS2 {allsocl.mean():.3f}')
        # MARL usage
        P(f'- hybrid: mean MARL iterations {J.n_marl_h.mean():.0f}, PP iterations {J.n_pp_h.mean():.0f}, mean switch time {J.switch_time_h.dropna().mean():.2f}s '
          f'(switched in {J.switch_time_h.notna().mean():.2f} of runs); CP reduced by MARL {J.gain_marl_h.sum()} vs PP {J.gain_pp_h.sum()}; '
          f'time in MARL {J.t_marl_h.sum():.0f}s vs PP {J.t_pp_h.sum():.0f}s')
        P(f'- iterations/s: hybrid {np.mean(J.iters_h / J.time_h):.1f}, LNS2 {np.mean(J.iters_l / J.time_l):.1f}')
    # seed control
    if 'LNS2 (seed 1)' in R and 'LNS2' in R:
        A = R['LNS2 (seed 1)'].set_index(['config', 'idx']).join(R['LNS2'].set_index(['config', 'idx']), suffixes=('_s', '_l'), how='inner')
        rr = (A.cp_s + 1) / (A.cp_l + 1)
        try: p = wilcoxon(A.cp_s, A.cp_l).pvalue
        except Exception: p = np.nan
        P(f'\n## Seed-noise control: LNS2 seed1 vs seed0 on {len(A)} instances: GM (CP+1) ratio {np.exp(np.log(rr).mean()):.3f}, '
          f'success {A.success_s.sum()} vs {A.success_l.sum()}, Wilcoxon p={p:.3g}')
    # SoC of all methods on instances solved by all of LNS2+RL, LNS2, LaCAM
    sols = {m: d[d.success].set_index(['config', 'idx'])['soc'] for m, d in R.items() if len(d) and d.success.any()}
    if sols:
        P('\n## SoC / sum-of-shortest-paths on solved instances (per method, own solved set)')
        for m, d in R.items():
            s = d[d.success]
            if len(s): P(f'- {m}: n={len(s)}, mean SoC ratio {(s.soc / s.sum_sp).mean():.3f}')
        common = None
        for m in ['LNS2+RL', 'LNS2', 'LaCAM']:
            if m in sols: common = sols[m].index if common is None else common.intersection(sols[m].index)
        if common is not None and len(common):
            P(f'- on {len(common)} instances solved by LNS2+RL, LNS2 and LaCAM: ' + ', '.join(
                f'{m} SoC {sols[m].loc[common].mean():.0f}' for m in ['LNS2+RL', 'LNS2', 'LaCAM'] if m in sols))
    open(os.path.join(OUT, 'summary.md'), 'w').write('\n'.join(lines))
    print('\n'.join(lines))
    figs(R, S)
    ablation()

def figs(R, S):
    # success bar chart per family
    fams = [f for f in FAMILIES if f in set(S.family)]
    meths = [m for m in MAIN if m in R]
    fig, axes = plt.subplots(1, len(fams), figsize=(3.0 * len(fams), 3.2), sharey=True)
    for ax, fam in zip(np.atleast_1d(axes), fams):
        sub = S[S.family == fam]; ns = sorted(sub.agents.unique()); x = np.arange(len(ns)); wd = 0.8 / len(meths)
        for j, m in enumerate(meths):
            v = [sub[(sub.method == m) & (sub.agents == n)].succ.mean() if len(sub[(sub.method == m) & (sub.agents == n)]) else np.nan for n in ns]
            ax.bar(x + (j - len(meths) / 2 + 0.5) * wd, v, wd, color=COL[m], label=m)
        ax.set_xticks(x); ax.set_xticklabels(ns, fontsize=8); ax.set_title(fam, fontsize=10); ax.set_ylim(0, 1.05)
        ax.set_xlabel('agents', fontsize=8)
    np.atleast_1d(axes)[0].set_ylabel('success rate'); np.atleast_1d(axes)[-1].legend(fontsize=7, loc='upper right')
    plt.tight_layout(); plt.savefig(os.path.join(IMG, 'success_rates.png'), dpi=150); plt.close()
    # anytime CP curves (normalised by initial CP) per family
    if 'LNS2+RL' in R and 'LNS2' in R:
        fig, axes = plt.subplots(1, len(fams), figsize=(3.0 * len(fams), 3.0))
        for ax, fam in zip(np.atleast_1d(axes), fams):
            for m in ['LNS2+RL', 'LNS2']:
                d = R[m][R[m].family == fam]
                curves = []
                for _, r in d.iterrows():
                    tr = np.array([x[:2] for x in r['trace']], float); B = r['budget']
                    grid = np.linspace(0, B, 50); cp0 = max(tr[0, 1], 1)
                    idx = np.searchsorted(tr[:, 0], grid, side='right') - 1
                    vals = np.where(idx >= 0, tr[np.maximum(idx, 0), 1], tr[0, 1]) / cp0
                    curves.append(vals)
                if curves:
                    c = np.array(curves); ax.plot(grid / B, np.exp(np.log(c + 1e-3).mean(0)), color=COL[m], label=m)
            ax.set_yscale('log'); ax.set_title(fam, fontsize=10); ax.set_xlabel('fraction of time budget', fontsize=8)
        np.atleast_1d(axes)[0].set_ylabel('CP / initial CP (geo-mean)'); np.atleast_1d(axes)[0].legend(fontsize=7)
        plt.tight_layout(); plt.savefig(os.path.join(IMG, 'anytime_cp.png'), dpi=150); plt.close()
        # CP ratio box per family
        H = R['LNS2+RL'].set_index(['config', 'idx']); L = R['LNS2'].set_index(['config', 'idx'])
        J = H.join(L, suffixes=('_h', '_l'), how='inner')
        J['lr'] = np.log2((J.cp_h + 1) / (J.cp_l + 1))
        fig, ax = plt.subplots(figsize=(7, 3.2))
        data = [J[J.family_h == f].lr.values for f in fams]
        ax.boxplot(data, showmeans=True); ax.set_xticks(range(1, len(fams) + 1)); ax.set_xticklabels(fams, fontsize=8, rotation=15)
        ax.axhline(0, color='k', lw=0.8); ax.set_ylabel('log2((CP_hyb+1)/(CP_LNS2+1))')
        for i, f in enumerate(fams):
            ax.axvspan(i + 0.5, i + 1.5, color='#eeeeee' if f.startswith('random') else '#fff3e0', zorder=-1)
        ax.set_title('Final CP of LNS2+RL relative to LNS2 (<0: hybrid better); grey = training distribution', fontsize=9)
        plt.tight_layout(); plt.savefig(os.path.join(IMG, 'cp_ratio_family.png'), dpi=150); plt.close()

def ablation():
    A = {k: load(f) for k, f in ABL.items()}; A = {k: d for k, d in A.items() if len(d)}
    if 'LNS2' not in A or len(A) < 2: return
    L = A['LNS2'].set_index(['config', 'idx'])
    rows = []
    for k, d in A.items():
        J = d.set_index(['config', 'idx']).join(L, suffixes=('', '_l'), how='inner')
        rr = (J.cp + 1) / (J.cp_l + 1)
        try: p = wilcoxon(J.cp, J.cp_l).pvalue if k != 'LNS2' else np.nan
        except Exception: p = np.nan
        st = J.switch_time.dropna() if 'switch_time' in J else pd.Series(dtype=float)
        rows.append(dict(variant=k, n=len(J), success=J.success.mean(), gm_ratio=np.exp(np.log(rr).mean()), p=p,
                         room_ratio=np.exp(np.log(rr[J.family == 'room']).mean()) if (J.family == 'room').any() else np.nan,
                         switch=st.mean() if len(st) else np.nan, marl_share=(J.n_marl / J.iters).mean() if 'n_marl' in J else 0))
    T = pd.DataFrame(rows); T.to_csv(os.path.join(OUT, 'table_ablation.csv'), index=False)
    s = ['\n## Ablations (paired vs LNS2 on the ablation subset)', '| variant | n | success | GM (CP+1) ratio vs LNS2 | Wilcoxon p | room GM ratio | mean switch time (s) | MARL share of iterations |', '|---|---|---|---|---|---|---|---|']
    for _, r in T.iterrows():
        s.append(f'| {r.variant} | {r.n} | {r.success:.2f} | {r.gm_ratio:.3f} | {r.p:.3g} | {r.room_ratio:.3f} | {r.switch:.2f} | {r.marl_share:.2f} |')
    print('\n'.join(s)); open(os.path.join(OUT, 'summary.md'), 'a').write('\n'.join(s))
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.barh(T.variant, T.gm_ratio, color=['#d62728' if v < 1 else '#7f7f7f' for v in T.gm_ratio]); ax.axvline(1, color='k', lw=0.8)
    ax.set_xlabel('geo-mean (CP+1) ratio vs LNS2 (lower is better)'); plt.tight_layout()
    plt.savefig(os.path.join(IMG, 'ablation.png'), dpi=150); plt.close()

if __name__ == '__main__':
    main()
