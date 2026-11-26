"""Paper-format verification figures and tables from outputs/metrics.npz and the forecast arrays."""
import os, json, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import *

IM = '/workspace/report/images'; os.makedirs(IM, exist_ok=True)
M = dict(np.load(f'{ROOT}/metrics.npz'))
inits = M['inits'].astype('datetime64[h]'); n = len(inits)
groups = np.arange(n) // 4                       # 44 groups of 4 consecutive inits (block bootstrap unit)
V8 = ['MSL', 'T2M', 'U10', 'V10', 'Z500', 'T500', 'U500', 'V500']
VI = {v: VER.index(v) for v in VER}
UNITS = {'MSL': 'Pa', 'T2M': 'K', 'U10': 'm/s', 'V10': 'm/s', 'Z500': 'm²/s²', 'T500': 'K', 'U500': 'm/s', 'V500': 'm/s', 'Z850': 'm²/s²', 'T850': 'K', 'TP6': 'm'}
LEADS = np.arange(1, 61) * 0.25                   # days
SYS = {'cascade': ('FuXi-style cascade (ours)', '#d62728', '-'),
       'short_only': ('Single model: Short only (ours)', '#1f77b4', '-'),
       'shortplus': ('Short+ (compute-matched, ours)', '#9467bd', '--'),
       'ens_mean': ('ECMWF ENS mean', '#000000', '-'),
       'hres': ('ECMWF HRES', '#7f7f7f', '--'),
       'fuxi_wb2': ('FuXi (published, 0.25°→5.625°)', '#ff7f0e', ':'),
       'persistence': ('Persistence', '#bcbd22', ':'),
       'climatology': ('Climatology', '#17becf', ':')}
SYS = {k: v for k, v in SYS.items() if f'{k}_mse' in M}


def agg(s, metric, sel=None):
    a = M[f'{s}_{metric}']
    if sel is not None: a = a[sel]
    if metric == 'mse': return np.sqrt(np.nanmean(a, 0))
    return np.nanmean(a, 0)


def avail(s):
    """(lead mask) where system s has data for all inits that it covers."""
    a = M[f'{s}_mse'][..., 0]
    return ~np.all(np.isnan(a), 0)


def common_sel(*ss):
    ok = np.ones(n, bool)
    for s in ss:
        a = M[f'{s}_mse'][..., 0]; lm = avail(s)
        ok &= ~np.any(np.isnan(a[:, lm]), 1)
    return ok


def curves(metric, fname, ylab, vars_=V8, systems=None):
    systems = systems or [s for s in SYS if s != 'climatology' or metric == 'rmse']
    fig, axs = plt.subplots(2, 4, figsize=(18, 8.2))
    for ax, v in zip(axs.flat, vars_):
        j = VI[v]
        for s in systems:
            if s not in SYS: continue
            lab, c, ls = SYS[s]
            y = agg(s, 'mse' if metric == 'rmse' else 'acc')[:, j]; lm = avail(s) & np.isfinite(y)
            ax.plot(LEADS[lm], y[lm], color=c, ls=ls, lw=2 if s in ('cascade', 'ens_mean') else 1.4,
                    marker='o' if lm.sum() < 30 else None, ms=3, label=lab)
        for x in (5, 10): ax.axvline(x, color='k', lw=0.6, alpha=0.4)
        if metric == 'acc': ax.axhline(0.6, color='grey', lw=0.8, ls='--'); ax.set_ylim(0, 1.0)
        ax.set_title(v); ax.set_xlabel('lead time (days)'); ax.set_ylabel(f'{ylab}' + (f' [{UNITS[v]}]' if metric == 'rmse' else ''))
        ax.set_xlim(0, 15.2); ax.grid(alpha=0.3)
    axs.flat[0].legend(fontsize=8, loc='lower right' if metric == 'acc' else 'upper left')
    plt.suptitle(f'Latitude-weighted {ylab} vs ERA5, 2020 test inits (n={n}); vertical lines = cascade switch points (Short→Medium→Long)', fontsize=11)
    plt.tight_layout(); plt.savefig(f'{IM}/{fname}', dpi=110); plt.close()


def boot_ci(d, nb=2000, seed=0):
    """d: per-init values (n,) possibly NaN; block bootstrap over groups. Returns mean, lo, hi."""
    rng = np.random.default_rng(seed); g = np.unique(groups[np.isfinite(d)])
    gm = np.array([np.nanmean(d[groups == k]) for k in g]); cnt = np.array([np.isfinite(d[groups == k]).sum() for k in g])
    bs = [np.sum((gm * cnt)[ii]) / np.sum(cnt[ii]) for ii in (rng.integers(0, len(g), len(g)) for _ in range(nb))]
    return np.nanmean(d), np.percentile(bs, 2.5), np.percentile(bs, 97.5)


def normdiff(fname, ref='ens_mean', others=('cascade', 'short_only', 'shortplus', 'hres', 'fuxi_wb2')):
    """(ACC_c-ACC_ref)/(1-ACC_ref) and (RMSE_c-RMSE_ref)/RMSE_ref on paired inits/leads."""
    res = {}
    fig, axs = plt.subplots(2, 8, figsize=(26, 7), sharex=True)
    for col, v in enumerate(V8):
        j = VI[v]
        for s in others:
            if s not in SYS: continue
            sel = common_sel(s, ref); lm = avail(s) & avail(ref)
            ar = agg(ref, 'acc', sel)[:, j]; ac = agg(s, 'acc', sel)[:, j]
            rr = agg(ref, 'mse', sel)[:, j]; rc = agg(s, 'mse', sel)[:, j]
            nacc = (ac - ar) / (1 - ar); nrm = (rc - rr) / rr
            lab, c, ls = SYS[s]
            axs[0, col].plot(LEADS[lm], nacc[lm], color=c, ls=ls, marker='o', ms=3, label=lab)
            axs[1, col].plot(LEADS[lm], nrm[lm], color=c, ls=ls, marker='o', ms=3, label=lab)
            res[(s, v)] = dict(leads=LEADS[lm].tolist(), nacc=nacc[lm].tolist(), nrmse=nrm[lm].tolist())
        for r in (0, 1):
            axs[r, col].axhline(0, color='k', lw=1); axs[r, col].grid(alpha=0.3)
            for x in (5, 10): axs[r, col].axvline(x, color='k', lw=0.5, alpha=0.4)
        axs[0, col].set_title(v); axs[1, col].set_xlabel('lead (days)')
        axs[0, col].set_ylim(-3.5, 0.7); axs[1, col].set_ylim(-0.35, 2.3)
    axs[0, 0].set_ylabel('normalised ACC difference\n(ACC−ACC_EM)/(1−ACC_EM)  [>0 better]')
    axs[1, 0].set_ylabel('normalised RMSE difference\n(RMSE−RMSE_EM)/RMSE_EM  [<0 better]')
    axs[0, 0].legend(fontsize=7, loc='lower left')
    plt.suptitle('Skill relative to the ECMWF ensemble mean (paired inits, ERA5 truth)', fontsize=12)
    plt.tight_layout(); plt.savefig(f'{IM}/{fname}', dpi=100); plt.close()
    return res


def scorecard(fname, s='cascade', ref='ens_mean'):
    """Heatmap of normalised RMSE difference with block-bootstrap significance markers."""
    sel = common_sel(s, ref); lm = np.where(avail(s) & avail(ref))[0]
    vars_ = V8 + ['Z850', 'T850', 'TP6']
    H = np.full((len(vars_), len(lm)), np.nan); S = np.zeros_like(H, bool)
    for a, v in enumerate(vars_):
        j = VI[v]
        for b, k in enumerate(lm):
            rr = np.sqrt(np.nanmean(M[f'{ref}_mse'][sel, k, j])); rc = np.sqrt(np.nanmean(M[f'{s}_mse'][sel, k, j]))
            H[a, b] = (rc - rr) / rr
            d = np.sqrt(M[f'{s}_mse'][sel, k, j]) - np.sqrt(M[f'{ref}_mse'][sel, k, j])
            dfull = np.full(n, np.nan); dfull[sel] = d
            m_, lo, hi = boot_ci(dfull, nb=500)
            S[a, b] = (lo > 0) or (hi < 0)
    fig, ax = plt.subplots(figsize=(13, 5))
    im = ax.imshow(H * 100, cmap='RdBu_r', vmin=-20, vmax=20, aspect='auto')
    for a in range(H.shape[0]):
        for b in range(H.shape[1]):
            ax.text(b, a, f'{H[a,b]*100:.0f}' + ('*' if S[a, b] else ''), ha='center', va='center', fontsize=7)
    ax.set_xticks(range(len(lm))); ax.set_xticklabels([f'{LEADS[k]:g}' for k in lm]); ax.set_xlabel('lead (days)')
    ax.set_yticks(range(len(vars_))); ax.set_yticklabels(vars_)
    plt.colorbar(im, ax=ax, label='% RMSE difference vs ENS mean (blue = cascade better)')
    ax.set_title(f'Scorecard: {SYS[s][0]} vs {SYS[ref][0]}  (* = 95% block-bootstrap CI excludes 0)')
    plt.tight_layout(); plt.savefig(f'{IM}/{fname}', dpi=110); plt.close()
    return H, S, [float(LEADS[k]) for k in lm], vars_


def win_fractions(s, ref, vars_=V8):
    sel = common_sel(s, ref); lm = avail(s) & avail(ref)
    out = {}
    for name, rng in [('all', (0, 99)), ('0-5d', (0, 5)), ('5-10d', (5.01, 10)), ('10-15d', (10.01, 15))]:
        L = lm & (LEADS >= rng[0]) & (LEADS <= rng[1])
        ac = agg(s, 'acc', sel)[L][:, [VI[v] for v in vars_]]; ar = agg(ref, 'acc', sel)[L][:, [VI[v] for v in vars_]]
        rc = agg(s, 'mse', sel)[L][:, [VI[v] for v in vars_]]; rr = agg(ref, 'mse', sel)[L][:, [VI[v] for v in vars_]]
        out[name] = dict(acc_win=float(np.mean(ac > ar)), rmse_win=float(np.mean(rc < rr)), ncell=int(ac.size))
    return out


def skillful_lead(s, v, thr=0.6):
    a = agg(s, 'acc')[:, VI[v]]; lm = avail(s) & np.isfinite(a)
    L, A = LEADS[lm], a[lm]
    if len(A) == 0: return None
    if A[-1] >= thr: return f'>{L[-1]:g}'
    k = np.where(A < thr)[0][0]
    if k == 0: return f'<{L[0]:g}'
    x0, x1, y0, y1 = L[k - 1], L[k], A[k - 1], A[k]
    return round(float(x0 + (y0 - thr) * (x1 - x0) / (y0 - y1)), 2)


if __name__ == '__main__':
    curves('acc', 'fig05_acc_8var.png', 'ACC')
    curves('rmse', 'fig06_rmse_8var.png', 'RMSE')
    nd = normdiff('fig07_normdiff_vs_ensmean.png')
    H, S, lds, vv = scorecard('fig08_scorecard_cascade_vs_em.png')
    tab = {}
    for s in [k for k in SYS if k not in ('climatology',)]:
        tab[s] = {v: skillful_lead(s, v) for v in V8 + ['Z850', 'T850']}
    wins = {f'{s}_vs_{r}': win_fractions(s, r) for s, r in [('cascade', 'ens_mean'), ('cascade', 'hres'), ('cascade', 'short_only'),
                                                             ('cascade', 'fuxi_wb2'), ('short_only', 'ens_mean'), ('fuxi_wb2', 'ens_mean'),
                                                             ('hres', 'ens_mean'), ('cascade', 'shortplus'), ('shortplus', 'ens_mean')]
            if s in SYS and r in SYS}
    json.dump(dict(skillful_lead=tab, wins=wins, scorecard=dict(H=H.tolist(), sig=S.tolist(), leads=lds, vars=vv)),
              open(f'{ROOT}/summary_tables.json', 'w'), indent=1, default=str)
    print(json.dumps(tab, indent=0, default=str)); print(json.dumps(wins, indent=0))
