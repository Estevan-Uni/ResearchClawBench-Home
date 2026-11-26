"""Terminal-model (Mayo-Lewis) Monte Carlo of six-monomer free-radical copolymerization (after paper_001).
P(x | chain end y) = k_yx [M_x] / sum_z k_yz [M_z], with k_yx = k_yy / r_yx  ->  P ∝ [M_x] / r_yx (r_yy = 1).
Monomers are removed from the pool as they are consumed (composition drift).
Two growth modes: 'frp' (conventional free radical: chains born and completed sequentially as conversion
proceeds; drift -> chain-to-chain heterogeneity) and 'crp' (all chains grow simultaneously; drift -> gradient).
Reactivity ratios: NO measured six-monomer r values are available in the provided files; the non-ideal
scenarios are HYPOTHETICAL sensitivity cases."""
import sys; sys.path.insert(0, 'code')
from style import *
import pandas as pd, numpy as np
rng0 = np.random.default_rng(0)
key = pd.read_csv('outputs/key_formulations.csv')
FORMS = {'G-042 (G-max)': key.iloc[0][MON].astype(float).values, 'R1-max (old-SM-GP#1)': key.iloc[2][MON].astype(float).values}
I = {m: i for i, m in enumerate(MON)}
def R_ideal(): return np.ones((6, 6))
def R_mild():  # same-family acrylate dispersion (hypothetical): log10 r ~ N(0, 0.15)
    r = 10 ** rng0.normal(0, 0.15, (6, 6)); np.fill_diagonal(r, 1); return r
def R_aam():   # acrylamide differs from the five acrylates (hypothetical): AAm adds slowly to acrylate radicals; AAm radical prefers acrylates
    r = np.ones((6, 6)); a = I['AAm']
    r[:, a] = 3.0; r[a, :] = 0.5; r[a, a] = 1; return r
def R_ionic():  # charged ATAC strongly self-avoiding / slow (hypothetical, e.g. electrostatic effects)
    r = np.ones((6, 6)); a = I['ATAC']; r[:, a] = 4.0; r[a, :] = 0.25; r[a, a] = 1; return r
def R_block():  # hydrophobic BA and aromatic PEA each prefer self-propagation (hypothetical): r_BA,PEA = r_PEA,BA = 5
    r = np.ones((6, 6)); r[I['BA'], :] = 5; r[I['PEA'], :] = 5; r[I['BA'], I['BA']] = r[I['PEA'], I['PEA']] = 1
    r[I['BA'], I['PEA']] = 5; r[I['PEA'], I['BA']] = 5; return r
SCEN = {'ideal (all r = 1)': R_ideal(), 'mild dispersion (log10 r ~ N(0,0.15))': R_mild(),
        'AAm acrylamide mismatch (r = 3 / 0.5)': R_aam(), 'ATAC slow (r = 4 / 0.25)': R_ionic(),
        'BA/PEA self-preferring (r = 5)': R_block()}

def simulate(f, R, DP=200, nchain=600, conv=0.95, mode='frp', seed=1):
    rng = np.random.default_rng(seed)
    pool = np.round(f / f.sum() * DP * nchain).astype(float)
    total = pool.sum(); target = int(conv * total)
    chains = [[] for _ in range(nchain)]
    added = []; K = 1.0 / R  # relative rate constants k_yx / k_yy
    def pick(prev):
        w = pool * (K[prev] if prev is not None else 1.0)
        w = np.clip(w, 0, None); p = w / w.sum()
        return rng.choice(6, p=p)
    n = 0
    if mode == 'frp':
        ci = 0
        while n < target and ci < nchain:
            ch = chains[ci]
            while len(ch) < DP and n < target:
                x = pick(ch[-1] if ch else None); ch.append(x); pool[x] -= 1; added.append(x); n += 1
            ci += 1
    else:
        while n < target:
            c = rng.integers(nchain); ch = chains[c]
            if len(ch) >= DP * 1.5: continue
            x = pick(ch[-1] if ch else None); ch.append(x); pool[x] -= 1; added.append(x); n += 1
    return [np.array(c) for c in chains if len(c) > 0], np.array(added), total

def metrics(chains, added, total, f):
    f = f / f.sum()
    conv = np.arange(1, len(added) + 1) / total
    oh = np.eye(6)[added]
    cum = np.cumsum(oh, 0) / np.arange(1, len(added) + 1)[:, None]
    # instantaneous composition in 20 conversion windows
    w = np.array_split(np.arange(len(added)), 19)
    inst = np.array([oh[ix].mean(0) for ix in w]); cw = np.array([conv[ix].mean() for ix in w])
    comp = np.array([np.bincount(c, minlength=6) / len(c) for c in chains if len(c) >= 20])
    # dyads
    dy = np.zeros((6, 6))
    for c in chains:
        np.add.at(dy, (c[:-1], c[1:]), 1)
    dy /= dy.sum()
    # run lengths
    runs = {m: [] for m in range(6)}
    for c in chains:
        if len(c) == 0: continue
        b = np.flatnonzero(np.diff(c)) + 1; st = np.r_[0, b]; en = np.r_[b, len(c)]
        for s, e in zip(st, en): runs[c[s]].append(e - s)
    # 20-mer segment hydrophobic+aromatic fraction (BA+PEA) vs ideal binomial expectation
    seg = []; segA = []
    for c in chains:
        for s in range(0, len(c) - 19, 20):
            seg.append(np.isin(c[s:s + 20], [I['BA'], I['PEA']]).mean()); segA.append(int((c[s:s + 20] == I['ATAC']).sum()))
    return dict(conv=conv, cum=cum, inst=inst, cw=cw, comp=comp, dyad=dy, runs=runs, seg=np.array(seg), segA=np.array(segA))

results = {}; summ = []
for fn, f in FORMS.items():
    for sn, R in SCEN.items():
        for mode in ['frp', 'crp']:
            ch, add, tot = simulate(f, R, mode=mode)
            M = metrics(ch, add, tot, f); results[(fn, sn, mode)] = M
            fr = f / f.sum(); present = fr > 0
            sd = M['comp'].std(0)
            dev = np.abs(M['comp'] - fr).sum(1) / 2  # per-chain total-variation distance to feed
            mr = {MON[m]: np.mean(M['runs'][m]) if M['runs'][m] else np.nan for m in range(6)}
            ideal_run = {MON[m]: 1 / (1 - fr[m]) if fr[m] > 0 else np.nan for m in range(6)}
            pBP = fr[I['BA']] + fr[I['PEA']]
            summ.append(dict(formulation=fn, scenario=sn, mode=mode,
                             mean_chain_TVdist_to_feed=dev.mean(), p90_chain_TVdist=np.quantile(dev, 0.9),
                             frac_chains_within_0p1=float((dev < 0.1).mean()),
                             inst_ATAC_start=M['inst'][0, I['ATAC']], inst_ATAC_end=M['inst'][-1, I['ATAC']],
                             inst_BA_start=M['inst'][0, I['BA']], inst_BA_end=M['inst'][-1, I['BA']],
                             meanrun_BA=mr['BA'], idealrun_BA=ideal_run['BA'], meanrun_PEA=mr['PEA'], idealrun_PEA=ideal_run['PEA'],
                             meanrun_ATAC=mr['ATAC'], idealrun_ATAC=ideal_run['ATAC'],
                             seg20_BAPEA_sd=M['seg'].std(), seg20_BAPEA_sd_binomial=np.sqrt(pBP * (1 - pBP) / 20),
                             dyad_BA_PEA_obs=M['dyad'][I['BA'], I['PEA']] + M['dyad'][I['PEA'], I['BA']],
                             dyad_BA_PEA_random=2 * fr[I['BA']] * fr[I['PEA']],
                             segA_frac_zero_ATAC=float((M['segA'] == 0).mean()), segA_frac_zero_binomial=(1 - fr[I['ATAC']]) ** 20,
                             segA_var_ratio=M['segA'].var() / (20 * fr[I['ATAC']] * (1 - fr[I['ATAC']])),
                             dyad_ATAC_ATAC_obs=M['dyad'][I['ATAC'], I['ATAC']], dyad_ATAC_ATAC_random=fr[I['ATAC']] ** 2))
            print(fn, sn, mode, round(dev.mean(), 3))
S = pd.DataFrame(summ); S.to_csv('outputs/copolymer_mc_summary.csv', index=False)
pd.DataFrame({sn: pd.DataFrame(R, index=MON, columns=MON).stack() for sn, R in SCEN.items()}).to_csv('outputs/copolymer_mc_reactivity_ratios_used.csv')

# ---------------- Figure 7: drift ----------------
fn = 'R1-max (old-SM-GP#1)'
show = ['ideal (all r = 1)', 'mild dispersion (log10 r ~ N(0,0.15))', 'ATAC slow (r = 4 / 0.25)', 'BA/PEA self-preferring (r = 5)']
fig, axs = plt.subplots(2, 4, figsize=(15, 6.4), sharex=True)
for j, sn in enumerate(show):
    for row, fname in enumerate(['G-042 (G-max)', fn]):
        M = results[(fname, sn, 'frp')]; ax = axs[row, j]; f = FORMS[fname] / FORMS[fname].sum()
        for m in range(6):
            if f[m] == 0: continue
            ax.plot(M['cw'], M['inst'][:, m], color=MONCOL[MON[m]], lw=2, label=MON[m])
            ax.plot(M['conv'][500::200], M['cum'][500::200, m], color=MONCOL[MON[m]], lw=1, ls='--')
            ax.axhline(f[m], color=MONCOL[MON[m]], lw=0.6, ls=':')
        ax.set_ylim(0, 0.9 if row else 0.8)
        if row == 0: ax.set_title(sn, fontsize=9)
        if j == 0: ax.set_ylabel(f'{fname}\nmole fraction in copolymer')
        if row == 1: ax.set_xlabel('conversion')
        if j == 0 and row == 0: ax.legend(fontsize=7, ncol=2, loc='upper right')
fig.suptitle('Composition drift (terminal-model Monte Carlo, DP 200; solid = instantaneous, dashed = cumulative, dotted = feed). Non-ideal r are hypothetical.', fontsize=9.5)
fig.tight_layout(); save(fig, 'fig07_copolymer_drift')

# ---------------- Figure 8: sequence statistics ----------------
fig, axs = plt.subplots(1, 4, figsize=(15.5, 3.8))
cols = [CAT[0], CAT[3], CAT[1], CAT[7]]
ax = axs[0]
for k, sn in enumerate(show):
    dev = np.abs(results[(fn, sn, 'frp')]['comp'] - FORMS[fn] / FORMS[fn].sum()).sum(1) / 2
    ax.hist(dev, bins=np.linspace(0, 0.6, 31), histtype='step', lw=2, color=cols[k], label=sn.split(' (')[0])
ax.set_xlabel('per-chain distance to target composition\n(total variation, FRP mode)'); ax.set_ylabel('chains'); ax.legend(fontsize=7)
ax.set_title('a  Chain-to-chain fidelity (R1-max)', loc='left', fontsize=9.5)
ax = axs[1]
kk = np.arange(0, 11)
for k, sn in enumerate(show):
    sa = results[(fn, sn, 'frp')]['segA']
    ax.plot(kk, np.bincount(sa, minlength=21)[:11] / len(sa), 'o-', color=cols[k], ms=4, lw=1.5)
from scipy.stats import binom
pA = FORMS[fn][I['ATAC']] / FORMS[fn].sum()
ax.plot(kk, binom.pmf(kk, 20, pA), 'k:', lw=1.5, label='ideal random (binomial)'); ax.legend(fontsize=7)
ax.set_xlabel('ATAC units per 20-mer segment'); ax.set_ylabel('fraction of segments'); ax.set_title('b  Cationic anchors per segment (R1-max)', loc='left', fontsize=9.5)
ax = axs[2]
for k, sn in enumerate(show):
    r = np.array(results[(fn, sn, 'frp')]['runs'][I['BA']]); h = np.bincount(r)[1:16] / len(r)
    ax.plot(np.arange(1, 16), h, 'o-', color=cols[k], ms=4, lw=1.5)
fb = FORMS[fn][I['BA']] / FORMS[fn].sum()
ax.plot(np.arange(1, 16), (1 - fb) * fb ** np.arange(0, 15), 'k:', lw=1.5, label='geometric (ideal)')
ax.set_yscale('log'); ax.set_xlabel('BA run (block) length'); ax.set_ylabel('fraction of BA runs'); ax.legend(fontsize=7)
ax.set_title('c  BA block-length distribution', loc='left', fontsize=9.5)
ax = axs[3]
M = results[(fn, 'ideal (all r = 1)', 'frp')]; f = FORMS[fn] / FORMS[fn].sum(); nz = [i for i in range(6) if f[i] > 0]
obs = M['dyad'][np.ix_(nz, nz)]; exp = np.outer(f[nz], f[nz])
M2 = results[(fn, 'BA/PEA self-preferring (r = 5)', 'frp')]; obs2 = M2['dyad'][np.ix_(nz, nz)]
lab = [f'{MON[a]}-{MON[b]}' for a in nz for b in nz]
xx = np.arange(len(lab))
ax.bar(xx - 0.2, (obs / exp).ravel(), 0.4, color=CAT[0], label='ideal')
ax.bar(xx + 0.2, (obs2 / exp).ravel(), 0.4, color=CAT[7], label='BA/PEA self-preferring')
ax.axhline(1, color=INK2, lw=0.8); ax.set_xticks(xx); ax.set_xticklabels(lab, rotation=60, fontsize=7)
ax.set_ylabel('dyad frequency / random expectation'); ax.legend(fontsize=7); ax.set_title('d  Neighbour (dyad) statistics, R1-max', loc='left', fontsize=9.5)
fig.tight_layout(); save(fig, 'fig08_copolymer_sequence_stats')
print(S.round(3).to_string())
