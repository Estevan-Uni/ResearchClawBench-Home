import glob, json, collections, numpy as np
from figstyle import *
D = T = 0; H = set(); kinds = collections.Counter(); L = []; NP = []; aux = 0; chk = need = 0; ts = []; npts = []
ex_aux = 0
for f in sorted(glob.glob('../outputs/synth_A/*.jsonl')):
    for line in open(f):
        try: r = json.loads(line)
        except json.JSONDecodeError: continue
        D += 1; ts.append(r['t_total']); npts.append(r['n_points'])
        for t in r['theorems']:
            T += 1; H.add(t['h']); kinds[t['kind']] += 1; L.append(t['n_steps']); NP.append(t['n_prem'])
            aux += t['n_aux'] > 0
            if 'aux_needed' in t:
                chk += 1; need += bool(t['aux_needed'])
L = np.array(L); NP = np.array(NP)
st = dict(diagrams=D, theorems=T, unique_theorems=len(H), kinds=dict(kinds), mean_steps=float(L.mean()),
          median_steps=float(np.median(L)), max_steps=int(L.max()), frac_steps_ge10=float((L >= 10).mean()),
          frac_steps_ge20=float((L >= 20).mean()), mean_premises=float(NP.mean()),
          with_aux_point=int(aux), frac_aux=aux / T, aux_checked=chk, aux_needed=need,
          frac_needed_among_checked=need / max(1, chk), est_aux_needed_total=aux * need / max(1, chk),
          sec_per_diagram=float(np.mean(ts)), mean_points=float(np.mean(npts)))
print(json.dumps(st, indent=1))
json.dump(st, open('../outputs/synth_A_stats.json', 'w'), indent=1)
fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))
bins = np.arange(1, min(L.max(), 80) + 2)
ax[0].hist(np.minimum(L, 80), bins=np.arange(1, 82), color=SERIES[0]); ax[0].set_yscale('log')
ax[0].text(79, (L >= 80).sum() * 1.5, '≥80', ha='center', fontsize=8, color=INK2)
ax[0].set_xlabel('proof length (steps, incl. premises)'); ax[0].set_ylabel('theorems'); ax[0].set_title('(a) Proof-length distribution')
kc = kinds.most_common()
ax[1].barh([k for k, _ in kc][::-1], [v for _, v in kc][::-1], color=SERIES[0]); ax[1].set_xscale('log')
ax[1].set_title('(b) Synthetic theorem types'); ax[1].set_xlabel('count (log)')
lab = ['AlphaGeometry\n(cited)', 'this work']
th = [1e8, T]; ax_ = [9e6, st['est_aux_needed_total']]
x = np.arange(2)
ax[2].bar(x - 0.2, th, 0.38, color=SERIES[0], label='synthetic theorems')
ax[2].bar(x + 0.2, ax_, 0.38, color=SERIES[1], label='theorems needing aux points')
ax[2].set_yscale('log'); ax[2].set_xticks(x); ax[2].set_xticklabels(lab); ax[2].set_ylim(1, 3e9)
for xi, v in zip(x - 0.2, th): ax[2].text(xi, v * 1.4, '%.1e' % v, ha='center', fontsize=8, color=INK2)
for xi, v in zip(x + 0.2, ax_): ax[2].text(xi, v * 1.4, '%.1e' % v, ha='center', fontsize=8, color=INK2)
ax[2].legend(loc='upper right'); ax[2].set_title('(c) Data scale vs. AlphaGeometry')
plt.tight_layout(); plt.savefig('/workspace/report/images/fig6_synthetic_data.png')
