import collections, numpy as np
from figstyle import *
from geo.problem import load_problems, Problem
probs = load_problems()
names, npts, nprem, goals, cons = [], [], [], [], collections.Counter()
for nm, t in probs.items():
    pr = Problem(nm, t)
    names.append(nm.replace('imo_', '').replace('_', ' ')); npts.append(len(pr.points))
    nprem.append(sum(p in ('coll','cong','perp','para','eqangle','cyclic','eqratio') for _,_,p,_ in pr.premises()))
    goals.append(pr.goal[0])
    for cl in pr.clauses:
        for c, _ in cl.constructs: cons[c] += 1
fig, ax = plt.subplots(1, 3, figsize=(15, 4.6), gridspec_kw=dict(width_ratios=[2.4, 1, 1.4]))
x = np.arange(len(names))
ax[0].bar(x - 0.2, npts, 0.38, color=SERIES[0], label='points')
ax[0].bar(x + 0.2, nprem, 0.38, color=SERIES[1], label='premise predicates')
ax[0].set_xticks(x); ax[0].set_xticklabels(names, rotation=75, fontsize=8)
ax[0].set_title('(a) Size of each IMO-AG-30 problem'); ax[0].set_ylim(0, 42); ax[0].legend(loc='upper center', ncol=2)
gc = collections.Counter(goals).most_common()
ax[1].barh([g for g, _ in gc][::-1], [c for _, c in gc][::-1], color=SERIES[0])
for i, (g, c) in enumerate(gc[::-1]): ax[1].text(c + 0.1, i, str(c), va='center', color=INK2)
ax[1].set_title('(b) Goal predicate'); ax[1].set_xlabel('problems')
cc = cons.most_common(14)
ax[2].barh([g for g, _ in cc][::-1], [c for _, c in cc][::-1], color=SERIES[0])
ax[2].set_title('(c) Most frequent constructions'); ax[2].set_xlabel('uses across 30 problems')
plt.tight_layout(); plt.savefig('/workspace/report/images/fig1_data_overview.png')
print(np.mean(npts), min(npts), max(npts), np.mean(nprem), collections.Counter(goals))
