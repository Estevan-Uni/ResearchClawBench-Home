import json, numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
exec(open('code/summarize.py').read().split("IMG='report/images'")[0].split('sys.path.insert')[0])
plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'#fcfcfb','axes.facecolor':'#fcfcfb','savefig.facecolor':'#fcfcfb','axes.grid':True,'axes.axisbelow':True,'grid.color':'#e6e5e0'})
T=json.load(open('outputs/timing/timing.json'))
fig,ax=plt.subplots(1,2,figsize=(11,3.8),gridspec_kw={'width_ratios':[1.4,1]})
labs=[]; cols=[('fs_easy','Foldseek-Multimer (PDB in, end-to-end)','#2a78d6'),('fs_db','Foldseek-Multimer (pre-built DBs)','#1baf7a'),('us','US-align -mm 1 -ter 0','#eb6834')]
pairs=list(T['pairs']); w=0.26
for j,(k,lab,c) in enumerate(cols):
    med=[T['pairs'][p][k+'_median'] for p in pairs]; lo=[min(T['pairs'][p][k]) for p in pairs]; hi=[max(T['pairs'][p][k]) for p in pairs]
    x=np.arange(len(pairs))+(j-1)*w
    ax[0].bar(x,med,w*0.92,color=c,label=lab); ax[0].errorbar(x,med,yerr=[np.subtract(med,lo),np.subtract(hi,med)],fmt='none',ecolor='#0b0b0b',elinewidth=0.8,capsize=2)
    for xi,m in zip(x,med): ax[0].text(xi,m*1.15,f'{m:.2f}' if m<10 else f'{m:.0f}',ha='center',fontsize=6.5)
ax[0].set_yscale('log'); ax[0].set_xticks(range(len(pairs))); ax[0].set_xticklabels([p+('\n(9 vs 9 chains)' if '6N40' not in p else '\n(9 vs 1 chain)') for p in pairs],fontsize=8)
ax[0].set_ylabel('wall time per pair (s, log)'); ax[0].legend(fontsize=7,frameon=False,loc='upper right'); ax[0].set_ylim(0.02,400)
ax[0].set_title('Isolated pairwise comparison, 1 thread (median, min–max; FS n=5, US n=3)',fontsize=8.5,loc='left')
A=T['amortised']; ks=list(A)
ax[1].barh(range(len(ks)),[A[k]['per_entry'] for k in ks],color=['#2a78d6','#6250d6','#6250d6'],height=0.6)
for i,k in enumerate(ks): ax[1].text(A[k]['per_entry']+0.003,i,f"{A[k]['per_entry']*1000:.0f} ms/entry, {A[k]['n_hit_complexes']}/{A[k]['n_entries']} entries reported",va='center',fontsize=7)
ax[1].set_yticks(range(len(ks))); ax[1].set_yticklabels(['exhaustive','prefilter (default\n--max-seqs 300)','prefilter\n--max-seqs 3000'],fontsize=8)
ax[1].set_xlim(0,0.3); ax[1].set_xlabel('s per database entry (createdb excluded)'); ax[1].set_title('7XG4 vs 192-entry redundant DB (8× replicate)',fontsize=8.5,loc='left')
plt.tight_layout(); plt.savefig('report/images/fig7_timing.png',dpi=200)
