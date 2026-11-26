"""Merge Foldseek-Multimer and US-align results; accuracy, thresholds, assignment consistency; figures."""
import json, glob, os, sys, numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
sys.path.insert(0,'code'); from parse_foldseek import load_m8, best_assignments, rot_angle
import pdbutil as P
plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':'#52514e','axes.labelcolor':'#0b0b0b',
                     'xtick.color':'#52514e','ytick.color':'#52514e','figure.facecolor':'#fcfcfb','axes.facecolor':'#fcfcfb','savefig.facecolor':'#fcfcfb','axes.grid':True,'axes.axisbelow':True,'grid.color':'#e6e5e0','grid.linewidth':0.6})
TIERC={'T1_same_system':'#2a78d6','T2_IVA1_ortholog':'#eb6834','T3_IVA3_remote':'#1baf7a','T3b_IVC':'#eda100','T4_typeI_III':'#e87ba4','N_unrelated':'#6250d6','self':'#52514e'}
TIERL={'T1_same_system':'T1 same system (P. aeruginosa IV-A1)','T2_IVA1_ortholog':'T2 IV-A1 ortholog (P. oleovorans)','T3_IVA3_remote':'T3 IV-A3 remote (K. pneumoniae)',
       'T3b_IVC':'T3b type IV-C (P. abyssi)','T4_typeI_III':'T4 type I / III effectors','N_unrelated':'N unrelated controls','self':'self (7XG4)'}
IMG='report/images'; os.makedirs(IMG,exist_ok=True)
meta=json.load(open('outputs/db_meta.json'))
tier=lambda t: meta[t]['tier'] if t in meta else 'self'
CH=json.load(open('outputs/chain_tables.json'))['tables']
def fs_table(fn):
    best=best_assignments(load_m8(fn)); d={}
    for tc,rs in best.items():
        d[tc]=dict(qtm=rs[0]['complexqtmscore'],ttm=rs[0]['complexttmscore'],npairs=len(rs),pairs={r['qch']:r['tch'] for r in rs},cU=rs[0]['cU'],cT=rs[0]['cT'])
    return d
EXH=fs_table('outputs/search/exh.m8'); DEF=fs_table('outputs/search/def.m8')
US={}
for f in glob.glob('outputs/usalign/7XG4_vs_*.json'):
    r=json.load(open(f)); US[r['target']]=r
rows=[]
for t in ['7XG4']+list(meta):
    e=EXH.get(t); d=DEF.get(t); u=US.get(t)
    row=dict(target=t,tier=tier(t),nchain=meta[t]['nchain'] if t in meta else 9,nres=meta[t]['nres'] if t in meta else 2876,
             fs_qtm=e['qtm'] if e else 0.0, fs_ttm=e['ttm'] if e else 0.0, fs_npairs=e['npairs'] if e else 0,
             fsdef_qtm=d['qtm'] if d else 0.0, fsdef_found=bool(d),
             fident_mean=CH[t]['fident_mean'] if t in CH else np.nan, fident_min=CH[t]['fident_min'] if t in CH else np.nan, fident_max=CH[t]['fident_max'] if t in CH else np.nan)
    if u:
        up=dict(u['pairs']); row.update(us_qtm=u['qtm'],us_ttm=u['ttm'],us_npairs=len(up),us_time=u['time'],us_seqid=u['seqid'],us_rmsd=u['rmsd'])
        if e:
            shared=[q for q in e['pairs'] if q in up]
            row['assign_agree']=sum(e['pairs'][q]==up[q] for q in shared)/max(1,len(shared))
            row['n_shared']=len(shared); row['coverage_vs_us']=len(e['pairs'])/len(up)
            # rotation angle between complex superpositions (Foldseek target->query inverted to query->target)
            Ufs=e['cU'].T; Uus=np.array(u['u']); row['dAngle_complex_vs_us']=rot_angle(Ufs,Uus)
            # RMS displacement of query CAs under the two transforms
            X=np.vstack([v[1] for v in P.ca_chains('outputs/structs/7XG4.pdb').values()])
            A=(X-e['cT'])@e['cU']; B=X@Uus.T+np.array(u['t']); row['disp_complex_vs_us']=float(np.sqrt(((A-B)**2).sum(1).mean()))
    rows.append(row)
df=pd.DataFrame(rows); df.to_csv('outputs/summary_targets.csv',index=False)
nonself=df[df.target!='7XG4']
m=nonself.dropna(subset=['us_qtm'])
acc=dict(n=len(m),pearson=pearsonr(m.fs_qtm,m.us_qtm)[0],spearman=spearmanr(m.fs_qtm,m.us_qtm)[0],mae=float((m.fs_qtm-m.us_qtm).abs().mean()),
         bias=float((m.fs_qtm-m.us_qtm).mean()),pearson_ttm=pearsonr(m.fs_ttm,m.us_ttm)[0],spearman_ttm=spearmanr(m.fs_ttm,m.us_ttm)[0],mae_ttm=float((m.fs_ttm-m.us_ttm).abs().mean()))
# thresholds
thr=[]
for c in [0.3,0.5,0.65,0.8]:
    for name,col in [('Foldseek-Multimer exhaustive','fs_qtm'),('Foldseek-Multimer default','fsdef_qtm'),('US-align','us_qtm')]:
        sub=m if col=='us_qtm' else nonself
        hit=sub[sub[col]>=c]
        thr.append(dict(cutoff=c,method=name,n_hits=len(hit),tiers=dict(hit.tier.value_counts()),
                        IVA_hits=int(hit.tier.isin(['T1_same_system','T2_IVA1_ortholog','T3_IVA3_remote']).sum()),
                        nonIVA_hits=int((~hit.tier.isin(['T1_same_system','T2_IVA1_ortholog','T3_IVA3_remote'])).sum())))
json.dump(dict(accuracy=acc,thresholds=thr),open('outputs/accuracy_thresholds.json','w'),indent=1,default=str)
print(json.dumps(acc,indent=1)); print(pd.DataFrame(thr)[['cutoff','method','n_hits','IVA_hits','nonIVA_hits']].to_string())
print(df[['target','tier','fs_qtm','fs_ttm','fsdef_qtm','us_qtm','us_ttm','fs_npairs','us_npairs','assign_agree','coverage_vs_us','dAngle_complex_vs_us','disp_complex_vs_us','fident_mean','fident_min','fident_max']].round(3).to_string())

# ---------- Fig 1: data overview
q=P.ca_chains('outputs/structs/7XG4.pdb'); names={'A':'Csf1 (Cas8-like)','B':'Csf3 (Cas5)','C':'Csf2.1','D':'Csf2.2','E':'Csf2.3','F':'Csf2.4','G':'Csf2.5','H':'Csf5 (Cas6)','L':'Csf4 / CasDinG'}
fig,ax=plt.subplots(1,3,figsize=(13,5),gridspec_kw={'width_ratios':[1.3,0.35,1.6]})
ks=list(q); ax[0].bar(range(len(ks)),[len(q[k][0]) for k in ks],color='#2a78d6',width=0.7)
ax[0].set_xticks(range(len(ks))); ax[0].set_xticklabels([f'{k}\n{names[k]}' for k in ks],fontsize=6.5,rotation=40,ha='right')
ax[0].set_ylabel('Cα residues'); ax[0].set_title('7XG4 query: 9 protein chains, 2,876 residues',fontsize=9,loc='left')
ax[1].bar([0],[726],color='#6250d6',width=0.5); ax[1].set_xticks([0]); ax[1].set_xticklabels(['A\nMmpL3'],fontsize=7); ax[1].set_xlim(-0.8,0.8)
ax[1].set_title('6N40: 1 chain',fontsize=9,loc='left')
order=['self','T1_same_system','T2_IVA1_ortholog','T3_IVA3_remote','T3b_IVC','T4_typeI_III','N_unrelated']
d2=df.assign(o=df.tier.map(order.index)).sort_values(['o','target'],ascending=[False,False]).reset_index(drop=True)
ax[2].barh(range(len(d2)),d2.nres,color=[TIERC[t] for t in d2.tier],height=0.75)
for i,r in d2.iterrows(): ax[2].text(r.nres+40,i,f'{r.nchain} ch',va='center',fontsize=6,color='#52514e')
ax[2].set_yticks(range(len(d2))); ax[2].set_yticklabels(d2.target,fontsize=6.5); ax[2].set_xlabel('protein residues (Cα)')
ax[2].set_title('Benchmark database (25 entries + query)',fontsize=9,loc='left'); ax[2].grid(axis='y',visible=False)
from matplotlib.patches import Patch
ax[2].legend(handles=[Patch(color=TIERC[t],label=TIERL[t]) for t in order],fontsize=6.5,frameon=False,loc='lower right')
plt.tight_layout(); plt.savefig(f'{IMG}/fig1_data_overview.png',dpi=200); plt.close()

# ---------- Fig 2: requested pair 7XG4 vs 6N40
pt=pd.read_csv('outputs/pair/7XG4_vs_6N40.m8',sep='\t',header=None,names=open('outputs/pair/fmt.txt').read().strip().split(','))
pt['qch']=pt['query'].str.split('_').str[1]
fig,ax=plt.subplots(1,2,figsize=(10,3.3))
x=np.arange(len(pt)); w=0.38
ax[0].bar(x-w/2,pt.qtmscore,w,color='#2a78d6',label='chain TM (norm. by query chain)')
ax[0].bar(x+w/2,pt.complexqtmscore,w,color='#eb6834',label='complex qTM (norm. by 2,876 res)')
ax[0].axhline(0.5,color='#52514e',ls='--',lw=0.8); ax[0].text(len(pt)-0.5,0.51,'TM = 0.5',fontsize=7,ha='right',color='#52514e')
ax[0].set_xticks(x); ax[0].set_xticklabels(pt.qch); ax[0].set_ylim(0,0.6); ax[0].set_xlabel('7XG4 chain aligned to 6N40 chain A'); ax[0].set_ylabel('TM-score')
ax[0].legend(fontsize=7,frameon=False); ax[0].set_title('Foldseek-Multimer: 9 one-chain assignments, none significant',fontsize=9,loc='left')
u=US['6N40'] if '6N40' in US else None
vals=[('FS-M best\ncomplex qTM',pt.complexqtmscore.max()),('FS-M best\ncomplex tTM',pt.complexttmscore.max()),('US-align\nqTM',u['qtm'] if u else np.nan),('US-align\ntTM',u['ttm'] if u else np.nan),('9N3Y FS-M\nqTM (ref.)',EXH['9N3Y']['qtm'])]
ax[1].bar(range(len(vals)),[v for _,v in vals],color=['#eb6834','#eb6834','#1baf7a','#1baf7a','#2a78d6'],width=0.6)
for i,(_,v) in enumerate(vals): ax[1].text(i,v+0.02,f'{v:.3f}',ha='center',fontsize=7)
ax[1].set_xticks(range(len(vals))); ax[1].set_xticklabels([k for k,_ in vals],fontsize=7); ax[1].set_ylim(0,1.1)
ax[1].set_title('7XG4 → 6N40 complex scores vs a true homolog',fontsize=9,loc='left')
plt.tight_layout(); plt.savefig(f'{IMG}/fig2_requested_pair.png',dpi=200); plt.close()

# ---------- Fig 3: Csf2 swap matrices
S=json.load(open('outputs/csf2_swap.json'))
fig,ax=plt.subplots(3,3,figsize=(10,9))
for i,tc in enumerate(['9N3Y','8RFJ','8S35']):
    s=S[tc]
    for j,(k,cm,lab) in enumerate([('chainTM','Blues','chain TM (query-norm.)'),('angle','Oranges','rotation vs complex U (°)'),('disp','Oranges','RMS displacement vs complex (Å)')]):
        M=np.array(s[k],dtype=float); a=ax[i,j]; im=a.imshow(M,cmap=cm,vmin=(0.4 if k=='chainTM' else 0),vmax=(1 if k=='chainTM' else None))
        for r in range(5):
            for c in range(5): a.text(c,r,f'{M[r,c]:.2f}' if k=='chainTM' else f'{M[r,c]:.0f}',ha='center',va='center',fontsize=7,color='white' if (M[r,c]>(0.8 if k=='chainTM' else np.nanmax(M)*0.6)) else '#0b0b0b')
        a.set_xticks(range(5)); a.set_xticklabels([f'{tc}_{x}' for x in s['tchains']],fontsize=6.5,rotation=30); a.set_yticks(range(5)); a.set_yticklabels([f'7XG4_{x}' for x in s['qchains']],fontsize=6.5)
        a.grid(False); a.set_title(f'{tc}: {lab}',fontsize=8.5,loc='left'); plt.colorbar(im,ax=a,fraction=0.046)
plt.tight_layout(); plt.savefig(f'{IMG}/fig3_csf2_homomer_swap.png',dpi=200); plt.close()

# ---------- Fig 4: superposition consistency scatter, all chain pairs
A=json.load(open('outputs/chain_tables.json'))['all_chain_pairs']
fig,ax=plt.subplots(1,3,figsize=(12,3.6),sharey=True)
for i,tc in enumerate(['9N3Y','8RFJ','8S35']):
    l=pd.DataFrame(A[tc]); a=ax[i]
    w_=l[~l.correct]; c_=l[l.correct]
    a.scatter(w_.dAngle_deg,w_.dispRMS,s=14,c=w_.chainTM_q,cmap='Greys',vmin=0,vmax=1.2,edgecolor='#52514e',linewidth=0.3,label='non-assigned chain pairs (colour = chain TM)')
    a.scatter(c_.dAngle_deg,c_.dispRMS,s=40,color='#eb6834',edgecolor='#fcfcfb',linewidth=1.5,label='pairs in Foldseek-Multimer assignment',zorder=3)
    a.set_yscale('symlog',linthresh=5); a.set_xlabel('rotation angle to complex U (°)'); a.set_title(f'7XG4 vs {tc} ({TIERL[tier(tc)].split(" (")[0]})',fontsize=8.5,loc='left')
ax[0].set_ylabel('RMS displacement of query Cα (Å)'); ax[0].legend(fontsize=6.5,frameon=False,loc='lower right')
plt.tight_layout(); plt.savefig(f'{IMG}/fig4_superposition_consistency.png',dpi=200); plt.close()

# ---------- Fig 5: FS-M vs US-align
fig,ax=plt.subplots(1,2,figsize=(10,4.2))
for k,(col,ucol,lab) in enumerate([('fs_qtm','us_qtm','complex TM normalised by query (7XG4)'),('fs_ttm','us_ttm','complex TM normalised by target')]):
    a=ax[k]
    for tr,g in m.groupby('tier'):
        a.scatter(g[ucol],g[col],s=42,color=TIERC[tr],label=TIERL[tr],edgecolor='#fcfcfb',linewidth=1.5,zorder=3)
    a.plot([0,1],[0,1],color='#52514e',lw=0.8,ls='--'); a.set_xlim(-0.02,1.02); a.set_ylim(-0.02,1.02)
    for c in [0.3,0.5,0.65]: a.axhline(c,color='#c3c2b7',lw=0.6); a.axvline(c,color='#c3c2b7',lw=0.6)
    a.set_xlabel(f'US-align -mm 1: {lab}',fontsize=8); a.set_ylabel(f'Foldseek-Multimer: {lab}',fontsize=8)
    r=pearsonr(m[col],m[ucol])[0]; rho=spearmanr(m[col],m[ucol])[0]
    a.text(0.02,0.96,f'n={len(m)}  Pearson r={r:.3f}  Spearman ρ={rho:.3f}\nMAE={(m[col]-m[ucol]).abs().mean():.3f}',fontsize=7.5,va='top')
    for _,rr in m.iterrows():
        if rr.tier in ('T3_IVA3_remote','T3b_IVC','T4_typeI_III') or rr.target in ('6N40','7XFZ','7XG1'): a.annotate(rr.target,(rr[ucol],rr[col]),fontsize=6,xytext=(3,-6),textcoords='offset points',color='#52514e')
ax[0].legend(fontsize=6.5,frameon=False,loc='lower right')
plt.tight_layout(); plt.savefig(f'{IMG}/fig5_fsm_vs_usalign.png',dpi=200); plt.close()

# ---------- Fig 6: twilight zone - fident vs qTM
fig,ax=plt.subplots(figsize=(6,4))
g=df[(df.target!='7XG4')&(df.fs_npairs>=2)]
for tr,gg in g.groupby('tier'):
    ax.errorbar(gg.fident_mean,gg.fs_qtm,xerr=[gg.fident_mean-gg.fident_min,gg.fident_max-gg.fident_mean],fmt='o',color=TIERC[tr],label=TIERL[tr],ms=6,elinewidth=1,capsize=2)
ax.axvline(0.25,color='#52514e',ls='--',lw=0.8); ax.text(0.26,0.05,'25 % identity\n(twilight zone)',fontsize=7,color='#52514e')
ax.set_xlabel('sequence identity of aligned chain pairs (mean, min–max)'); ax.set_ylabel('Foldseek-Multimer complex qTM')
ax.legend(fontsize=6.5,frameon=False,loc='lower right'); plt.tight_layout(); plt.savefig(f'{IMG}/fig6_twilight.png',dpi=200); plt.close()
print('figures done')
