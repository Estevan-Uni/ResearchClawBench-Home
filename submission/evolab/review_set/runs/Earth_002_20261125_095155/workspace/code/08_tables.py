"""Write markdown tables used in the report."""
import json, csv, numpy as np
W='/workspace/'; o=[]
S=json.load(open(W+'outputs/summary_central.json'))
SSPL={'ssp245':'SSP2-4.5','ssp370':'SSP3-7.0','ssp585':'SSP5-8.5'}; VL={'A_MIT':'A','B_Knutson':'B'}
names=['None','Low','Moderate','High','Very high','Severe']
o.append('| Variant | SSP | Weight | '+' | '.join(f'{k} {n}' for k,n in enumerate(names))+' | **≥3** |'); o.append('|'+'---|'*10)
for v in VL:
    for s in SSPL:
        for wk,wl in [('area','area'),('pop','people'),('stock','built capital')]:
            sc=S[f'{v}|{s}'][wk]['score']
            o.append(f'| {VL[v]} | {SSPL[s]} | {wl} | '+' | '.join(f'{100*x:.1f}' for x in sc)+f' | **{100*sum(sc[3:]):.1f}** |')
o.append('\n')
o.append('| Variant | SSP | TC level 0 / 1 / 2 / 3 (% area) | SLR level 0 / 1 / 2 (% area) | TC-only | SLR(≥7)-only | Compound |'); o.append('|---|---|---|---|---|---|---|')
for v in VL:
    for s in SSPL:
        a=S[f'{v}|{s}']['area']; t=a['typ']
        o.append(f"| {VL[v]} | {SSPL[s]} | {' / '.join(f'{100*x:.1f}' for x in a['tc'])} | {' / '.join(f'{100*x:.1f}' for x in a['slr'])} | {100*t['tc_only']:.1f} | {100*t['slr_only']:.1f} | {100*t['compound']:.1f} |")
o.append('\n')
rows=list(csv.DictReader(open(W+'outputs/regional_diagnosis.csv')))
o.append('| Region | Area (Mha) | Baseline TC-affected | Variant | SSP | TC 0/1/2/3 (%) | SLR ≥7 (%) | Score ≥3 (%) | Severe (%) |'); o.append('|---|---|---|---|---|---|---|---|---|')
for r in rows:
    f=lambda k: 100*float(r[k])
    o.append(f"| {r['region']} | {float(r['area_ha'])/1e6:.2f} | {f('base_affected'):.0f}% | {VL[r['variant']]} | {SSPL[r['ssp']]} | {f('tc0'):.0f}/{f('tc1'):.0f}/{f('tc2'):.0f}/{f('tc3'):.0f} | {f('slr2'):.0f} | {f('ge3'):.0f} | {f('s5'):.0f} |")
o.append('\n')
MC=json.load(open(W+'outputs/montecarlo.json'))['res']
o.append('| Variant | SSP | central ≥3 | MC ≥3 median [5–95%] | MC Severe median [5–95%] | MC ≥3 people | MC ≥3 built capital |'); o.append('|---|---|---|---|---|---|---|')
for v in VL:
    for s in SSPL:
        m=MC[f'{v}|{s}']; p=lambda x: '%.0f [%.0f–%.0f]'%tuple(np.percentile(np.array(x)*100,[50,5,95]))
        o.append(f"| {VL[v]} | {SSPL[s]} | {100*S[f'{v}|{s}']['area']['ge3']:.1f} | {p(m['ge3'])} | {p(m['s5'])} | {p(m['ge3_pop'])} | {p(m['ge3_stock'])} |")
o.append('\n')
SE=json.load(open(W+'outputs/sensitivity.json'))
o.append('| Sensitivity | A 2-4.5 | A 3-7.0 | A 5-8.5 | B 2-4.5 | B 3-7.0 | B 5-8.5 |'); o.append('|---|---|---|---|---|---|---|')
for k,v in SE.items():
    o.append(f'| {k} | '+' | '.join(f"{100*v[f'{vv}|{s}']['ge3']:.1f}" if f'{vv}|{s}' in v else '–' for vv in VL for s in SSPL)+' |')
o.append('\n')
C=list(csv.DictReader(open(W+'outputs/country_table.csv')))[:20]
o.append('| Country | Mangrove area (kha) | ≥3 share A: 2-4.5 / 3-7.0 / 5-8.5 | ≥3 share B 3-7.0 | People at ≥3 (k, A 3-7.0) | Built capital at ≥3 (US$ M, A 3-7.0) |'); o.append('|---|---|---|---|---|---|')
for d in C:
    o.append(f"| {d['country']} | {float(d['area_ha'])/1e3:.0f} | {100*float(d['ge3_A_MIT_ssp245']):.0f} / {100*float(d['ge3_A_MIT_ssp370']):.0f} / {100*float(d['ge3_A_MIT_ssp585']):.0f} | {100*float(d['ge3_B_Knutson_ssp370']):.0f} | {float(d['pop_ge3_A_MIT_ssp370'])/1e3:.0f} | {float(d['stock_ge3_A_MIT_ssp370'])/1e6:.0f} |")
V=json.load(open(W+'outputs/validation.json'))['ref']
o.append('\n| Site | ≈Observed RP (yr) | Model RP R=100 km, basin-corrected | uncorrected | R=50 | R=200 |'); o.append('|---|---|---|---|---|---|')
for n,d in V.items():
    o.append(f"| {n} | {d['obs_rp']:.1f} | {d['rp_R100']:.1f} | {d['rp_R100_nocorr']:.1f} | {d['rp_R50']:.1f} | {d['rp_R200']:.1f} |")
open(W+'outputs/tables.md','w').write('\n'.join(o)); print('\n'.join(o))
