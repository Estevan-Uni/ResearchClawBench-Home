import json
import numpy as np
from plotstyle import plt, C, style
style()
models = [('sr', 'SR (no charge info)'), ('les', 'LES (no charge info)'), ('sr_q', 'SR + charge embedding'),
          ('les_q', 'LES + charge embedding'), ('sep', 'separate SR models')]
rows = []
for data in ['orig', 'mod']:
    for m, lab in models:
        try:
            r = json.load(open(f'../outputs/results/ag3_{data}_{m}_s0.json'))
        except FileNotFoundError:
            continue
        rows.append(dict(data=data, model=lab, E_p=r['Q+1']['E_mae_atom'] * 1e3, F_p=r['Q+1']['F_mae'] * 1e3,
                         E_m=r['Q-1']['E_mae_atom'] * 1e3, F_m=r['Q-1']['F_mae'] * 1e3))
for r in rows: print(r)
json.dump(rows, open('../outputs/results/summary_ag3.json', 'w'), indent=1)
fig, ax = plt.subplots(1, 3, figsize=(14, 3.9))
cols = {'sr': C['orange'], 'les': C['magenta'], 'sr_q': C['blue'], 'les_q': C['aqua'], 'sep': C['violet']}
lss = {'sr': '--', 'les': '-.', 'sr_q': '-', 'les_q': ':', 'sep': (0, (1, 1))}
for a_, data, title in [(ax[0], 'orig', '(a) dataset as provided'), (ax[1], 'mod', '(b) controlled variant (distinct anion PES)')]:
    ref = np.load(f'../outputs/preds/ag3_{data}_sr_q_s0.npz')
    a_.plot(ref['ds'], ref['ref_p'], color=C['ink'], lw=3, label='reference Ag$_3^+$')
    a_.plot(ref['ds'], ref['ref_m'], color=C['ink2'], lw=3, alpha=0.5, label='reference Ag$_3^-$')
    for m, lab in models:
        try:
            d = np.load(f'../outputs/preds/ag3_{data}_{m}_s0.npz')
        except FileNotFoundError:
            continue
        a_.plot(d['ds'], d['scan_p'], color=cols[m], ls=lss[m], lw=1.6, label=lab)
        a_.plot(d['ds'], d['scan_m'], color=cols[m], ls=lss[m], lw=1.6)
    a_.set_ylim(0.2, 2.5); a_.set_xlabel('Ag–Ag distance, equilateral Ag$_3$ (Å)'); a_.set_ylabel('energy (eV)')
    a_.set_title(title)
ax[0].legend(fontsize=7, loc='upper right')
# (c) bar chart of energy MAE on the controlled variant
rm = [r for r in rows if r['data'] == 'mod']
x = np.arange(len(rm)); w = 0.38
ax[2].bar(x - w / 2 - 0.01, [r['E_p'] for r in rm], w, color=C['blue'], label='Ag$_3^+$')
ax[2].bar(x + w / 2 + 0.01, [r['E_m'] for r in rm], w, color=C['orange'], label='Ag$_3^-$')
ax[2].set_yscale('log'); ax[2].set_xticks(x); short = {'SR (no charge info)': 'SR', 'LES (no charge info)': 'LES', 'SR + charge embedding': 'SR+Q', 'LES + charge embedding': 'LES+Q', 'separate SR models': 'separate'}
ax[2].set_xticklabels([short[r['model']] for r in rm], fontsize=9)
ax[2].set_ylabel('test energy MAE (meV/atom)'); ax[2].set_title('(c) errors on controlled variant'); ax[2].legend(fontsize=8)
plt.tight_layout(); plt.savefig('../report/images/fig4_ag3.png')
