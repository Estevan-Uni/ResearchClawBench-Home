import sys; sys.path.insert(0, 'code')
from style import *
from matplotlib.patches import FancyBboxPatch
fig, ax = plt.subplots(figsize=(12.5, 4.4)); ax.axis('off'); ax.set_xlim(0, 12.7); ax.set_ylim(0, 4.2)
boxes = [
 (0.1, 2.4, 'Step 1  Data mining\nadhesive-protein sequences\n(raw corpus not in provided files)\n→ 6 residue classes', CAT[0]),
 (3.0, 2.4, 'Class → monomer\nnucleophilic→HEA  hydrophobic→BA\nacidic→CBEA  cationic→ATAC\naromatic→PEA  amide→AAm', CAT[0]),
 (5.9, 2.4, 'Step 2  DM-designed library\n180 random copolymer gels\n(relative compositions\nreplicating protein statistics)', CAT[2]),
 (8.8, 2.4, 'Screening\nunderwater F_a on glass (10 s)\n(+Q, phase sep., modulus,\ntanδ, steel subset)', CAT[2]),
 (8.8, 0.25, 'Step 3  ML optimisation\nGP / RFR surrogates +\nbatch EI (KB, CL, LP, ENU)\n& predicted-mean (SM) lists', CAT[6]),
 (5.9, 0.25, 'Rounds 1→3\n130 + 40 + 40 list entries\n(113 + 29 + 26 new\nunique formulations)', CAT[6]),
 (3.0, 0.25, 'Best gels\nG-max  G-042  146.6 kPa\nR1-max old-SM-GP#1\nBA/ATAC/PEA  353.3 kPa', CAT[1]),
 (0.1, 0.25, 'Design rule & checks\nhydrophobic + aromatic +\ncationic; near-ideal\nrandom copolymerization', CAT[1]),
]
W, H = 2.7, 1.5
for x, y, t, c in boxes:
    ax.add_patch(FancyBboxPatch((x, y), W, H, boxstyle='round,pad=0.02,rounding_size=0.12', fc=c + '22', ec=c, lw=1.4))
    ax.text(x + W / 2, y + H / 2, t, ha='center', va='center', fontsize=8, color=INK)
arr = dict(arrowstyle='-|>', color=INK2, lw=1.3)
for (x0, y0), (x1, y1) in [((2.85, 3.15), (3.0, 3.15)), ((5.75, 3.15), (5.9, 3.15)), ((8.65, 3.15), (8.8, 3.15)),
                           ((10.15, 2.4), (10.15, 1.75)), ((8.8, 1.0), (8.65, 1.0)), ((5.9, 1.0), (5.75, 1.0)), ((3.0, 1.0), (2.85, 1.0))]:
    ax.annotate('', xy=(x1, y1), xytext=(x0, y0), arrowprops=arr)
ax.annotate('', xy=(11.5, 1.5), xytext=(11.5, 0.5), arrowprops=dict(arrowstyle='-|>', color=CAT[6], lw=1.2, connectionstyle='arc3,rad=0.9'))
ax.text(12.0, 1.0, 'retrain\non new\ndata', fontsize=7.5, color=INK2, va='center')
ax.text(6.1, 4.15, 'Data-driven de novo design of adhesive hydrogels: workflow reconstructed from the provided files', ha='center', fontsize=10, color=INK)
save(fig, 'fig01_workflow')
