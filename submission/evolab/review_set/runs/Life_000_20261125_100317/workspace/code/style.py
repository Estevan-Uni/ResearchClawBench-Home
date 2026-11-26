import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
CAT = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948']
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e4e3df'
MON = ['HEA', 'BA', 'CBEA', 'ATAC', 'PEA', 'AAm']
MONCOL = dict(zip(MON, CAT[:6]))
plt.rcParams.update({'font.size': 9, 'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2,
                     'ytick.color': INK2, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6, 'axes.axisbelow': True,
                     'legend.frameon': False, 'savefig.dpi': 200, 'savefig.bbox': 'tight', 'figure.dpi': 100})
def save(fig, name):
    fig.savefig(f'report/images/{name}.png'); plt.close(fig); print('saved', name)
