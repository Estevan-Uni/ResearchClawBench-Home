import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
BEH_COL = {'Attack': '#2a78d6', 'Sniffing': '#eb6834'}
CAT_ORDER = ['distance', 'movement', 'shape', 'angle', 'tracking', 'position', 'roi', 'index_control']
CAT_COL = {'distance': '#2a78d6', 'movement': '#eb6834', 'shape': '#1baf7a', 'angle': '#eda100', 'tracking': '#e87ba4',
           'position': '#6250d6', 'roi': '#8a8984', 'index_control': '#e34948'}
ROLE_COL = {'M1': '#2a78d6', 'M2': '#eb6834', 'inter': '#1baf7a', 'none': '#8a8984'}
REC_COL = {'Together_1': '#2a78d6', 'Reference': '#eb6834'}
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e4e3df'
plt.rcParams.update({'axes.spines.top': False, 'axes.spines.right': False, 'axes.edgecolor': INK2, 'axes.labelcolor': INK,
                     'xtick.color': INK2, 'ytick.color': INK2, 'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6,
                     'axes.axisbelow': True, 'font.size': 9, 'axes.titlesize': 10, 'legend.frameon': False, 'lines.linewidth': 2,
                     'figure.facecolor': '#fcfcfb', 'axes.facecolor': '#fcfcfb', 'savefig.facecolor': '#fcfcfb'})
