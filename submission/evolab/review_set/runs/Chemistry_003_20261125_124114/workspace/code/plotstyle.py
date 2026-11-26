import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

C = dict(blue='#2a78d6', orange='#eb6834', aqua='#1baf7a', yellow='#eda100', magenta='#e87ba4',
         green='#008300', violet='#6250d6', red='#e34948', ink='#0b0b0b', ink2='#52514e', muted='#8a8984',
         grid='#e4e3df', surface='#fcfcfb')
SERIES = [C['blue'], C['orange'], C['aqua'], C['yellow'], C['magenta'], C['green'], C['violet'], C['red']]


def style():
    plt.rcParams.update({
        'figure.facecolor': C['surface'], 'axes.facecolor': C['surface'], 'savefig.facecolor': C['surface'],
        'axes.edgecolor': C['muted'], 'axes.labelcolor': C['ink'], 'xtick.color': C['ink2'], 'ytick.color': C['ink2'],
        'text.color': C['ink'], 'axes.grid': True, 'grid.color': C['grid'], 'grid.linewidth': 0.8,
        'axes.spines.top': False, 'axes.spines.right': False, 'font.size': 10, 'axes.titlesize': 11,
        'axes.titleweight': 'bold', 'legend.frameon': False, 'lines.linewidth': 2, 'figure.dpi': 150, 'axes.axisbelow': True,
    })
