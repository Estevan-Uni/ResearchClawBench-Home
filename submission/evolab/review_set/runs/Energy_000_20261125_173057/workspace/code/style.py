"""shared plotting style (validated categorical palette, recessive axes)"""
import matplotlib.pyplot as plt
C = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948']
SEQ = ['#9ec5f0', '#5f9fe6', '#2a78d6', '#164f94']   # single-hue light->dark (blue)
plt.rcParams.update({'font.size': 10, 'axes.titlesize': 11, 'axes.edgecolor': '#8a8984', 'axes.labelcolor': '#0b0b0b',
                     'xtick.color': '#52514e', 'ytick.color': '#52514e', 'figure.facecolor': '#fcfcfb', 'axes.facecolor': '#fcfcfb',
                     'savefig.facecolor': '#fcfcfb'})
def clean(a):
    a.spines['top'].set_visible(False); a.spines['right'].set_visible(False)
    a.grid(True, color='#e6e5e0', lw=0.6); a.set_axisbelow(True)
