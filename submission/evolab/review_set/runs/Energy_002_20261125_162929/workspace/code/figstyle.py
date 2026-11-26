import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np, shapefile
from shapely.geometry import shape

CAT = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948']
INK = '#0b0b0b'; INK2 = '#52514e'; MUTED = '#898781'; GRID = '#e1e0d9'; AXIS = '#c3c2b7'; SURF = '#fcfcfb'
SEQ = LinearSegmentedColormap.from_list('seqblue', ['#cde2fb', '#86b6ef', '#3987e5', '#1c5cab', '#0d366b'])
SCEN_COL = {'Commercial': CAT[1], 'Offtake': CAT[3], 'Blended': CAT[2], 'Full de-risking': CAT[0]}
EU_COL = {'Germany (north)': CAT[6], 'Spain (+2,000 km pipe)': CAT[4]}

plt.rcParams.update({
    'figure.facecolor': SURF, 'axes.facecolor': SURF, 'savefig.facecolor': SURF,
    'axes.edgecolor': AXIS, 'axes.labelcolor': INK2, 'xtick.color': INK2, 'ytick.color': INK2,
    'text.color': INK, 'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6,
    'axes.spines.top': False, 'axes.spines.right': False, 'font.size': 9, 'axes.titlesize': 10,
    'axes.titleweight': 'bold', 'legend.frameon': False, 'lines.linewidth': 2, 'savefig.dpi': 160,
})

_BASE = None
def basemap(bbox=(8, -36, 32, -12)):
    global _BASE
    if _BASE is None:
        sf = shapefile.Reader('/workspace/data/africa_map/ne_10m_admin_0_countries.shp', encoding='latin-1')
        flds = [f[0] for f in sf.fields[1:]]
        _BASE = []
        for sr in sf.iterShapeRecords():
            b = sr.shape.bbox
            if b[2] < bbox[0] or b[0] > bbox[2] or b[3] < bbox[1] or b[1] > bbox[3]: continue
            rec = dict(zip(flds, sr.record))
            _BASE.append((rec['NAME'], shape(sr.shape.__geo_interface__)))
    return _BASE

def draw_base(ax, bbox=(10, -30, 26, -16), labels=True):
    for name, g in basemap():
        geoms = getattr(g, 'geoms', [g])
        for p in geoms:
            x, y = p.exterior.xy
            ax.fill(x, y, color='#f0efec', ec='#a8a69e', lw=0.6, zorder=0)
        if labels and name in ('Namibia', 'Botswana', 'South Africa', 'Angola', 'Zambia'):
            c = g.representative_point()
            lx, ly = {'Namibia': (17.0, -21.0), 'Botswana': (24.0, -21.5), 'South Africa': (21.5, -29.4),
                      'Angola': (16.5, -16.4), 'Zambia': (24.5, -16.4)}.get(name, (c.x, c.y))
            if bbox[0] < lx < bbox[2] and bbox[1] < ly < bbox[3]:
                ax.text(lx, ly, name, color=MUTED, fontsize=7.5, ha='center', style='italic', zorder=1)
    ax.set_xlim(bbox[0], bbox[2]); ax.set_ylim(bbox[1], bbox[3])
    ax.set_aspect(1 / np.cos(np.radians(np.mean(bbox[1::2]))))
    ax.grid(False); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_visible(False)
    ax.set_facecolor('#e9f1fb')

def save(fig, name):
    fig.savefig(f'/workspace/report/images/{name}.png', bbox_inches='tight')
    plt.close(fig)
