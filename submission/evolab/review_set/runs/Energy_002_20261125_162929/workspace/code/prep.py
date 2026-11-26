"""Data loading, checks, country assignment (point-in-polygon on Natural Earth), port distances."""
import numpy as np, pandas as pd, shapefile
from shapely.geometry import shape, Point
import params as P

ROOT = '/workspace'

def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = p2 - p1; dl = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))

def load_countries(bbox=(-25, -40, 55, 40)):
    sf = shapefile.Reader(f'{ROOT}/data/africa_map/ne_10m_admin_0_countries.shp', encoding='latin-1')
    flds = [f[0] for f in sf.fields[1:]]
    out = []
    for sr in sf.iterShapeRecords():
        rec = dict(zip(flds, sr.record))
        b = sr.shape.bbox
        if b[2] < bbox[0] or b[0] > bbox[2] or b[3] < bbox[1] or b[1] > bbox[3]:
            continue
        out.append(dict(iso=rec.get('ISO_A2_EH', rec.get('ISO_A2')), name=rec['NAME'], cont=rec.get('CONTINENT'),
                        geom=shape(sr.shape.__geo_interface__)))
    return out

def load_hex():
    d = pd.read_csv(f'{ROOT}/data/hex_final_NA_min.csv')
    assert len(d) == 30 and d.isna().sum().sum() == 0
    assert d.theo_pv.between(0, 1).all() and d.theo_wind.between(0, 1).all()
    for c in ['grid_dist_km', 'road_dist_km', 'ocean_dist_km', 'waterbody_dist_km']:
        assert (d[c] >= 0).all()
    cs = load_countries()
    iso, name, offshore_km = [], [], []
    for _, r in d.iterrows():
        pt = Point(r.lon, r.lat)
        hit = [c for c in cs if c['geom'].contains(pt)]
        if hit:
            iso.append(hit[0]['iso']); name.append(hit[0]['name']); offshore_km.append(0.0)
        else:
            dists = [(c['geom'].distance(pt), c) for c in cs]
            dmin, cn = min(dists, key=lambda x: x[0])
            iso.append('SEA'); name.append(f"Atlantic (near {cn['name']})"); offshore_km.append(dmin * 111.0 * np.cos(np.radians(r.lat)))
    d['ne_iso'] = iso; d['ne_name'] = name; d['offshore_km'] = np.round(offshore_km, 1)
    d['onshore'] = d.ne_iso != 'SEA'
    d['cf_pv'] = P.pv_cf(d.theo_pv); d['cf_wind'] = P.wind_cf(d.theo_wind)
    for p, v in P.PORTS.items():
        d[f'km_{p}'] = haversine(d.lat, d.lon, v['lat'], v['lon']) * P.DETOUR
    return d, cs

if __name__ == '__main__':
    d, cs = load_hex()
    d.to_csv(f'{ROOT}/outputs/hex_prepared.csv', index=False)
    print(d[['hex_id', 'lat', 'lon', 'ne_iso', 'ne_name', 'offshore_km', 'cf_pv', 'cf_wind']].to_string())
    print(d.ne_iso.value_counts())
    print(d.drop(columns=['hex_id', 'ne_iso', 'ne_name']).corr(numeric_only=True).round(2).to_string())
