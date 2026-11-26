"""Load GMW sample points, CWON country polygons, AR6 SLR rates and MIT TC track points.
Outputs: outputs/sites.npz, outputs/tc_points.npz, outputs/cwon.json"""
import sqlite3, struct, json, numpy as np, h5py, shapely
from shapely import STRtree
W='/workspace/'
def gpkg_geoms(path, table, cols):
    con=sqlite3.connect(f'file:{path}?immutable=1',uri=True)
    rows=con.execute(f'select geom,{",".join(cols)} from "{table}"').fetchall()
    geoms=[]
    for r in rows:
        b=r[0]; flags=b[3]; env=(flags>>1)&7
        n={0:0,1:32,2:48,3:48,4:64}[env]
        geoms.append(shapely.from_wkb(bytes(b[8+n:])))
    return geoms,[r[1:] for r in rows]
def tables(path):
    con=sqlite3.connect(f'file:{path}?immutable=1',uri=True)
    return con.execute("select table_name from gpkg_contents").fetchall()
if __name__=='__main__':
    mp=W+'data/mangroves/gmw_v4_ref_smpls_qad_v12.gpkg'; cp=W+'data/ecosystem/UCSC_CWON_countrybounds.gpkg'
    print(tables(mp),tables(cp))
    con=sqlite3.connect(f'file:{cp}?immutable=1',uri=True)
    t=tables(cp)[0][0]; cols=[c[1] for c in con.execute(f'pragma table_info("{t}")')]
    print(cols)
    tm=tables(mp)[0][0]
    g,a=gpkg_geoms(mp,tm,['ref_cls'])
    lon=np.array([p.x for p in g]); lat=np.array([p.y for p in g]); cls=np.array([x[0] for x in a])
    print('n pts',len(g),np.unique(cls,return_counts=True), lon.min(),lon.max(),lat.min(),lat.max())
    keep=['Mang_Ha_2020','Ben_Pop_2020','Ben_Stock_2020']
    namecol=[c for c in cols if c.lower() in('country','name','iso3','iso','admin','cntry_name','gid_0')]
    print('namecols',namecol)
    cg,ca=gpkg_geoms(cp,t,[c for c in cols if c not in('geom','fid')])
    cnames=[c for c in cols if c not in('geom','fid')]
    recs=[dict(zip(cnames,r)) for r in ca]
    print(recs[0])
    tree=STRtree(cg); pts=shapely.points(lon,lat)
    pi,ci=tree.query(pts,predicate='intersects')
    cidx=np.full(len(pts),-1); cidx[pi]=ci
    miss=np.where(cidx<0)[0]
    ni,dist=tree.query_nearest(pts[miss],return_distance=True,all_matches=False)
    md=dist; cidx[miss[ni[0]]]=np.where(md<=0.5,ni[1],-1)
    print('unassigned after snap',(cidx<0).sum(),'snapped',(md<=0.5).sum())
    np.savez(W+'outputs/sites.npz',lon=lon,lat=lat,cls=cls,cidx=cidx)
    json.dump(recs,open(W+'outputs/cwon.json','w'),default=str)
    # country polygons simplified for plotting
    simp=[shapely.to_wkt(shapely.simplify(x,0.05)) for x in cg]
    json.dump(simp,open(W+'outputs/cwon_geom_wkt.json','w'))
    # TC
    f=h5py.File(W+'data/tc/tracks_mit_mpi-esm1-2-hr_historical_reduced.nc','r')
    print({k:(f[k].shape,dict(f[k].attrs)) for k in f.keys()})
    print(dict(f.attrs))
    # SLR
    s=h5py.File(W+'data/slr/total_ssp245_medium_confidence_rates.nc','r')
    print({k:(s[k].shape,{a:s[k].attrs[a] for a in s[k].attrs if a!='DIMENSION_LIST' and a!='REFERENCE_LIST'}) for k in s.keys()})
