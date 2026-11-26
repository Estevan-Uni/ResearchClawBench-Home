import json, urllib.request, os, sys, gzip
sys.path.insert(0, os.path.dirname(__file__)); import pdbutil as P
TIERS = {
 'T1_same_system': ['7XG0','7XG1','7XG2','7XG3','7XFZ','9N3Y','9N3Z'],
 'T2_IVA1_ortholog': ['8RC3','8RFJ'],
 'T3_IVA3_remote': ['8RC2','8S35','8S36','8S37'],
 'T3b_IVC': ['11HC'],
 'T4_typeI_III': ['4TVX','5H9F','6B44','7UZW','8ANE','6IG0'],
 'N_unrelated': ['4HHB','1A0R','1FIN','6N40'],
}
out='outputs/raw'; sd='outputs/structs'; meta={}
for tier, ids in TIERS.items():
    for i in ids:
        try:
            e=json.load(urllib.request.urlopen(f"https://data.rcsb.org/rest/v1/core/entry/{i}"))
            fn=f"{out}/{i}.cif.gz"
            if not os.path.exists(fn): urllib.request.urlretrieve(f"https://files.rcsb.org/download/{i}.cif.gz", fn)
            if i=='6N40':
                atoms=P.read_atoms('data/6n40.pdb')
            else:
                atoms=P.read_atoms(fn)
            m=P.write_clean_pdb(atoms, f"{sd}/{i}.pdb")
            ch=P.ca_chains(f"{sd}/{i}.pdb")
            # organism
            orgs=set()
            for ent in e['rcsb_entry_container_identifiers'].get('polymer_entity_ids',[]):
                try:
                    pe=json.load(urllib.request.urlopen(f"https://data.rcsb.org/rest/v1/core/polymer_entity/{i}/{ent}"))
                    for s in pe.get('rcsb_entity_source_organism',[]) or []: orgs.add(s.get('ncbi_scientific_name'))
                except Exception: pass
            meta[i]=dict(tier=tier,title=e['struct']['title'],nchain=len(ch),nres=int(sum(len(v[0]) for v in ch.values())),chainmap=m,
                         chains={k:len(v[0]) for k,v in ch.items()},organism=sorted(o for o in orgs if o))
            print(i,tier,meta[i]['nchain'],meta[i]['nres'],meta[i]['organism'],meta[i]['title'][:70],flush=True)
        except Exception as ex:
            print('FAIL',i,ex,flush=True)
json.dump(meta,open('outputs/db_meta.json','w'),indent=1)
