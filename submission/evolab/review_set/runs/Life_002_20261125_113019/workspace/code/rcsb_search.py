import json, urllib.request, sys
q = {"query":{"type":"terminal","service":"full_text","parameters":{"value":sys.argv[1]}},"return_type":"entry","request_options":{"paginate":{"start":0,"rows":100}}}
r = urllib.request.urlopen(urllib.request.Request("https://search.rcsb.org/rcsbsearch/v2/query", data=json.dumps(q).encode(), headers={"Content-Type":"application/json"}))
res = json.load(r); ids=[x["identifier"] for x in res.get("result_set",[])]
print(len(ids)); 
for i in ids:
    try:
        e = json.load(urllib.request.urlopen(f"https://data.rcsb.org/rest/v1/core/entry/{i}"))
        print(i, e["rcsb_entry_info"].get("deposited_polymer_entity_instance_count"), e["rcsb_entry_info"].get("deposited_modeled_polymer_monomer_count"), e["struct"]["title"][:110])
    except Exception as ex: print(i, ex)
