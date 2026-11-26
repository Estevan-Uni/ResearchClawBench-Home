import torch, sys
sys.path.insert(0, '/workspace/code')
from loader import load_graphs
out = {}
for f in ['pretrain', 'finetune', 'candidate']:
    gs, meta = load_graphs(f'/workspace/data/{f}_data.pt')
    out[f] = [{'x': g['x'].argmax(1).long(), 'xoh': g['x'], 'ei': g['edge_index'].long(),
               'ea': g['edge_attr'].float(), 'y': int(g['y'].item())} for g in gs]
    out[f + '_meta'] = {k: v for k, v in meta.items() if k in ('num_samples', 'pos_ratio')}
torch.save(out, '/workspace/outputs/graphs_cache.pt')
print('cached')
