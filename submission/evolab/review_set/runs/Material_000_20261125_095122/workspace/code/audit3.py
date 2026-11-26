import sys; sys.path.insert(0, '/workspace/code')
import numpy as np, collections
from common import *
D = load_cache()
for s in ['pretrain', 'finetune', 'candidate']:
    norev = 0; diffattr = 0; odd = 0; ys=[]
    for g in D[s]:
        ei = g['ei'].numpy(); ea = np.round(g['ea'].numpy(), 6)
        E = collections.Counter(map(tuple, ei.T))
        bad = any(E[(b, a)] != c for (a, b), c in E.items())
        norev += bad; odd += ei.shape[1] % 2
        d = collections.defaultdict(set)
        for (a, b), e in zip(ei.T, ea): d[(a, b)].add(tuple(e))
        da = any(d[(a, b)] != d[(b, a)] for (a, b) in d)
        diffattr += da; ys.append(int(bad or da))
    y = np.array([g['y'] for g in D[s]]); ys = np.array(ys)
    print(s, 'missing-reverse', norev, 'odd #edges', odd, 'reverse-attr-mismatch', diffattr, 'pos rate irregular', y[ys==1].mean().round(3), 'n', ys.sum())
