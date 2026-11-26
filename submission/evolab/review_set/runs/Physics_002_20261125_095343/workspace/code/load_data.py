"""Load, parse and audit the RCS experimental data. Produces outputs/audit_table.csv and a cached pickle."""
import glob, json, os, re, hashlib, pickle
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAT = re.compile(r'N(\d+)_d(\d+)_r(\d+)_(XEB|MB|Transport_1QRB)_counts\.json')

def parse_key(k):
    k = k.strip()
    if k.startswith('('):
        return tuple(int(x) for x in k.strip('()').split(',') if x.strip() != '')
    return tuple(int(c) for c in k)

def md5(path):
    return hashlib.md5(open(path, 'rb').read()).hexdigest()

def load_all():
    recs, seen = [], {}
    dups = []
    for f in sorted(glob.glob(os.path.join(ROOT, 'data/results/*/*/*_counts.json'))):
        m = PAT.search(os.path.basename(f))
        if not m: continue
        N, d, r, typ = int(m[1]), int(m[2]), int(m[3]), m[4]
        family = f.split('/')[-3]
        h = md5(f) + os.path.basename(f)
        if h in seen:
            dups.append((family, os.path.basename(f), seen[h])); continue
        seen[h] = family
        counts = {parse_key(k): int(v) for k, v in json.load(open(f)).items()}
        rec = dict(family=family, type=typ, N=N, d=d, r=r, counts=counts, path=f)
        if typ == 'XEB':
            af = f.replace('/results/', '/amplitudes/').replace('_counts.json', '_amplitudes.json')
            if os.path.exists(af):
                amp = json.load(open(af))
                rec['ideal_probs'] = {parse_key(k): abs(complex(v.replace(' ', '')))**2 for k, v in amp.items()}
        else:
            bf = f.replace('_counts.json', '_ideal_bitstring.json')
            rec['ideal_bitstring'] = tuple(int(x) for x in json.load(open(bf)))
        recs.append(rec)
    return recs, dups

if __name__ == '__main__':
    recs, dups = load_all()
    print(f'loaded {len(recs)} unique circuits; {len(dups)} duplicate files skipped')
    dupdf = pd.DataFrame(dups, columns=['family', 'file', 'first_seen_in'])
    dupdf.to_csv(os.path.join(ROOT, 'outputs/duplicates_removed.csv'), index=False)
    print(dupdf.groupby(['family', 'first_seen_in']).size())
    rows = []
    for rec in recs:
        n = sum(rec['counts'].values())
        lens = {len(k) for k in rec['counts']}
        assert lens == {rec['N']}, (rec['path'], lens)
        row = dict(family=rec['family'], type=rec['type'], N=rec['N'], d=rec['d'], r=rec['r'], shots=n,
                   distinct=len(rec['counts']))
        if 'ideal_probs' in rec:
            matched = sum(1 for k in rec['counts'] if k in rec['ideal_probs'])
            row.update(n_amp_keys=len(rec['ideal_probs']), matched=matched, missing=len(rec['counts']) - matched)
        if 'ideal_bitstring' in rec:
            row['target_hits'] = rec['counts'].get(rec['ideal_bitstring'], 0)
            assert len(rec['ideal_bitstring']) == rec['N']
        rows.append(row)
    df = pd.DataFrame(rows)
    agg = df.groupby(['family', 'type', 'N', 'd']).agg(circuits=('r', 'size'), r_min=('r', 'min'), r_max=('r', 'max'),
        shots_per_circuit=('shots', 'median'), shots_min=('shots', 'min'), shots_max=('shots', 'max'),
        amp_keys=('n_amp_keys', 'median'), matched=('matched', 'median'), missing_total=('missing', 'sum'),
        has_amp=('n_amp_keys', lambda s: int(s.notna().sum()))).reset_index()
    pd.set_option('display.width', 250); pd.set_option('display.max_rows', 200)
    print(agg)
    agg.to_csv(os.path.join(ROOT, 'outputs/audit_table.csv'), index=False)
    xe = df[(df.type == 'XEB') & df.n_amp_keys.notna()]
    assert (xe.missing == 0).all(), 'unmatched sampled bitstrings!'
    print('All XEB circuits with amplitudes have every sampled bitstring matched.')
    pickle.dump(recs, open(os.path.join(ROOT, 'outputs/data_cache.pkl'), 'wb'))
