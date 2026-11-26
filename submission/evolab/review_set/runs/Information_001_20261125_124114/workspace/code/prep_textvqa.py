"""Seeded TextVQA-val subset (from lmms-lab/textvqa validation shard 0 = first 1667 val questions).
eval: 200 questions; calib: 40 questions on disjoint images (used only for layer/hyper-parameter checks)."""
import pyarrow.parquet as pq, json, os, random
f = pq.ParquetFile('/workspace/outputs/ext/textvqa_hf/data/validation-00000-of-00003.parquet')
t = f.read(columns=['question_id', 'image_id', 'question', 'answers', 'image_width', 'image_height', 'image']).to_pylist()
rng = random.Random(0)
idx = list(range(len(t))); rng.shuffle(idx)
ev, cal, ev_imgs = [], [], set()
for i in idx:
    r = t[i]
    if len(ev) < 200: ev.append(i); ev_imgs.add(r['image_id'])
    elif len(cal) < 40 and r['image_id'] not in ev_imgs: cal.append(i)
od = '/workspace/outputs/ext/textvqa_imgs'; os.makedirs(od, exist_ok=True)
def rec(i):
    r = t[i]
    p = os.path.join(od, r['image_id'] + '.jpg')
    if not os.path.exists(p): open(p, 'wb').write(r['image']['bytes'])
    return dict(question_id=r['question_id'], image_id=r['image_id'], question=r['question'], answers=r['answers'],
                W=r['image_width'], H=r['image_height'])
meta = dict(source='lmms-lab/textvqa validation-00000-of-00003.parquet', seed=0,
            eval=[rec(i) for i in ev], calib=[rec(i) for i in cal])
json.dump(meta, open('/workspace/outputs/textvqa_ids.json', 'w'), indent=1)
print(len(meta['eval']), len(meta['calib']), len(ev_imgs))
import collections; print(collections.Counter((m['W'], m['H']) for m in meta['eval']).most_common(5))
