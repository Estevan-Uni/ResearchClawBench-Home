"""TextVQA ViCrop evaluation driver (sharded, resumable).

usage: python run_textvqa.py --model llava|qwen --ids outputs/textvqa_ids.json --split eval|calib
                             --shard i --nshards n [--threads 8] [--no-pure] [--rel-band a,b] [--grad-band a,b]
Per question it stores answers for: baseline, rel-att, grad-att, pure-grad, random-crop (size-matched to rel-att),
the selected boxes, timings, and saves all-layer attention maps to an .npz (for diagnostics).
"""
import argparse, json, os, sys, time, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from PIL import Image
from vicrop_common import select_bbox, random_bbox_like, vqa_accuracy, clean_pred, append_jsonl, read_jsonl

ap = argparse.ArgumentParser()
ap.add_argument('--model', required=True)
ap.add_argument('--ids', default='outputs/textvqa_ids.json')
ap.add_argument('--split', default='eval')
ap.add_argument('--shard', type=int, default=0)
ap.add_argument('--nshards', type=int, default=1)
ap.add_argument('--threads', type=int, default=8)
ap.add_argument('--no-pure', action='store_true')
ap.add_argument('--rel-band', default=None)
ap.add_argument('--grad-band', default=None)
ap.add_argument('--max-pixels', type=int, default=448 * 448)
ap.add_argument('--limit', type=int, default=0)
ap.add_argument('--tag', default='')
ap.add_argument('--maps-only', action='store_true', help='calibration: only compute maps (all layers)')
ap.add_argument('--baseline-only', action='store_true', help='only the no-crop baseline (e.g. native resolution)')
args = ap.parse_args()
torch.set_num_threads(args.threads)

ROOT = '/workspace'
meta = json.load(open(os.path.join(ROOT, args.ids)))
qs = meta[args.split]
qs = qs[args.shard::args.nshards]
if args.limit: qs = qs[:args.limit]
outdir = os.path.join(ROOT, 'outputs', f'textvqa_{args.model}{args.tag}', args.split)
mapdir = os.path.join(outdir, 'maps'); os.makedirs(mapdir, exist_ok=True)
outf = os.path.join(outdir, f'shard{args.shard}of{args.nshards}.jsonl')
done = {r['question_id'] for r in read_jsonl(outf)}
print(f'{len(qs)} questions, {len(done)} done', flush=True)

if args.model == 'llava':
    from engine_llava import LlavaEngine, GRID
    eng = LlavaEngine(os.path.join(ROOT, 'outputs/ext/llava15_7b'))
    L = eng.n_layers
    rel_band = list(range(*map(int, (args.rel_band or '12,17').split(','))))
    grad_band = list(range(*map(int, (args.grad_band or '12,17').split(','))))
else:
    from engine_qwen import QwenEngine
    eng = QwenEngine(os.path.join(ROOT, 'outputs/ext/qwen25vl3b'), max_pixels=args.max_pixels)
    L = eng.n_layers
    rel_band = list(range(*map(int, (args.rel_band or '14,21').split(','))))
    grad_band = list(range(*map(int, (args.grad_band or '14,21').split(','))))
all_layers = list(range(L))
EPS = 1e-8


def to_img_box(box, off):
    x1, y1, x2, y2 = box
    return (x1 - off[0], y1 - off[1], x2 - off[0], y2 - off[1])


def clip_box(b, W, H):
    x1, y1, x2, y2 = b
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(W, x2), min(H, y2)
    if x2 - x1 < 8: x2 = min(W, x1 + 8)
    if y2 - y1 < 8: y2 = min(H, y1 + 8)
    return (int(x1), int(y1), int(x2), int(y2))


for qi, q in enumerate(qs):
    if q['question_id'] in done: continue
    if args.baseline_only:
        t0 = time.time()
        img = Image.open(os.path.join(ROOT, 'outputs/ext/textvqa_imgs', q['image_id'] + '.jpg')).convert('RGB')
        p, _ = eng.greedy([img], eng.p_base(q['question'].strip()), max_new=10)
        p = clean_pred(p)
        append_jsonl(outf, dict(question_id=q['question_id'], answers=q['answers'], pred={'baseline': p},
                                acc={'baseline': vqa_accuracy(p, q['answers'])}, total_time=time.time() - t0))
        print(qi, p, round(time.time() - t0, 1), flush=True); continue
    t0 = time.time(); T = {}
    img = Image.open(os.path.join(ROOT, 'outputs/ext/textvqa_imgs', q['image_id'] + '.jpg')).convert('RGB')
    W, H = img.size
    question = q['question'].strip()
    if not question.endswith('?'): question = question
    rng = np.random.default_rng(q['question_id'])
    rec = dict(question_id=q['question_id'], image_id=q['image_id'], question=question, answers=q['answers'],
               W=W, H=H)
    gband = all_layers  # grads for all layers (one backward); crop uses grad_band only
    if args.model == 'llava':
        t = time.time(); pre = eng.build_prefix(img, question); T['prefix'] = time.time() - t
        t = time.time(); Aq, lq, G = eng.suffix_attn(pre, eng.p_base(question), want_grad=True, grad_layers=gband,
                                                    keep_cache=not args.maps_only)
        T['att_q+grad'] = time.time() - t
        if not args.maps_only:
            t = time.time(); base_pred = eng.continue_greedy(pre, int(lq.argmax()), None); T['baseline'] = time.time() - t
        t = time.time(); Ag, _, _ = eng.suffix_attn(pre, eng.p_generic()); T['att_g'] = time.time() - t
        grid = (GRID, GRID)
        valid = eng.valid_mask((W, H), pre['off'], pre['S'])
        region = (pre['S'], pre['S']); bbox_size = 336; off = pre['off']
        PG = None
        if not args.no_pure and not args.maps_only:
            t = time.time(); PG = eng.pure_grad(img, question); T['pure_grad'] = time.time() - t
    else:
        t = time.time()
        Aq, grid, _, G, PG = eng.forward_attn(img, eng.p_base(question), want_grad=True, grad_layers=gband)
        T['att_q+grad'] = time.time() - t
        t = time.time(); Ag, _, _, _, _ = eng.forward_attn(img, eng.p_generic()); T['att_g'] = time.time() - t
        valid = np.ones(grid); region = (W, H); bbox_size = eng.bbox_size; off = (0, 0)
        PG = PG.numpy().reshape(grid) if (PG is not None and not args.no_pure) else None
    Aq_np, Ag_np = Aq.numpy(), Ag.numpy()
    np.savez_compressed(os.path.join(mapdir, f"{q['question_id']}.npz"), Aq=Aq_np.astype(np.float32),
                        Ag=Ag_np.astype(np.float32), G=G.numpy().astype(np.float32),
                        PG=(PG.astype(np.float32) if PG is not None else np.zeros(1)), grid=np.array(grid),
                        valid=valid, off=np.array(off), region=np.array(region), gband=np.array(gband))
    if args.maps_only:
        rec['timing'] = T; rec['grid'] = list(grid)
        append_jsonl(outf, rec); print(qi, 'maps', round(time.time() - t0, 1), flush=True); continue

    maps = {
        'rel': (Aq_np[rel_band] / (Ag_np[rel_band] + EPS)).mean(0).reshape(grid),
        'grad': G.numpy()[grad_band].sum(0).reshape(grid),
    }
    if PG is not None: maps['pure'] = PG.reshape(grid)
    boxes = {}
    for m, a in maps.items():
        b, r = select_bbox(a, region, bbox_size, valid=valid)
        boxes[m] = clip_box(to_img_box(b, off), W, H)
    boxes['random'] = random_bbox_like(boxes['rel'], (W, H), rng)
    rec['boxes'] = boxes
    preds = {}
    if args.model == 'llava':
        preds['baseline'] = base_pred
        t = time.time()
        uniq = []
        for m, b in boxes.items():
            if tuple(b) not in uniq: uniq.append(tuple(b))
        outs, _ = eng.greedy_batch(pre, eng.p_crop(question), [eng.crop_pv(img, b)[0] for b in uniq])
        for m, b in boxes.items(): preds[m] = outs[uniq.index(tuple(b))]
        T['crops_batched'] = time.time() - t; T['n_unique_crops'] = len(uniq)
    else:
        t = time.time(); preds['baseline'], _ = eng.greedy([img], eng.p_base(question)); T['baseline'] = time.time() - t
    for m, b in (boxes.items() if args.model != 'llava' else []):
        t = time.time()
        if args.model == 'llava':
            pv, crop = eng.crop_pv(img, b)
            preds[m], _ = eng.greedy(pre, eng.p_crop(question), n_images=2, crop_pv=pv)
        else:
            crop = eng.upscale_to_budget(img.crop(b))
            preds[m], _ = eng.greedy([img, crop], eng.p_crop(question))
        T['crop_' + m] = time.time() - t
    preds = {k: clean_pred(v) for k, v in preds.items()}
    rec['pred'] = preds
    rec['acc'] = {k: vqa_accuracy(v, q['answers']) for k, v in preds.items()}
    rec['timing'] = T; rec['total_time'] = time.time() - t0
    rec['rel_band'] = rel_band; rec['grad_band'] = grad_band
    append_jsonl(outf, rec)
    print(qi, q['question_id'], json.dumps(rec['pred']), json.dumps(rec['acc']), round(rec['total_time'], 1), flush=True)
print('FINISHED', flush=True)
