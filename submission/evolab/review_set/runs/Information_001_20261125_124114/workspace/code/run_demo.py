"""Experiment 1: demo-image MCQs (fixed options, frozen in code/demo_questions.json before any run).
For each model: baseline vs rel-att / grad-att / pure-grad ViCrop, plus oracle (GT) crop and random crop.
Answer = argmax over first-token logits of the option letters A-D (free-form greedy output also stored).
usage: python run_demo.py --model llava|qwen [--threads 16]"""
import argparse, json, os, sys, time, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from PIL import Image
from vicrop_common import select_bbox, random_bbox_like, append_jsonl

ap = argparse.ArgumentParser()
ap.add_argument('--model', required=True)
ap.add_argument('--threads', type=int, default=16)
ap.add_argument('--rel-band', default=None)
ap.add_argument('--grad-band', default=None)
ap.add_argument('--max-pixels', type=int, default=448 * 448)
ap.add_argument('--freeform', action='store_true', help='open-ended version of the question (secondary)')
ap.add_argument('--only', default='')
args = ap.parse_args()
torch.set_num_threads(args.threads)
ROOT = '/workspace'
QS = json.load(open(os.path.join(ROOT, 'code/demo_questions.json')))
od = os.path.join(ROOT, 'outputs/demo'); os.makedirs(od, exist_ok=True)
SUF = '_free' if args.freeform else ''
if args.only: QS = [q for q in QS if q['id'] in args.only.split(',')]
outf = os.path.join(od, f'demo_{args.model}{SUF}.jsonl')
if os.path.exists(outf): os.remove(outf)
LET = ['A', 'B', 'C', 'D']
MCQ_SUFFIX = "Answer with the option's letter from the given choices directly."


def mcq_text(q):
    if args.freeform: return q['question'] + '\nAnswer the question using a single word or phrase.'
    return q['question'] + '\n' + '\n'.join(f'({l}) {o}' for l, o in zip(LET, q['options'])) + '\n' + MCQ_SUFFIX


if args.model == 'llava':
    from engine_llava import LlavaEngine, GRID
    eng = LlavaEngine(os.path.join(ROOT, 'outputs/ext/llava15_7b'))
    rel_band = list(range(*map(int, (args.rel_band or '12,17').split(','))))
    grad_band = list(range(*map(int, (args.grad_band or '12,17').split(','))))
    # prompts without the short-answer instruction (MCQ instruction instead)
    P_BASE = lambda t: f"USER: <image>\n{t} ASSISTANT:"
    P_CROP = lambda t: f"USER: <image>\n<image>\n{t} ASSISTANT:"
else:
    from engine_qwen import QwenEngine
    eng = QwenEngine(os.path.join(ROOT, 'outputs/ext/qwen25vl3b'), max_pixels=args.max_pixels)
    rel_band = list(range(*map(int, (args.rel_band or '14,21').split(','))))
    grad_band = list(range(*map(int, (args.grad_band or '14,21').split(','))))
    P_BASE = lambda t: eng.prompt(t, 1)
    P_CROP = lambda t: eng.prompt(t, 2)
letter_ids = [eng.tok.encode(l, add_special_tokens=False)[-1] for l in LET]
if args.model == 'llava':  # llama tokenizer: letter after "ASSISTANT:" is tokenised as "▁A"
    letter_ids = [eng.tok.convert_tokens_to_ids('▁' + l) for l in LET]
print('letter ids', letter_ids, flush=True)


def mcq_from_logits(lg):
    s = lg[letter_ids].float()
    p = torch.softmax(s, 0).numpy()
    return LET[int(np.argmax(p))], p.tolist()


EPS = 1e-8
TILE_MAX = 1024  # high-resolution rule (fixed a priori): images with a side > 1024 px get tiled maps


def maps_for(part, text, with_pure=True):
    """Localisation maps for one (sub-)image, returned on an image-aligned token grid."""
    W, H = part.size
    if args.model == 'llava':
        pre = eng.build_prefix(part, 'x')
        Aq, lq, G = eng.suffix_attn(pre, P_BASE(text), want_grad=True, grad_layers=grad_band)
        Ag, _, _ = eng.suffix_attn(pre, eng.p_generic())
        PG = eng.pure_grad(part, None, prompt=P_BASE(text)) if with_pure else None
        valid = eng.valid_mask((W, H), pre['off'], pre['S']).astype(bool)
        rows = np.where(valid.any(1))[0]; cols = np.where(valid.any(0))[0]
        sub = lambda a: np.asarray(a).reshape(GRID, GRID)[rows.min():rows.max() + 1, cols.min():cols.max() + 1]
        grid = (GRID, GRID)
    else:
        Aq, grid, lq, G, PG = eng.forward_attn(part, P_BASE(text), want_grad=True, grad_layers=grad_band)
        Ag, _, _, _, _ = eng.forward_attn(part, eng.p_generic())
        PG = PG.numpy() if with_pure else None
        sub = lambda a: np.asarray(a).reshape(grid)
    Aq, Ag = Aq.numpy(), Ag.numpy()
    m = {'rel': sub((Aq[rel_band] / (Ag[rel_band] + EPS)).mean(0)), 'grad': sub(G.numpy().sum(0)),
         'raw': sub(Aq[rel_band].mean(0))}
    if PG is not None: m['pure'] = sub(PG)
    return m, dict(Aq=Aq, Ag=Ag)


def tiled_maps(img, text, with_pure):
    from scipy.ndimage import zoom
    W, H = img.size
    nx, ny = int(np.ceil(W / TILE_MAX)), int(np.ceil(H / TILE_MAX))
    xs = np.linspace(0, W, nx + 1).astype(int); ys = np.linspace(0, H, ny + 1).astype(int)
    rows = []; ref = None
    for j in range(ny):
        row = []
        for i in range(nx):
            m, _ = maps_for(img.crop((xs[i], ys[j], xs[i + 1], ys[j + 1])), text, with_pure)
            if ref is None: ref = m['rel'].shape
            row.append({k: zoom(v, (ref[0] / v.shape[0], ref[1] / v.shape[1]), order=1) for k, v in m.items()})
        rows.append(row)
    return {k: np.block([[t[k] for t in row] for row in rows]) for k in rows[0][0]}, (nx, ny)


for q in QS:
    img = Image.open(os.path.join(ROOT, q['image'])).convert('RGB'); W, H = img.size
    text = mcq_text(q)
    t0 = time.time()
    maps, raw = maps_for(img, text, with_pure=True)
    tiles = None
    if max(W, H) > TILE_MAX:
        tmaps, tiles = tiled_maps(img, text, with_pure=(args.model == 'qwen'))
        for k, v in tmaps.items(): maps[k + '_hr'] = v
    bbox_size = 336 if args.model == 'llava' else eng.bbox_size
    boxes = {}
    for m in ['rel', 'grad', 'pure', 'rel_hr', 'grad_hr', 'pure_hr']:
        if m not in maps: continue
        b, r = select_bbox(maps[m], (W, H), bbox_size)
        boxes[m] = tuple(int(v) for v in b)
    if args.model == 'llava': pre = eng.build_prefix(img, 'x')
    # oracle: GT box enlarged to the minimum crop side (bbox_size), centred on the target
    gx1, gy1, gx2, gy2 = q['gt_box']; cx, cy = (gx1 + gx2) / 2, (gy1 + gy2) / 2
    side = max(bbox_size, gx2 - gx1, gy2 - gy1)
    ox1 = int(min(max(0, cx - side / 2), W - side)); oy1 = int(min(max(0, cy - side / 2), H - side))
    boxes['oracle'] = (ox1, oy1, int(ox1 + side), int(oy1 + side))
    boxes['random'] = random_bbox_like(boxes['rel'], (W, H), np.random.default_rng(0))
    res = {}
    if args.model == 'llava':
        base_txt, lg = eng.greedy(pre, P_BASE(text))
    else:
        base_txt, lg = eng.greedy([img], P_BASE(text))
    res['baseline'] = dict(text=base_txt, letter=mcq_from_logits(lg)[0], probs=mcq_from_logits(lg)[1])
    for m, b in boxes.items():
        if args.model == 'llava':
            pv, crop = eng.crop_pv(img, b)
            txt, lg = eng.greedy(pre, P_CROP(text), n_images=2, crop_pv=pv)
        else:
            crop = eng.upscale_to_budget(img.crop(b))
            txt, lg = eng.greedy([img, crop], P_CROP(text))
        res[m] = dict(text=txt, letter=mcq_from_logits(lg)[0], probs=mcq_from_logits(lg)[1])
    # GT attention ratio per layer (raw and relative), map->image coords
    rec = dict(id=q['id'], model=args.model, answer=q['answer'], boxes=boxes, results=res, tiles=tiles,
               time=time.time() - t0)
    np.savez_compressed(os.path.join(od, f"maps_{q['id']}_{args.model}{SUF}.npz"), Aq=raw['Aq'], Ag=raw['Ag'],
                        **{f'map_{k}': v for k, v in maps.items()})
    append_jsonl(outf, rec)
    print(q['id'], {k: v['letter'] for k, v in res.items()}, 'gt', q['answer'], boxes, round(time.time() - t0), flush=True)
print('FINISHED', flush=True)
