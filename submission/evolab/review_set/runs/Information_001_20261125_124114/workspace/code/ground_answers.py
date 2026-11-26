"""Pseudo ground-truth evidence boxes for TextVQA questions (no box annotations are available offline).
Qwen2.5-VL-3B at high resolution (640x640 px budget; first few calib items at 1024x1024) is asked to ground the majority human answer string.
Used ONLY for diagnostics (attention-ratio-by-layer curves, crop hit rates), never for choosing crops.
usage: python ground_answers.py --split eval|calib --shard i --nshards n"""
import argparse, json, os, re, sys, collections, time, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from PIL import Image
from vicrop_common import append_jsonl, read_jsonl

ap = argparse.ArgumentParser()
ap.add_argument('--split', default='eval'); ap.add_argument('--shard', type=int, default=0)
ap.add_argument('--nshards', type=int, default=1); ap.add_argument('--threads', type=int, default=8)
args = ap.parse_args(); torch.set_num_threads(args.threads)
from engine_qwen import QwenEngine
ROOT = '/workspace'
eng = QwenEngine(os.path.join(ROOT, 'outputs/ext/qwen25vl3b'), max_pixels=640 * 640)
qs = json.load(open(os.path.join(ROOT, 'outputs/textvqa_ids.json')))[args.split][args.shard::args.nshards]
od = os.path.join(ROOT, 'outputs/grounding'); os.makedirs(od, exist_ok=True)
outf = os.path.join(od, f'{args.split}_shard{args.shard}of{args.nshards}.jsonl')
done = {r['question_id'] for r in read_jsonl(outf)}
for q in qs:
    if q['question_id'] in done: continue
    t0 = time.time()
    img = Image.open(os.path.join(ROOT, 'outputs/ext/textvqa_imgs', q['image_id'] + '.jpg')).convert('RGB')
    W, H = img.size
    ans = collections.Counter(a.lower().strip() for a in q['answers']).most_common(1)[0][0]
    if ans in ('unanswerable', 'yes', 'no'):
        append_jsonl(outf, dict(question_id=q['question_id'], answer=ans, raw='', box=None, skipped=True)); continue
    prompt = eng.prompt(f'Locate the text or object that answers the question "{q["question"]}" '
                        f'(the answer is "{ans}"). Output its bbox coordinates in JSON format.', 1)
    inp = eng.inputs([img], prompt)
    thw = inp['image_grid_thw'][0]; rh, rw = int(thw[1]) * 14, int(thw[2]) * 14
    eng.model.model.rope_deltas = None
    with torch.no_grad():
        out = eng.model.generate(**inp, max_new_tokens=40, do_sample=False)
    txt = eng.tok.decode(out[0, inp['input_ids'].shape[1]:], skip_special_tokens=True)
    eng.model.model.rope_deltas = None
    m = re.search(r'\[\s*(\d+\.?\d*)\s*,\s*(\d+\.?\d*)\s*,\s*(\d+\.?\d*)\s*,\s*(\d+\.?\d*)\s*\]', txt)
    box = None
    if m:
        x1, y1, x2, y2 = [float(v) for v in m.groups()]
        box = [x1 * W / rw, y1 * H / rh, x2 * W / rw, y2 * H / rh]
        if box[2] <= box[0] or box[3] <= box[1]: box = None
    append_jsonl(outf, dict(question_id=q['question_id'], answer=ans, raw=txt, box=box, resized=[rw, rh],
                            time=time.time() - t0))
    print(q['question_id'], ans, box, round(time.time() - t0, 1), flush=True)
print('FINISHED', flush=True)
