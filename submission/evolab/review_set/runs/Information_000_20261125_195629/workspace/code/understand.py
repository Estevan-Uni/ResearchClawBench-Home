"""Multimodal understanding with Janus-1.3B (SigLIP -> und. adaptor -> LLM).
Usage: python understand.py jobs.json out.jsonl
jobs.json: list of {"id","image","turns":[q1,(q2..)] } ; multi-turn uses previous answers.
Official conversation format, greedy decoding, max_new_tokens=512."""
import json, sys, time, os
from janus_common import load, torch
from PIL import Image

jobs = json.load(open(sys.argv[1])); out = sys.argv[2]
done = set()
if os.path.exists(out):
    done = {json.loads(l)["id"] for l in open(out)}
proc, model = load(); tok = proc.tokenizer

@torch.inference_mode()
def ask(conv, imgs):
    inp = proc(conversations=conv, images=imgs, force_batchify=True)
    emb = model.prepare_inputs_embeds(**inp)
    o = model.language_model.generate(inputs_embeds=emb, attention_mask=inp.attention_mask,
        pad_token_id=tok.eos_token_id, bos_token_id=tok.bos_token_id, eos_token_id=tok.eos_token_id,
        max_new_tokens=512, do_sample=False, use_cache=True)
    return tok.decode(o[0].cpu().tolist(), skip_special_tokens=True), inp["sft_format"][0]

for j in jobs:
    if j["id"] in done: continue
    img = j["image"]
    pil = Image.open(img).convert("RGB") if isinstance(img, str) else None
    if "transform" in j:
        from robust_tf import apply
        pil = apply(pil, j["transform"])
    hist = []; answers = []
    for k, q in enumerate(j["turns"]):
        content = ("<image_placeholder>\n" + q) if k == 0 else q
        hist.append({"role": "User", "content": content, **({"images": ["x"]} if k == 0 else {})})
        conv = hist + [{"role": "Assistant", "content": ""}]
        t = time.time(); a, sft = ask(conv, [pil]); dt = time.time() - t
        hist.append({"role": "Assistant", "content": a})
        answers.append({"q": q, "a": a, "sec": round(dt, 1)})
        print(f"[{j['id']}] turn{k} ({dt:.0f}s): {a}\n", flush=True)
    rec = {"id": j["id"], "image": img if isinstance(img, str) else None, "transform": j.get("transform"),
           "answers": answers, "last_prompt": sft}
    open(out, "a").write(json.dumps(rec, ensure_ascii=False) + "\n")
