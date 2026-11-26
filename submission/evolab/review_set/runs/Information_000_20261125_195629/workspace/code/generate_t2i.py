"""Text-to-image with Janus-1.3B (LLM -> gen head over VQ codebook -> VQ decoder), official CFG loop.
Usage: python generate_t2i.py jobs.json outdir"""
import json, sys, os, time
import numpy as np, PIL.Image
from janus_common import load, torch

jobs = json.load(open(sys.argv[1])); outdir = sys.argv[2]; os.makedirs(outdir, exist_ok=True)
proc, model = load()

@torch.inference_mode()
def generate(prompt, seed, cfg_weight=5.0, n=2, temperature=1.0, ntok=576, img_size=384, patch=16):
    torch.manual_seed(seed); np.random.seed(seed)
    conv = [{"role": "User", "content": prompt}, {"role": "Assistant", "content": ""}]
    sft = proc.apply_sft_template_for_multi_turn_prompts(conversations=conv, sft_format=proc.sft_format, system_prompt="")
    ids = torch.LongTensor(proc.tokenizer.encode(sft + proc.image_start_tag))
    tokens = torch.zeros((n * 2, len(ids)), dtype=torch.int)
    for i in range(n * 2):
        tokens[i] = ids
        if i % 2: tokens[i, 1:-1] = proc.pad_id  # unconditional branch
    emb = model.language_model.get_input_embeddings()(tokens)
    gen = torch.zeros((n, ntok), dtype=torch.int); past = None
    for i in range(ntok):
        out = model.language_model.model(inputs_embeds=emb, use_cache=True, past_key_values=past)
        past = out.past_key_values
        logits = model.gen_head(out.last_hidden_state[:, -1, :])
        lc, lu = logits[0::2], logits[1::2]
        probs = torch.softmax((lu + cfg_weight * (lc - lu)) / temperature, dim=-1)
        nxt = torch.multinomial(probs, 1)
        gen[:, i] = nxt.squeeze(-1)
        emb = model.prepare_gen_img_embeds(torch.cat([nxt, nxt], 1).view(-1)).unsqueeze(1)
        if i % 96 == 0: print(f"  token {i}/{ntok}", flush=True)
    dec = model.gen_vision_model.decode_code(gen, shape=[n, 8, img_size // patch, img_size // patch])
    dec = np.clip((dec.float().numpy().transpose(0, 2, 3, 1) + 1) / 2 * 255, 0, 255).astype(np.uint8)
    return dec, gen.numpy()

for j in jobs:
    tag = j["tag"]
    if os.path.exists(os.path.join(outdir, f"{tag}_0.png")): continue
    t = time.time()
    imgs, codes = generate(j["prompt"], j.get("seed", 0), j.get("cfg", 5.0), j.get("n", 2))
    for i, im in enumerate(imgs):
        PIL.Image.fromarray(im).save(os.path.join(outdir, f"{tag}_{i}.png"))
    np.save(os.path.join(outdir, f"{tag}_codes.npy"), codes)
    rec = {**j, "sec": round(time.time() - t, 1)}
    open(os.path.join(outdir, "log.jsonl"), "a").write(json.dumps(rec) + "\n")
    print(f"[{tag}] done in {rec['sec']}s", flush=True)
