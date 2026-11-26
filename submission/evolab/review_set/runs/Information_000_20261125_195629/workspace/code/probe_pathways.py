"""Mechanistic probes of the decoupled visual encoders in Janus-1.3B.
(1) VQ (generation encoder) reconstruction of the two probe images -> PSNR (fine detail retained?)
(2) Pathway swap: feed VQ codes through gen_embed->gen_aligner into the *understanding* prompt
    instead of SigLIP->aligner, with and without banning image special tokens.
(3) Embedding geometry of the two adaptors' outputs in the shared LLM input space.
Outputs -> /workspace/outputs/probes/"""
import json, os, numpy as np, PIL.Image
from janus_common import load, torch
O = "/workspace/outputs/probes"; os.makedirs(O, exist_ok=True)
proc, model = load(); tok = proc.tokenizer
IMGS = {"equation": "/workspace/data/equation.png", "doge": "/workspace/data/doge.png"}
Q = {"equation": "Convert the formula into latex code.", "doge": "Explain this meme."}
special = [proc.image_start_id, proc.image_end_id, proc.image_id]
res = {}

@torch.inference_mode()
def run(name):
    pil = PIL.Image.open(IMGS[name]).convert("RGB")
    conv = [{"role": "User", "content": "<image_placeholder>\n" + Q[name], "images": ["x"]},
            {"role": "Assistant", "content": ""}]
    inp = proc(conversations=conv, images=[pil], force_batchify=True)
    px = inp.pixel_values[0]                       # [1,3,384,384] in [-1,1]
    # (1) VQ encode/decode
    quant, _, info = model.gen_vision_model.encode(px)
    codes = info[2].view(1, -1)                    # 576 ids
    rec = model.gen_vision_model.decode_code(codes, shape=[1, 8, 24, 24])
    a = ((px[0].permute(1, 2, 0).numpy() + 1) / 2).clip(0, 1); b = ((rec[0].permute(1, 2, 0).numpy() + 1) / 2).clip(0, 1)
    mse = float(((a - b) ** 2).mean()); psnr = 10 * np.log10(1 / mse)
    PIL.Image.fromarray((a * 255).astype(np.uint8)).save(f"{O}/{name}_input384.png")
    PIL.Image.fromarray((b * 255).astype(np.uint8)).save(f"{O}/{name}_vqrecon.png")
    # (2) pathway swap
    und = model.aligner(model.vision_model(px))[0]          # [576,2048]
    gen = model.prepare_gen_img_embeds(codes[0])            # [576,2048]
    base = model.language_model.get_input_embeddings()(inp.input_ids.clamp(min=0))
    mask = inp.images_seq_mask[0]
    outs = {}
    for mode, feats in [("siglip_und_adaptor", und), ("vq_gen_adaptor", gen)]:
        for ban in [False, True]:
            e = base.clone(); e[0, mask] = feats
            kw = dict(bad_words_ids=[[s] for s in special]) if ban else {}
            o = model.language_model.generate(inputs_embeds=e, attention_mask=inp.attention_mask,
                pad_token_id=tok.eos_token_id, bos_token_id=tok.bos_token_id, eos_token_id=tok.eos_token_id,
                max_new_tokens=160, do_sample=False, use_cache=True, **kw)
            ids = o[0].tolist()
            outs[f"{mode}|ban={ban}"] = {"raw": tok.decode(ids, skip_special_tokens=False),
                                         "clean": tok.decode(ids, skip_special_tokens=True), "n_tokens": len(ids)}
            print(name, mode, ban, repr(outs[f"{mode}|ban={ban}"]["raw"][:300]), flush=True)
    # (3) geometry
    txt = model.language_model.get_input_embeddings().weight[:100000:50]
    def nrm(x): return x / x.norm(dim=-1, keepdim=True)
    geo = {"norm_und": und.norm(dim=-1).tolist(), "norm_gen": gen.norm(dim=-1).tolist(),
           "norm_txt": txt.norm(dim=-1).tolist(),
           "cos_und_gen_samepos": (nrm(und) * nrm(gen)).sum(-1).tolist(),
           "cos_und_und": (nrm(und) @ nrm(und).T)[np.triu_indices(576, 1)].tolist(),
           "cos_gen_gen": (nrm(gen) @ nrm(gen).T)[np.triu_indices(576, 1)].tolist(),
           "cos_und_txt_max": (nrm(und) @ nrm(txt).T).max(-1).values.tolist(),
           "cos_gen_txt_max": (nrm(gen) @ nrm(txt).T).max(-1).values.tolist()}
    np.savez_compressed(f"{O}/{name}_geometry.npz", **{k: np.array(v, dtype=np.float32) for k, v in geo.items()})
    res[name] = {"psnr_vq": psnr, "mse_vq": mse, "n_unique_codes": int(len(set(codes[0].tolist()))), "outputs": outs}

for n in IMGS: run(n)
json.dump(res, open(f"{O}/pathway_probe.json", "w"), indent=1, ensure_ascii=False)
