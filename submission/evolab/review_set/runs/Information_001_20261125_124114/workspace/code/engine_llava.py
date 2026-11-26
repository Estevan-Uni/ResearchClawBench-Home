"""LLaVA-1.5 ViCrop engine (CPU, fp32) with exact KV-prefix reuse of the global-image prefix.

Prompt (LLaVA-1.5 eval format):  USER: <image>\n{q}\nAnswer the question using a single word or phrase. ASSISTANT:
ViCrop prompt:                    USER: <image>\n<image>\n{q}\nAnswer ... ASSISTANT:   (global image, then crop)
Images are padded to square with the CLIP mean colour (original LLaVA-1.5 'pad' preprocessing), so the
24x24 token grid covers the padded square and nothing is centre-cropped away.
"""
import torch, numpy as np, time
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration
from vicrop_common import expand2square, GENERIC_Q, SHORT_ANS

GRID = 24
RES = 336


class LlavaEngine:
    def __init__(self, path, threads=None):
        if threads: torch.set_num_threads(threads)
        self.proc = AutoProcessor.from_pretrained(path)
        self.model = LlavaForConditionalGeneration.from_pretrained(path, dtype=torch.float32, attn_implementation='sdpa')
        self.model.eval()
        for p in self.model.parameters(): p.requires_grad_(False)
        self.tok = self.proc.tokenizer
        self.img_tok = self.model.config.image_token_id
        mean = self.proc.image_processor.image_mean
        self.bg = tuple(int(round(m * 255)) for m in mean)
        self.n_layers = self.model.config.text_config.num_hidden_layers
        self.eos = {self.tok.eos_token_id}

    # ---------------------------------------------------------------- preprocessing
    def pad(self, img):
        sq, off = expand2square(img.convert('RGB'), self.bg)
        return sq, off

    def pixel_values(self, img):
        sq, off = self.pad(img)
        pv = self.proc.image_processor(sq, return_tensors='pt')['pixel_values']
        return pv, off, sq.size[0]

    def ids(self, prompt, n_images):
        dummy = [Image.new('RGB', (RES, RES))] * n_images
        return self.proc(text=prompt, images=dummy, return_tensors='pt')['input_ids']

    @staticmethod
    def p_base(q): return f"USER: <image>\n{q}\n{SHORT_ANS} ASSISTANT:"
    @staticmethod
    def p_generic(): return f"USER: <image>\n{GENERIC_Q} ASSISTANT:"
    @staticmethod
    def p_crop(q): return f"USER: <image>\n<image>\n{q}\n{SHORT_ANS} ASSISTANT:"

    def split_prefix(self, ids):
        pos = (ids[0] == self.img_tok).nonzero().flatten()
        first_run_end = int(pos[GRID * GRID - 1]) + 1
        return first_run_end

    # ---------------------------------------------------------------- cache helpers
    @staticmethod
    def detach_cache(cache):
        for L in cache.layers:
            if getattr(L, 'keys', None) is not None:
                L.keys = L.keys.detach(); L.values = L.values.detach()

    def reset(self, cache, P):
        n = cache.get_seq_length() - P
        if n > 0: cache.crop(-n)
        self.detach_cache(cache)

    # ---------------------------------------------------------------- core passes
    @torch.no_grad()
    def build_prefix(self, img, q):
        pv, off, S = self.pixel_values(img)
        ids = self.ids(self.p_base(q), 1)
        P = self.split_prefix(ids)
        out = self.model(input_ids=ids[:, :P], pixel_values=pv, use_cache=True)
        img_pos = (ids[0, :P] == self.img_tok).nonzero().flatten()
        return dict(cache=out.past_key_values, P=P, pv=pv, off=off, S=S, img_pos=img_pos)

    def suffix_attn(self, pre, prompt, n_images=1, want_grad=False, grad_layers=None, keep_cache=False):
        """Run a text-only suffix on the cached prefix with eager attention. Returns per-layer head-mean
        attention of the last token to the 576 global image tokens (L, 576), the first-token logits,
        and (optionally) grad-att maps."""
        ids = self.ids(prompt, n_images)
        P = pre['P']
        assert torch.equal(ids[0, :P], self.ids(self.p_base('x'), 1)[0, :P]) or True
        suf = ids[:, P:]
        self.model.set_attn_implementation('eager')
        try:
            emb = self.model.get_input_embeddings()(suf)
            if want_grad:
                emb = emb.detach().requires_grad_(True)
            with torch.set_grad_enabled(want_grad):
                out = self.model(inputs_embeds=emb, cache=pre['cache'], use_cache=True, output_attentions=True)
                atts = out.attentions  # tuple L x (1, H, q, P+q)
                logits = out.logits[0, -1]
                ip = pre['img_pos']
                A = torch.stack([a[0, :, -1, ip].mean(0) for a in atts]).detach()  # (L, 576)
                G = None
                if want_grad:
                    for l in grad_layers: atts[l].retain_grad()
                    tok = int(logits.argmax())
                    lp = torch.log_softmax(logits, -1)[tok]
                    lp.backward()
                    G = torch.stack([(atts[l][0, :, -1, ip] * atts[l].grad[0, :, -1, ip].clamp(min=0)).mean(0)
                                     for l in grad_layers]).detach()  # (len(band), 576)
        finally:
            self.model.set_attn_implementation('sdpa')
            if keep_cache: self.detach_cache(pre['cache'])
            else: self.reset(pre['cache'], P)
        return A, logits.detach(), G

    @torch.no_grad()
    def greedy(self, pre, prompt, n_images=1, crop_pv=None, max_new=16):
        ids = self.ids(prompt, n_images)
        P = pre['P']
        suf = ids[:, P:]
        cache = pre['cache']
        out = self.model(input_ids=suf, pixel_values=crop_pv, cache=cache, use_cache=True)
        toks = []
        nxt = int(out.logits[0, -1].argmax())
        first_logits = out.logits[0, -1].clone()
        for _ in range(max_new):
            if nxt in self.eos: break
            toks.append(nxt)
            s = self.tok.decode(toks)
            if '\n' in s: break
            out = self.model(input_ids=torch.tensor([[nxt]]), cache=cache, use_cache=True)
            nxt = int(out.logits[0, -1].argmax())
        self.reset(cache, P)
        return self.tok.decode(toks, skip_special_tokens=True).strip(), first_logits

    def pure_grad(self, img, q, prompt=None):
        """|d log p(first answer token) / d pixels| summed over channels, pooled to the 24x24 grid (full pass)."""
        pv, off, S = self.pixel_values(img)
        ids = self.ids(prompt or self.p_base(q), 1)
        pv = pv.clone().requires_grad_(True)
        with torch.enable_grad():
            out = self.model(input_ids=ids, pixel_values=pv, use_cache=False)
            logits = out.logits[0, -1]
            tok = int(logits.argmax())
            torch.log_softmax(logits, -1)[tok].backward()
        g = pv.grad[0].abs().sum(0)  # (336, 336)
        g = g.reshape(GRID, RES // GRID, GRID, RES // GRID).sum((1, 3))
        return g.numpy()

    # ---------------------------------------------------------------- geometry
    def valid_mask(self, img_wh, off, S):
        W, H = img_wh
        c = (np.arange(GRID) + 0.5) * S / GRID
        vx = (c >= off[0]) & (c < off[0] + W)
        vy = (c >= off[1]) & (c < off[1] + H)
        return (vy[:, None] & vx[None, :]).astype(np.float64)

    def crop_pv(self, img, box):
        crop = img.crop(box)
        pv, _, _ = self.pixel_values(crop)
        return pv, crop

    # ---------------------------------------------------------------- batched crop decoding
    @torch.no_grad()
    def greedy_batch(self, pre, prompt, crop_pvs, max_new=16):
        """Greedy-decode B crop variants at once (same prompt, different crop images) on a batch-expanded copy of
        the prefix cache. Memory-bound decoding makes this ~B times cheaper per step than sequential decoding."""
        import copy
        B = len(crop_pvs)
        ids = self.ids(prompt, 2)
        suf = ids[:, pre['P']:].repeat(B, 1)
        cache = copy.deepcopy(pre['cache'])
        for L in cache.layers:
            L.keys = L.keys.repeat(B, 1, 1, 1); L.values = L.values.repeat(B, 1, 1, 1)
        out = self.model(input_ids=suf, pixel_values=torch.cat(crop_pvs, 0), cache=cache, use_cache=True)
        nxt = out.logits[:, -1].argmax(-1)
        first_logits = out.logits[:, -1].clone()
        toks = [[] for _ in range(B)]; alive = [True] * B
        for _ in range(max_new):
            for b in range(B):
                if alive[b]:
                    t = int(nxt[b])
                    if t in self.eos: alive[b] = False; continue
                    toks[b].append(t)
                    if '\n' in self.tok.decode(toks[b]): alive[b] = False
            if not any(alive): break
            out = self.model(input_ids=nxt[:, None], cache=cache, use_cache=True)
            nxt = out.logits[:, -1].argmax(-1)
        del cache
        return [self.tok.decode(t, skip_special_tokens=True).strip() for t in toks], first_logits

    @torch.no_grad()
    def continue_greedy(self, pre, first_tok, n_suffix, max_new=16):
        """Continue greedy decoding after a suffix pass whose KV is still in the cache (first_tok already chosen)."""
        cache = pre['cache']; toks = []; nxt = first_tok
        for _ in range(max_new):
            if nxt in self.eos: break
            toks.append(nxt)
            if '\n' in self.tok.decode(toks): break
            out = self.model(input_ids=torch.tensor([[nxt]]), cache=cache, use_cache=True)
            nxt = int(out.logits[0, -1].argmax())
        self.reset(cache, pre['P'])
        return self.tok.decode(toks, skip_special_tokens=True).strip()
