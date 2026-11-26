"""Qwen2.5-VL ViCrop engine (CPU, fp32). Full forwards (M-RoPE positions depend on the layout).

The global image AND the crop are both resized to a fixed pixel budget (MAX_PIXELS, default 448x448), which
emulates the limited-resolution bottleneck. Token grid of an image = (grid_h/2, grid_w/2), 28 px per token in the
smart-resized image.
"""
import math, torch, numpy as np
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from vicrop_common import GENERIC_Q, SHORT_ANS


class QwenEngine:
    def __init__(self, path, max_pixels=448 * 448, threads=None):
        if threads: torch.set_num_threads(threads)
        self.proc = AutoProcessor.from_pretrained(path)
        self.max_pixels = max_pixels
        self.proc.image_processor.size = {'shortest_edge': 56 * 56, 'longest_edge': max_pixels}
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(path, dtype=torch.float32,
                                                                        attn_implementation='sdpa')
        self.model.eval()
        for p in self.model.parameters(): p.requires_grad_(False)
        self.tok = self.proc.tokenizer
        self.img_tok = self.model.config.image_token_id
        self.n_layers = self.model.config.text_config.num_hidden_layers
        self.eos = {self.tok.convert_tokens_to_ids('<|im_end|>'), self.tok.convert_tokens_to_ids('<|endoftext|>')}
        self.bbox_size = int(math.sqrt(max_pixels))

    def prompt(self, text, n_images):
        content = [{'type': 'image'} for _ in range(n_images)] + [{'type': 'text', 'text': text}]
        return self.proc.apply_chat_template([{'role': 'user', 'content': content}], tokenize=False,
                                             add_generation_prompt=True)

    def p_base(self, q): return self.prompt(f"{q}\n{SHORT_ANS}", 1)
    def p_generic(self): return self.prompt(GENERIC_Q, 1)
    def p_crop(self, q): return self.prompt(f"{q}\n{SHORT_ANS}", 2)

    def upscale_to_budget(self, crop):
        """Resize a crop so that its area equals the pixel budget (zoom-in), keeping aspect ratio."""
        w, h = crop.size
        s = math.sqrt(self.max_pixels / (w * h))
        return crop.resize((max(28, int(round(w * s))), max(28, int(round(h * s)))), Image.BICUBIC)

    def inputs(self, images, prompt):
        return self.proc(text=[prompt], images=images, return_tensors='pt')

    def forward_attn(self, img, prompt, want_grad=False, grad_layers=None):
        """Full forward with eager attention. Returns head-mean attention (L, n_img_tokens) of the last token to
        the image tokens, grid shape (th, tw), first-token logits, grad-att maps (band, n) and pure-grad (n,)."""
        inp = self.inputs([img], prompt)
        thw = inp['image_grid_thw'][0]
        th, tw = int(thw[1]) // 2, int(thw[2]) // 2
        ip = (inp['input_ids'][0] == self.img_tok).nonzero().flatten()
        assert len(ip) == th * tw
        self.model.model.rope_deltas = None
        self.model.set_attn_implementation('eager')
        try:
            if want_grad:
                inp['pixel_values'] = inp['pixel_values'].clone().requires_grad_(True)
            with torch.set_grad_enabled(want_grad):
                out = self.model(**inp, use_cache=False, output_attentions=True)
                atts = out.attentions
                logits = out.logits[0, -1]
                A = torch.stack([a[0, :, -1, ip].mean(0) for a in atts]).detach().float()
                G = PG = None
                if want_grad:
                    for l in grad_layers: atts[l].retain_grad()
                    tok = int(logits.argmax())
                    torch.log_softmax(logits, -1)[tok].backward()
                    G = torch.stack([(atts[l][0, :, -1, ip] * atts[l].grad[0, :, -1, ip].clamp(min=0)).mean(0)
                                     for l in grad_layers]).detach()
                    g = inp['pixel_values'].grad.abs().sum(-1)  # (n_patches,) ordered by merged token (4 each)
                    PG = g.reshape(-1, 4).sum(-1).detach()
        finally:
            self.model.set_attn_implementation('sdpa')
            self.model.model.rope_deltas = None
        return A, (th, tw), logits.detach(), G, PG

    @torch.no_grad()
    def greedy(self, images, prompt, max_new=16):
        inp = self.inputs(images, prompt)
        self.model.model.rope_deltas = None
        out = self.model(**inp, use_cache=True)
        cache = out.past_key_values
        first_logits = out.logits[0, -1].clone()
        nxt = int(out.logits[0, -1].argmax())
        toks = []
        for _ in range(max_new):
            if nxt in self.eos: break
            toks.append(nxt)
            if '\n' in self.tok.decode(toks): break
            out = self.model(input_ids=torch.tensor([[nxt]]), cache=cache, use_cache=True)
            nxt = int(out.logits[0, -1].argmax())
        self.model.model.rope_deltas = None
        return self.tok.decode(toks, skip_special_tokens=True).strip(), first_logits
