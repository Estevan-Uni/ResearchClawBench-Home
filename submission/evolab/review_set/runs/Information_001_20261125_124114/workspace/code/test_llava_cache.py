"""Exactness check of the KV-prefix reuse in the LLaVA engine: cached prefix + suffix vs one full forward."""
import sys, os, torch, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from PIL import Image
from engine_llava import LlavaEngine
torch.set_num_threads(int(sys.argv[1]) if len(sys.argv) > 1 else 16)
eng = LlavaEngine('/workspace/outputs/ext/llava15_7b')
img = Image.open('/workspace/data/demo_imgs/demo1.png').convert('RGB')
q = 'What is the date printed in the bottom-right corner of the photo?'
t = time.time(); pre = eng.build_prefix(img, q); tp = time.time() - t
t = time.time(); A, lg, _ = eng.suffix_attn(pre, eng.p_base(q)); ts = time.time() - t
ids = eng.ids(eng.p_base(q), 1)
eng.model.set_attn_implementation('eager')
t = time.time()
with torch.no_grad():
    out = eng.model(input_ids=ids, pixel_values=pre['pv'], output_attentions=True)
tf = time.time() - t
eng.model.set_attn_implementation('sdpa')
Af = torch.stack([a[0, :, -1, pre['img_pos']].mean(0) for a in out.attentions])
res = dict(max_logit_diff=float((out.logits[0, -1] - lg).abs().max()), max_att_diff=float((Af - A).abs().max()),
           t_prefix=tp, t_suffix=ts, t_full_eager=tf, seq_len=int(ids.shape[1]))
t = time.time(); ans, _ = eng.greedy(pre, eng.p_base(q)); res['t_greedy'] = time.time() - t; res['answer'] = ans
box = (700, 500, 1024, 768); pv, _ = eng.crop_pv(img, box)
t = time.time(); ans2, _ = eng.greedy(pre, eng.p_crop(q), n_images=2, crop_pv=pv); res['t_crop_greedy'] = time.time() - t
res['answer_crop'] = ans2
# padded-preprocessing check: save de-normalised pixel_values
import numpy as np
m = np.array(eng.proc.image_processor.image_mean)[:, None, None]; s = np.array(eng.proc.image_processor.image_std)[:, None, None]
x = (pre['pv'][0].numpy() * s + m).clip(0, 1).transpose(1, 2, 0)
Image.fromarray((x * 255).astype('uint8')).save('/workspace/outputs/demo/llava_input_demo1_padded.png')
print(json.dumps(res, indent=1)); json.dump(res, open('/workspace/outputs/llava_cache_exactness.json', 'w'), indent=1)
