"""CLIPScore (Hessel et al. 2021 style: 100*max(cos,0)) with openai/clip-vit-base-patch32 for generated images."""
import json, os, sys, glob, torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
C = "/workspace/outputs/ext/clip"; T = "/workspace/outputs/t2i"
m = CLIPModel.from_pretrained(C).eval(); p = CLIPProcessor.from_pretrained(C)
logs = [json.loads(l) for l in open(f"{T}/log.jsonl")]
extra = {"neg_control": "A photo of a red sports car parked on a city street."}
res = []
with torch.no_grad():
    for j in logs:
        for f in sorted(glob.glob(f"{T}/{j['tag']}_[0-9].png")):
            im = Image.open(f).convert("RGB")
            for kind, txt in [("own", j["prompt"]), ("short", "Janus, the two-faced Roman god with an old face and a young face" if "janus" in j["tag"] else j["prompt"]), ("neg", extra["neg_control"])]:
                x = p(text=[txt], images=im, return_tensors="pt", padding=True, truncation=True, max_length=77)
                o = m(**x)
                cos = torch.nn.functional.cosine_similarity(o.image_embeds, o.text_embeds).item()
                res.append({"tag": j["tag"], "file": os.path.basename(f), "text": kind, "clipscore": 100 * max(cos, 0)})
json.dump(res, open("/workspace/outputs/clipscore.json", "w"), indent=1)
for r in res: print(r)
