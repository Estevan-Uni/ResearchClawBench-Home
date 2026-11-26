"""Assemble report/report.md from section files + computed tables (CLIPScore, extra T2I)."""
import json, os, collections
R = "/workspace/code/report_src"; OUT = "/workspace/report"; O = "/workspace/outputs"
def clip_block():
    if not os.path.exists(f"{O}/clipscore.json"): return ""
    C = json.load(open(f"{O}/clipscore.json")); d = collections.defaultdict(dict)
    for r in C: d[r["file"]][r["text"]] = r["clipscore"]
    rows = "\n".join(f"| `{f}` | {v['own']:.1f} | {v['short']:.1f} | {v['neg']:.1f} |" for f, v in sorted(d.items()))
    return f"""### 4.3 CLIPScore (supporting evidence only)
We computed CLIPScore = 100·max(cos, 0) with OpenAI CLIP ViT-B/32 (n = 1 image per row) for three texts:
- **own:** the full generation prompt, truncated to 77 tokens;
- **short:** "Janus, the two-faced Roman god with an old face and a young face" for the Janus images, the own prompt otherwise;
- **neg:** an unrelated control, "A photo of a red sports car parked on a city street."

| Image | own prompt | short Janus caption | unrelated control |
|---|---|---|---|
{rows}

Every image scores 2.3–3.4× higher on its own prompt than on the control, so the generations are prompt-relevant. CLIPScore is, however, **blind to the compositional errors** identified above. The single-face sample `janus_s0_0` has the *highest* own-prompt score of the Janus images, and the short "old face and young face" caption separates samples by only about 4 points. This is why we rely on the explicit visual inspection in §4.1.
"""
def extra_block():
    L = {j["tag"]: j for j in map(json.loads, open(f"{O}/t2i/log.jsonl"))}
    s = ""
    if "readme_opera_s0" in L:
        s += open(f"{R}/sections_T_readme.md").read() if os.path.exists(f"{R}/sections_T_readme.md") else ""
    if "janus_cfg1_s0" in L or "janus_cfg10_s0" in L:
        s += open(f"{R}/sections_T_cfg.md").read() if os.path.exists(f"{R}/sections_T_cfg.md") else ""
    if not s:
        s = "*(Additional T2I runs (the official-README showcase prompt and a CFG sweep) were still running on the CPU when this report was compiled; see `outputs/t2i/`.)*\n"
    return s
T = open(f"{R}/sections_T.md").read().replace("CLIP_PLACEHOLDER", clip_block()).replace("EXTRA_T2I_PLACEHOLDER", extra_block())
parts = [open(f"{R}/header.md").read(), open(f"{R}/sections_A.md").read(), open(f"{R}/sections_B.md").read(),
         open(f"{R}/sections_D.md").read(), T, open(f"{R}/sections_C.md").read(), open(f"{R}/sections_E.md").read(), open(f"{R}/appendix.md").read()]
open(f"{OUT}/report.md", "w").write("\n\n".join(parts))
import re
txt = open(f"{OUT}/report.md").read()
for p in re.findall(r"\]\((images/[^)]+)\)", txt):
    print(p, os.path.exists(f"{OUT}/{p}"))
print("placeholders left:", txt.count("PLACEHOLDER"))
