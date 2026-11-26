"""T2I figures: Janus-deity attempts, other showcase prompt, CFG sweep."""
import json, os, glob, textwrap
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from PIL import Image
T = "/workspace/outputs/t2i"; I = "/workspace/report/images"
logs = {j["tag"]: j for j in map(json.loads, open(f"{T}/log.jsonl"))}
def grid(tags, fname, title, wrap=95):
    items = [(t, f) for t in tags if t in logs for f in sorted(glob.glob(f"{T}/{t}_[0-9].png"))]
    if not items: return
    fig, ax = plt.subplots(1, len(items), figsize=(4.2 * len(items), 5.0))
    ax = [ax] if len(items) == 1 else ax
    for a, (t, f) in zip(ax, items):
        j = logs[t]; a.imshow(Image.open(f)); a.axis("off")
        a.set_title(f"{os.path.basename(f)[:-4]}\nseed={j.get('seed',0)}, CFG={j.get('cfg',5)}, {j['sec']/60:.0f} min/batch", fontsize=9)
    ps = []
    for t in tags:
        if t in logs and logs[t]["prompt"] not in ps: ps.append(logs[t]["prompt"])
    fig.suptitle(title + "\n" + "\n".join("Prompt: " + "\n".join(textwrap.wrap(p, wrap)) for p in ps), fontsize=8.5, y=0.99, ha="center")
    plt.tight_layout(rect=(0, 0, 1, 0.78 if len(ps) > 1 else 0.90)); plt.savefig(f"{I}/{fname}", dpi=130); plt.close()
grid(["janus_s0"], "fig5_t2i_janus_attempt1.png", "Attempt 1 — Janus-1.3B text-to-image (n=2, 384×384)", 110)
grid(["janus_v2_s1"], "fig6_t2i_janus_attempt2.png", "Attempt 2 — rephrased Janus prompt (n=2, 384×384)", 110)
grid(["readme_opera_s0"], "fig9_t2i_readme_showcase.png", "Official-README showcase prompt (n=2)", 110)
grid(["janus_cfg1_s0", "janus_cfg3_s0", "janus_s0", "janus_cfg10_s0"], "fig10_t2i_cfg_sweep.png", "CFG sweep on the Janus prompt (attempt-1 prompt)", 150)
print(sorted(logs))
