"""Shared loader for the released Janus-1.3B checkpoint on CPU (fp32, SDPA attention)."""
import os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(HERE, "shims"))
sys.path.insert(0, os.path.join(WS, "outputs/ext/janus_src/Janus-main"))
MODEL = os.path.join(WS, "outputs/ext/janus_weights")
import torch
torch.set_num_threads(os.cpu_count())
from transformers import AutoConfig, AutoModelForCausalLM
from janus.models import MultiModalityCausalLM, VLChatProcessor


def load():
    t = time.time()
    proc = VLChatProcessor.from_pretrained(MODEL)
    cfg = AutoConfig.from_pretrained(MODEL, trust_remote_code=True)
    cfg.language_config._attn_implementation = "sdpa"
    model = AutoModelForCausalLM.from_pretrained(MODEL, config=cfg, trust_remote_code=True,
                                                 torch_dtype=torch.float32).eval()
    print(f"loaded in {time.time()-t:.1f}s", flush=True)
    return proc, model
