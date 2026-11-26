import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import torch
torch.set_num_threads(1)
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MODEL = os.path.join(ROOT, "outputs", "model", "mace-mp-0b3-medium.model")
OUT = os.path.join(ROOT, "outputs")
IMG = os.path.join(ROOT, "report", "images")
def get_calc(dtype="float64", model_path=MODEL):
    from mace.calculators import MACECalculator
    return MACECalculator(model_paths=model_path, device="cpu", default_dtype=dtype)
