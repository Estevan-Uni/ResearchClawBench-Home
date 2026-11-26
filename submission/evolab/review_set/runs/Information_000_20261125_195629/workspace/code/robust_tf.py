"""Deterministic image degradations for the OCR robustness sweep."""
import numpy as np
from PIL import Image, ImageFilter

def apply(img, tf):
    kind, v = tf["kind"], tf["v"]
    if kind == "scale":
        w, h = img.size; s = img.resize((max(8, int(w * v)), max(8, int(h * v))), Image.BICUBIC)
        return s
    if kind == "blur":
        return img.filter(ImageFilter.GaussianBlur(v))
    if kind == "noise":
        rng = np.random.default_rng(0); a = np.asarray(img).astype(np.float32)
        return Image.fromarray(np.clip(a + rng.normal(0, v, a.shape), 0, 255).astype(np.uint8))
    if kind == "rotate":
        return img.rotate(v, resample=Image.BICUBIC, expand=True, fillcolor=(255, 255, 255))
    if kind == "jpeg":
        import io; b = io.BytesIO(); img.save(b, "JPEG", quality=int(v)); return Image.open(b).convert("RGB")
    raise ValueError(kind)
