"""Minimal PDF text extractor (zlib streams + Tj/TJ operators). Caps stream size."""
import re, zlib, sys
def extract(path, maxstreams=400, cap=3_000_000):
    data = open(path,'rb').read()
    out = []
    for i, m in enumerate(re.finditer(rb'stream\r?\n(.*?)\r?\nendstream', data, re.S)):
        if i > maxstreams*20: break
        raw = m.group(1)
        try:
            d = zlib.decompressobj().decompress(raw, cap)
        except Exception:
            continue
        if b'Tj' not in d and b'TJ' not in d: continue
        parts = []
        for tj in re.finditer(rb'\[(.*?)\]\s*TJ|\((.*?)\)\s*Tj', d, re.S):
            if tj.group(1) is not None:
                s = b''.join(re.findall(rb'\((.*?)(?<!\\)\)', tj.group(1), re.S))
                # large kerning negative numbers -> spaces
                parts.append(s)
            else:
                parts.append(tj.group(2))
        if parts:
            out.append(b' '.join(parts).decode('latin-1', 'ignore'))
    return '\n'.join(out)
if __name__ == '__main__':
    print(extract(sys.argv[1]))
