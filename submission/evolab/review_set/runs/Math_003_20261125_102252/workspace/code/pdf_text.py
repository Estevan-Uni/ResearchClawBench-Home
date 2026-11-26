"""Crude offline PDF text extractor (FlateDecode streams, Tj/TJ operators)."""
import sys, re, zlib
def extract(path):
    d = open(path, 'rb').read()
    out = []
    for m in re.finditer(rb'stream\r?\n(.*?)\r?\nendstream', d, re.S):
        s = m.group(1)
        try: s = zlib.decompress(s)
        except Exception: continue
        if b'TJ' not in s and b'Tj' not in s: continue
        for t in re.finditer(rb'\[(.*?)\]\s*TJ|\((.*?)\)\s*Tj', s, re.S):
            if t.group(1) is not None:
                parts = re.findall(rb'\(((?:\\.|[^\\)])*)\)|(-?\d+\.?\d*)', t.group(1))
                txt = b''
                for p, num in parts:
                    if p: txt += p
                    elif num and float(num) < -200: txt += b' '
            else: txt = t.group(2)
            out.append(txt.decode('latin1'))
        out.append('\n')
    return ''.join(out)
if __name__ == '__main__':
    print(extract(sys.argv[1]))
