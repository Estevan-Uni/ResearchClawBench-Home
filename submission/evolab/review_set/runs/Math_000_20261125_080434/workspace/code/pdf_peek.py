import re, zlib, sys
data = open(sys.argv[1], 'rb').read()
out = []
for m in re.finditer(rb'stream\r?\n(.*?)\r?\nendstream', data, re.S):
    try: s = zlib.decompress(m.group(1))
    except Exception: continue
    for t in re.findall(rb'\[(.*?)\]\s*TJ|\((.*?)\)\s*Tj', s, re.S):
        seg = t[0] or t[1]
        parts = re.findall(rb'\((.*?)(?<!\\)\)', seg) if t[0] else [seg]
        out.append(b''.join(parts).decode('latin1'))
print(' '.join(out)[:int(sys.argv[2]) if len(sys.argv)>2 else 3000])
