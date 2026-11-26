"""Minimal pure-python PDF text extractor (FlateDecode, object streams, ToUnicode CMaps).
Usage: python pdf_extract.py in.pdf out.txt
"""
import re, sys, zlib

def inflate(b):
    for wb in (zlib.MAX_WBITS, -zlib.MAX_WBITS):
        try:
            return zlib.decompressobj(wb).decompress(b)
        except Exception:
            pass
    return None

def parse_objects(data):
    objs = {}
    for m in re.finditer(rb'(\d+)\s+(\d+)\s+obj\b', data):
        num = int(m.group(1)); start = m.end()
        end = data.find(b'endobj', start)
        if end < 0: continue
        body = data[start:end]
        objs[num] = body
    # decode streams
    streams = {}
    for num, body in list(objs.items()):
        si = body.find(b'stream')
        if si < 0: continue
        d = body[:si]
        s = body[si+6:]
        if s.startswith(b'\r\n'): s = s[2:]
        elif s.startswith(b'\n') or s.startswith(b'\r'): s = s[1:]
        ei = s.rfind(b'endstream')
        s = s[:ei]
        if b'FlateDecode' in d:
            dec = inflate(s)
        elif b'Filter' not in d:
            dec = s
        else:
            dec = None
        if dec is not None:
            streams[num] = (d, dec)
    # object streams
    for num, (d, dec) in list(streams.items()):
        if b'/ObjStm' in d:
            mN = re.search(rb'/N\s+(\d+)', d); mF = re.search(rb'/First\s+(\d+)', d)
            if not (mN and mF): continue
            N = int(mN.group(1)); F = int(mF.group(1))
            hdr = dec[:F].split()
            offs = [(int(hdr[2*i]), int(hdr[2*i+1])) for i in range(min(N, len(hdr)//2))]
            for i, (on, oo) in enumerate(offs):
                e = offs[i+1][1] if i+1 < len(offs) else len(dec)-F
                objs.setdefault(on, dec[F+oo:F+e])
    return objs, streams

def parse_cmap(txt):
    cmap = {}
    for blk in re.findall(rb'beginbfchar(.*?)endbfchar', txt, re.S):
        for a, b in re.findall(rb'<([0-9A-Fa-f]+)>\s*<([0-9A-Fa-f]+)>', blk):
            cmap[int(a, 16)] = hexto_u(b)
    for blk in re.findall(rb'beginbfrange(.*?)endbfrange', txt, re.S):
        for a, b, c in re.findall(rb'<([0-9A-Fa-f]+)>\s*<([0-9A-Fa-f]+)>\s*(<[0-9A-Fa-f]+>|\[[^\]]*\])', blk):
            lo, hi = int(a, 16), int(b, 16)
            if c.startswith(b'['):
                items = re.findall(rb'<([0-9A-Fa-f]+)>', c)
                for k, it in enumerate(items):
                    cmap[lo+k] = hexto_u(it)
            else:
                base = int(c[1:-1], 16)
                for k in range(hi-lo+1):
                    try: cmap[lo+k] = chr(base+k)
                    except Exception: pass
    return cmap

def hexto_u(h):
    h = h.decode()
    if len(h) % 4: h = h.ljust((len(h)+3)//4*4, '0')
    try:
        return bytes.fromhex(h).decode('utf-16-be', errors='ignore')
    except Exception:
        return ''

def unescape(s):
    out = bytearray(); i = 0
    while i < len(s):
        c = s[i]
        if c == 0x5c and i+1 < len(s):
            n = s[i+1]
            mp = {ord('n'): 10, ord('r'): 13, ord('t'): 9, ord('b'): 8, ord('f'): 12, ord('('): 40, ord(')'): 41, 0x5c: 0x5c}
            if n in mp: out.append(mp[n]); i += 2; continue
            if 48 <= n <= 55:
                j = i+1; v = ''
                while j < len(s) and len(v) < 3 and 48 <= s[j] <= 55: v += chr(s[j]); j += 1
                out.append(int(v, 8) & 255); i = j; continue
            i += 2; continue
        out.append(c); i += 1
    return bytes(out)

def main(fn, out):
    data = open(fn, 'rb').read()
    objs, streams = parse_objects(data)
    # font name -> cmap, per page resources is complex; build global map fontobj->cmap
    font_cmap = {}
    for num, body in objs.items():
        m = re.search(rb'/ToUnicode\s+(\d+)\s+0\s+R', body)
        if m and int(m.group(1)) in streams:
            font_cmap[num] = parse_cmap(streams[int(m.group(1))][1])
    # resource name -> font obj (global, last wins; approximate)
    res_map = {}
    for num, body in objs.items():
        for fd in re.findall(rb'/Font\s*<<(.*?)>>', body, re.S):
            for name, ref in re.findall(rb'/([A-Za-z0-9_.+-]+)\s+(\d+)\s+0\s+R', fd):
                res_map.setdefault(name, int(ref))
        mfd = re.search(rb'/Font\s+(\d+)\s+0\s+R', body)
        if mfd and int(mfd.group(1)) in objs:
            for name, ref in re.findall(rb'/([A-Za-z0-9_.+-]+)\s+(\d+)\s+0\s+R', objs[int(mfd.group(1))]):
                res_map.setdefault(name, int(ref))
    texts = []
    tok = re.compile(rb'/([A-Za-z0-9_.+-]+)\s+[\d.]+\s+Tf|\[(.*?)\]\s*TJ|\((?:\\.|[^\\)])*\)\s*(?:Tj|\'|")|<([0-9A-Fa-f\s]+)>\s*Tj|(T\*|Td|TD|Tm|ET)', re.S)
    for num, (d, dec) in streams.items():
        if b'/ObjStm' in d or b'/XRef' in d: continue
        if not (b'Tj' in dec or b'TJ' in dec): continue
        cur = None; buf = []
        for m in tok.finditer(dec):
            if m.group(1):
                fo = res_map.get(m.group(1)); cur = font_cmap.get(fo)
                continue
            if m.group(4):
                if m.group(4) in (b'T*', b'Td', b'TD', b'Tm'):
                    buf.append(' ')
                else:
                    buf.append('\n')
                continue
            if m.group(2) is not None:
                arr = m.group(2)
                parts = re.findall(rb'\(((?:\\.|[^\\)])*)\)|<([0-9A-Fa-f\s]*)>|(-?[\d.]+)', arr, re.S)
                for lit, hx, numv in parts:
                    if numv:
                        try:
                            if float(numv) < -200: buf.append(' ')
                        except: pass
                    elif hx:
                        buf.append(decode_hex(hx, cur))
                    else:
                        buf.append(decode_lit(unescape(lit), cur))
            elif m.group(3) is not None:
                buf.append(decode_hex(m.group(3), cur))
            else:
                s = m.group(0); lit = s[1:s.rfind(b')')]
                buf.append(decode_lit(unescape(lit), cur))
        texts.append(''.join(buf))
    open(out, 'w').write('\n\n'.join(texts))
    print(fn, len(objs), 'objs', len(streams), 'streams', sum(len(t) for t in texts), 'chars')

def decode_hex(hx, cmap):
    hx = re.sub(rb'\s', b'', hx).decode()
    if cmap:
        w = 4 if max(cmap.keys(), default=0) > 255 else 2
        return ''.join(cmap.get(int(hx[i:i+w], 16), '') for i in range(0, len(hx), w))
    try: return bytes.fromhex(hx).decode('latin-1')
    except: return ''

def decode_lit(b, cmap):
    if cmap:
        if max(cmap.keys(), default=0) > 255:
            return ''.join(cmap.get(int.from_bytes(b[i:i+2], 'big'), '') for i in range(0, len(b)-1, 2))
        return ''.join(cmap.get(c, chr(c)) for c in b)
    return b.decode('latin-1')

if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
