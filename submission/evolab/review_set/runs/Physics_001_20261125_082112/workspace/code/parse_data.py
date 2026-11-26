"""Parse the MATBG superfluid stiffness text dataset into numpy arrays."""
import re, numpy as np, os
DATA = os.path.join(os.path.dirname(__file__), '..', 'data', 'MATBG Superfluid Stiffness Core Dataset.txt')

def parse(path=DATA):
    txt = open(path).read()
    files = re.split(r'\*\*File \d+: (.+?)\*\*', txt)
    out = {}; params = {}
    for i in range(1, len(files), 2):
        fname, body = files[i], files[i+1]
        parts = re.split(r'\*\*(.+?):\*\*', body)
        params[fname] = {}
        for j in range(1, len(parts), 2):
            head, content = parts[j], parts[j+1]
            if head.startswith('Fixed Parameters'):
                for line in content.strip().splitlines():
                    if '=' in line:
                        k, v = [s.strip() for s in line.split('=', 1)]
                        params[fname][k] = eval(v, {}, {})
                continue
            m = re.search(r'\(([^)]+)\)', head)
            key = m.group(1).split(' ')[0] if m else head
            arr = re.search(r'\[(.*?)\]', content, re.S).group(1)
            out[key] = np.array([float(x) for x in arr.split()])
    return out, params

if __name__ == '__main__':
    d, p = parse()
    for k, v in d.items(): print(f'{k:22s} len={len(v):4d} first={v[0]:.5g} last={v[-1]:.5g}')
    print(p)
