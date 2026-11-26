import sys, urllib.request, concurrent.futures as cf, os
url, out = sys.argv[1], sys.argv[2]; n = int(sys.argv[3]) if len(sys.argv)>3 else 16
req = urllib.request.Request(url, method="HEAD")
with urllib.request.urlopen(req, timeout=60) as r:
    size = int(r.headers["Content-Length"]); final = r.geturl()
print("size", size, flush=True)
chunk = size // n + 1
def get(i):
    s, e = i*chunk, min(size-1, (i+1)*chunk-1)
    for attempt in range(8):
        try:
            rq = urllib.request.Request(final, headers={"Range": f"bytes={s}-{e}"})
            with urllib.request.urlopen(rq, timeout=120) as r: data = r.read()
            if len(data) == e-s+1: return i, data
        except Exception as ex: print("retry", i, ex, flush=True)
    raise RuntimeError(i)
parts = {}
with cf.ThreadPoolExecutor(n) as ex:
    for i, d in ex.map(get, range(n)): parts[i] = d
with open(out, "wb") as f:
    for i in range(n): f.write(parts[i])
print("done", os.path.getsize(out))
