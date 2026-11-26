"""Parallel range download of a single URL with a chunk queue. usage: pfetch_url.py URL OUT [nthreads]"""
import sys, urllib.request, threading, time, queue
url, out = sys.argv[1], sys.argv[2]; N = int(sys.argv[3]) if len(sys.argv) > 3 else 16
for a in range(20):
    try:
        r = urllib.request.urlopen(urllib.request.Request(url, headers={'Range': 'bytes=0-0'}), timeout=30)
        size = int(r.headers['Content-Range'].split('/')[-1]); break
    except Exception as e: print('size retry', e, flush=True); time.sleep(3)
CH = 1 << 20; q = queue.Queue(); buf = {}
for s in range(0, size, CH): q.put(s)
print('size', size, 'chunks', q.qsize(), flush=True)
def w():
    while True:
        try: s = q.get_nowait()
        except queue.Empty: return
        e = min(size, s + CH) - 1
        for a in range(100):
            try:
                d = urllib.request.urlopen(urllib.request.Request(url, headers={'Range': f'bytes={s}-{e}'}), timeout=30).read()
                if len(d) == e - s + 1: buf[s] = d; break
            except Exception: time.sleep(1)
th = [threading.Thread(target=w, daemon=True) for _ in range(N)]; [t.start() for t in th]
while any(t.is_alive() for t in th): time.sleep(10); print(len(buf), flush=True)
open(out, 'wb').write(b''.join(buf[s] for s in sorted(buf))); print('DONE', size, sum(len(v) for v in buf.values()), flush=True)
