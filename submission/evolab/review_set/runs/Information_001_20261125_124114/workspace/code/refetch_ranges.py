"""Re-fetch specific 8MB chunks of a file in parallel. usage: refetch_ranges.py URL FILE idx,idx,..."""
import sys, os, urllib.request, threading, time
url, p, idxs = sys.argv[1], sys.argv[2], [int(x) for x in sys.argv[3].split(',')]
CH = 8 << 20; size = os.path.getsize(p); lock = threading.Lock()
def job(s, e):
    pos = s
    while pos <= e:
        try:
            r = urllib.request.urlopen(urllib.request.Request(url, headers={'Range': f'bytes={pos}-{e}'}), timeout=30)
            while pos <= e:
                b = r.read(1 << 16)
                if not b: break
                with lock:
                    with open(p, 'r+b') as fh: fh.seek(pos); fh.write(b)
                pos += len(b)
        except Exception: time.sleep(1)
th = []
for i in idxs:  # split each 8MB chunk into 8 sub-ranges for speed
    for k in range(8):
        s = i * CH + k * (CH // 8); e = min(size, s + CH // 8) - 1
        if s < size: th.append(threading.Thread(target=job, args=(s, e)))
[t.start() for t in th]; [t.join() for t in th]; print('DONE')
