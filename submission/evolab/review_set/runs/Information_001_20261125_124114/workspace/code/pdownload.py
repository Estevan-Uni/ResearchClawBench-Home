"""Parallel chunked downloader for HF repos (per-connection bandwidth is throttled on this node).
usage: pdownload.py REPO OUTDIR [nthreads] [--dataset] [--include substr,substr]"""
import sys, os, json, time, threading, queue, urllib.request
repo, outdir = sys.argv[1], sys.argv[2]
nth = int(sys.argv[3]) if len(sys.argv) > 3 and sys.argv[3].isdigit() else 32
typ = 'datasets' if '--dataset' in sys.argv else 'models'
inc = sys.argv[sys.argv.index('--include')+1].split(',') if '--include' in sys.argv else None
base = f'https://huggingface.co/{"datasets/" if typ=="datasets" else ""}{repo}/resolve/main/'
def get_json(u):
    for i in range(20):
        try: return json.load(urllib.request.urlopen(u, timeout=60))
        except Exception as e: print('api retry', e, flush=True); time.sleep(5)
files = [f for f in get_json(f'https://huggingface.co/api/{typ}/{repo}/tree/main?recursive=1') if f['type']=='file']
files = [f for f in files if not f['path'].startswith('onnx/') and not f['path'].endswith(('.gguf','.onnx','.bin','.msgpack','.h5','.pth'))]
if inc: files = [f for f in files if any(s in f['path'] for s in inc)]
CH = 8 << 20
q = queue.Queue()
for f in files:
    p = os.path.join(outdir, f['path']); os.makedirs(os.path.dirname(p), exist_ok=True)
    size = f['size']; done_p = p + '.chunks'
    if os.path.exists(p) and not os.path.exists(done_p) and os.path.getsize(p) == size: continue
    if not os.path.exists(p) or os.path.getsize(p) != size:
        with open(p, 'wb') as fh: fh.truncate(size)
        open(done_p, 'w').close()
    done = set(int(x) for x in open(done_p).read().split()) if os.path.exists(done_p) else set()
    for s in range(0, size, CH):
        if s not in done: q.put((f['path'], p, s, min(size, s+CH)-1, done_p))
print('chunks to fetch', q.qsize(), 'files', len(files), flush=True)
lock = threading.Lock(); total = [0]; t0 = time.time()
def worker():
    while True:
        try: path, p, s, e, done_p = q.get_nowait()
        except queue.Empty: return
        pos = s; fails = 0
        with open(p, 'r+b') as fh:
            while pos <= e and fails < 200:
                try:
                    req = urllib.request.Request(base + path, headers={'Range': f'bytes={pos}-{e}'})
                    r = urllib.request.urlopen(req, timeout=30)
                    while pos <= e:
                        b = r.read(1 << 16)
                        if not b: break
                        fh.seek(pos); fh.write(b); pos += len(b)
                        with lock: total[0] += len(b)
                except Exception as ex:
                    fails += 1; time.sleep(1)
        if pos > e:
            with lock:
                with open(done_p, 'a') as fh: fh.write(f'{s}\n')
        else:
            q.put((path, p, s, e, done_p))
th = [threading.Thread(target=worker, daemon=True) for _ in range(nth)]
[t.start() for t in th]
while any(t.is_alive() for t in th):
    time.sleep(30); print(f'{total[0]/1e9:.2f} GB  {total[0]/1e6/(time.time()-t0):.1f} MB/s  left~{q.qsize()}', flush=True)
# verify
ok = True
for f in files:
    p = os.path.join(outdir, f['path']); done_p = p + '.chunks'
    if os.path.exists(done_p):
        n = len(set(open(done_p).read().split())); need = (f['size'] + CH - 1)//CH
        if n == need: os.remove(done_p)
        else: ok = False; print('INCOMPLETE', p, n, need, flush=True)
print('ALLDONE' if ok else 'PARTIAL', flush=True)
