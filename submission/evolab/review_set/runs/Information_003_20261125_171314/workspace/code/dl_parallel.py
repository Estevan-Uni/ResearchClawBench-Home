"""Parallel HTTP-range downloader for the LLM weights (HF hub single-stream is ~0.2 MB/s here)."""
import sys, os, urllib.request, json
from concurrent.futures import ThreadPoolExecutor

repo = sys.argv[1]; dest = sys.argv[2]; nconn = int(sys.argv[3]) if len(sys.argv) > 3 else 32
os.makedirs(dest, exist_ok=True)
base = f'https://modelscope.cn/models/{repo}/resolve/master/'
files = ['config.json', 'generation_config.json', 'tokenizer.json', 'tokenizer_config.json', 'vocab.json', 'merges.txt', 'model.safetensors']


def size_of(url):
    for att in range(10):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={'Range': 'bytes=0-0'}), timeout=60) as r:
                return int(r.headers['Content-Range'].split('/')[-1])
        except Exception as e:
            print('retry size', e, flush=True)


def get_range(url, a, b, path):
    for attempt in range(20):
        try:
            req = urllib.request.Request(url, headers={'Range': f'bytes={a}-{b}'})
            with urllib.request.urlopen(req, timeout=120) as r:
                data = r.read()
            if len(data) == b - a + 1:
                with open(path, 'r+b') as f:
                    f.seek(a); f.write(data)
                return
        except Exception as e:
            print('retry', a, e, flush=True)
    raise RuntimeError('failed chunk')


for fn in files:
    url = base + fn; path = os.path.join(dest, fn)
    if os.path.exists(path + '.done'):
        continue
    if fn != 'model.safetensors':
        for att in range(10):
            try:
                urllib.request.urlretrieve(url, path); break
            except Exception as e:
                print('retry small', fn, e, flush=True)
        open(path + '.done', 'w').close(); continue
    n = size_of(url); print('size', n, flush=True)
    with open(path, 'wb') as f:
        f.truncate(n)
    ch = 8 * 1024 * 1024
    jobs = [(a, min(a + ch, n) - 1) for a in range(0, n, ch)]
    with ThreadPoolExecutor(nconn) as ex:
        for k, _ in enumerate(ex.map(lambda ab: get_range(url, ab[0], ab[1], path), jobs)):
            if k % 20 == 0: print(f'{k}/{len(jobs)}', flush=True)
    open(path + '.done', 'w').close()
print('ALLDONE')
