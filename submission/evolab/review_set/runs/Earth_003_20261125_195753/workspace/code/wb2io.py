"""Minimal reader for WeatherBench2 zarr-v2 stores on GCS over HTTPS (no zarr/xarray needed)."""
import json, urllib.request, time, numpy as np
from concurrent.futures import ThreadPoolExecutor
from numcodecs import Blosc
import numcodecs
BASE = "https://storage.googleapis.com/weatherbench2/datasets/"

import os, hashlib
CACHE = '/workspace/outputs/cache'
STATS = {'bytes': 0, 'n': 0, 'retries': 0}

def _read_deadline(url, deadline=90):
    t0 = time.time()
    r = urllib.request.urlopen(url, timeout=20)
    buf = []
    while True:
        b = r.read(1 << 18)
        if not b: break
        buf.append(b)
        if time.time() - t0 > deadline: raise TimeoutError('slow')
    return b''.join(buf)

def _get(url, tries=20, cache=True):
    fn = None
    if cache and '/.z' not in url:
        fn = os.path.join(CACHE, url.split('datasets/')[1])
        if os.path.exists(fn): return open(fn, 'rb').read()
    for i in range(tries):
        try:
            d = _read_deadline(url)
            STATS['bytes'] += len(d); STATS['n'] += 1
            if fn:
                os.makedirs(os.path.dirname(fn), exist_ok=True)
                open(fn + '.tmp', 'wb').write(d); os.rename(fn + '.tmp', fn)
            return d
        except urllib.error.HTTPError as e:
            if e.code == 404: return None
            STATS['retries'] += 1; time.sleep(min(1 + i, 10))
        except Exception:
            STATS['retries'] += 1; time.sleep(min(1 + i, 10))
    raise RuntimeError(url)

class Store:
    def __init__(self, path):
        self.path = path
        self.meta = json.loads(_get(BASE + path + '/.zmetadata'))['metadata']
    def zarray(self, var): return self.meta[var + '/.zarray']
    def read_chunk(self, var, idx):
        za = self.zarray(var)
        key = var + '/' + '.'.join(str(i) for i in idx)
        raw = _get(BASE + self.path + '/' + key)
        shape = za['chunks']
        if raw is None:
            return np.full(shape, np.nan, dtype=za['dtype'])
        codec = numcodecs.get_codec(za['compressor'])
        buf = codec.decode(raw)
        return np.frombuffer(buf, dtype=za['dtype']).reshape(shape, order=za.get('order', 'C')).copy()
    def coord(self, var):
        za = self.zarray(var)
        n = za['shape'][0]; c = za['chunks'][0]
        return np.concatenate([self.read_chunk(var, (i,)) for i in range((n + c - 1) // c)])[:n]
    def read_many(self, var, idx_list, workers=32):
        t = time.time()
        with ThreadPoolExecutor(workers) as ex:
            out = list(ex.map(lambda ix: self.read_chunk(var, ix), idx_list))
        print(f'  {self.path.split("/")[0]} {var} {len(idx_list)} chunks {time.time()-t:.0f}s  total {STATS["bytes"]/1e6:.0f}MB n={STATS["n"]} retries={STATS["retries"]}', flush=True)
        return out
