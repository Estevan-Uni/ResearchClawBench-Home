"""Resumable parallel HTTP-range downloader for HuggingFace files.
Usage: python download_hf.py <repo_id> <out_dir> [file ...]
"""
import os, sys, json, threading, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

CHUNK = 8 * 1024 * 1024
WORKERS = 24


def url_for(repo, f):
    return f"https://huggingface.co/{repo}/resolve/main/{f}"


def head_size(url):
    req = urllib.request.Request(url, method="HEAD")
    # follow redirects manually to get final size
    with urllib.request.urlopen(req, timeout=60) as r:
        return int(r.headers["Content-Length"]), r.geturl()


def fetch_range(url, start, end, tries=30):
    for t in range(tries):
        try:
            req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
            with urllib.request.urlopen(req, timeout=120) as r:
                data = r.read()
            if len(data) == end - start + 1:
                return data
        except Exception as e:
            time.sleep(min(2 + t, 20))
    raise RuntimeError(f"range {start}-{end} failed")


def download(repo, f, out_dir):
    url = url_for(repo, f)
    out = os.path.join(out_dir, f)
    size, final = head_size(url)
    if os.path.exists(out) and os.path.getsize(out) == size and not os.path.exists(out + ".log"):
        print(f"{f}: already done"); return
    if size < 2 * CHUNK:
        data = fetch_range(final, 0, size - 1)
        open(out, "wb").write(data); print(f"{f}: {size} bytes"); return
    logp = out + ".log"
    done = set()
    if os.path.exists(logp):
        done = {int(x) for x in open(logp).read().split()}
    if not os.path.exists(out):
        with open(out, "wb") as fh:
            fh.truncate(size)
    offsets = [o for o in range(0, size, CHUNK) if o not in done]
    lock = threading.Lock()
    fh = open(out, "r+b"); lg = open(logp, "a")
    t0 = time.time(); n0 = len(done); total = (size + CHUNK - 1) // CHUNK

    def work(o):
        e = min(o + CHUNK, size) - 1
        data = fetch_range(final, o, e)
        with lock:
            fh.seek(o); fh.write(data); fh.flush()
            lg.write(f"{o}\n"); lg.flush(); done.add(o)
            k = len(done)
            if k % 10 == 0:
                rate = (k - n0) * CHUNK / 1e6 / max(time.time() - t0, 1)
                print(f"{f}: {k}/{total} chunks, {rate:.1f} MB/s", flush=True)

    with ThreadPoolExecutor(WORKERS) as ex:
        list(ex.map(work, offsets))
    fh.close(); lg.close()
    if len(done) == total:
        os.remove(logp); print(f"{f}: complete")


if __name__ == "__main__":
    repo, out_dir, *files = sys.argv[1:]
    os.makedirs(out_dir, exist_ok=True)
    for f in files:
        download(repo, f, out_dir)
