"""Verify downloaded LFS files against the sha256 oids published by the HF API. usage: verify_sha.py REPO DIR"""
import sys, json, hashlib, os, urllib.request
repo, d = sys.argv[1], sys.argv[2]
files = json.load(urllib.request.urlopen(f'https://huggingface.co/api/models/{repo}/tree/main?recursive=1', timeout=60))
ok = True
for f in files:
    if 'lfs' not in f or not f['path'].endswith('.safetensors'): continue
    p = os.path.join(d, f['path']); h = hashlib.sha256()
    with open(p, 'rb') as fh:
        for b in iter(lambda: fh.read(1 << 24), b''): h.update(b)
    good = h.hexdigest() == f['lfs']['oid']; ok &= good
    print(f['path'], 'OK' if good else 'MISMATCH', flush=True)
print('ALL_OK' if ok else 'FAIL')
