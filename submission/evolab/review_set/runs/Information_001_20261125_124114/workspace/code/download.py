import sys, time
from huggingface_hub import snapshot_download
repo, local, *pat = sys.argv[1:]
kw = dict(allow_patterns=pat) if pat else {}
for i in range(30):
    try:
        p = snapshot_download(repo, local_dir=local, repo_type='dataset' if repo.startswith('ds:') else 'model', **kw) if False else \
            snapshot_download(repo.replace('ds:',''), local_dir=local, repo_type='dataset' if repo.startswith('ds:') else None, **kw)
        print('DONE', p, flush=True); break
    except Exception as e:
        print('retry', i, repr(e)[:300], flush=True); time.sleep(20)
