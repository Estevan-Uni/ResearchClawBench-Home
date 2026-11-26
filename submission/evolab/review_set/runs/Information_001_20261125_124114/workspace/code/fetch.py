"""Robust resumable HTTP downloader (Range-based). usage: fetch.py URL OUT"""
import sys, os, time, urllib.request
url, out = sys.argv[1], sys.argv[2]
for attempt in range(200):
    have = os.path.getsize(out) if os.path.exists(out) else 0
    try:
        req = urllib.request.Request(url, headers={'Range': f'bytes={have}-'} if have else {})
        r = urllib.request.urlopen(req, timeout=60)
        total = r.headers.get('Content-Range', '').split('/')[-1] or r.headers.get('Content-Length')
        if r.status == 200 and have: have = 0; open(out, 'wb').close()
        with open(out, 'ab') as f:
            while True:
                b = r.read(1 << 16)
                if not b: break
                f.write(b)
        print('DONE', out, os.path.getsize(out), total, flush=True); break
    except urllib.error.HTTPError as e:
        if e.code == 416: print('DONE(416)', out, flush=True); break
        print('err', attempt, e, flush=True); time.sleep(5)
    except Exception as e:
        print('err', attempt, repr(e)[:200], flush=True); time.sleep(3)
