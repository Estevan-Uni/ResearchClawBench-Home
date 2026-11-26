"""Tiny job queue: python queue.py jobs.txt NWORKERS. Each line = args for dyn.py or static.py.
Skips jobs whose result json exists (handled inside the scripts)."""
import sys, subprocess, os
from concurrent.futures import ThreadPoolExecutor

jobs = [l.strip() for l in open(sys.argv[1]) if l.strip() and not l.startswith('#')]
nw = int(sys.argv[2])
os.makedirs('/workspace/outputs/logs', exist_ok=True)


def run(j):
    name = j.replace(' ', '').replace('--', '_').replace('/', '')[:150]
    with open(f'/workspace/outputs/logs/{name}.log', 'w') as f:
        subprocess.run(f'cd /workspace && OMP_NUM_THREADS=2 python {j}', shell=True, stdout=f, stderr=subprocess.STDOUT)
    return j


with ThreadPoolExecutor(nw) as ex:
    for j in ex.map(run, jobs):
        print('finished', j, flush=True)
