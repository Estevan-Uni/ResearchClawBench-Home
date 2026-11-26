"""Turn corpus-A 'aux needed' theorems into supervised examples for the prior
(AlphaGeometry-style traceback labels): the reduced statement (closure of the
conclusion) is the problem, the auxiliary point used by the minimal proof is the
positive candidate (matched by coordinates inside our candidate pool), all other
pool candidates are negatives.  Only single-aux-point examples are used."""
import glob
import json
import os
import sys

import numpy as np

from geo.problem import Problem
from geo.aux import generate, features


def main(src_glob, out, quota=30):
    Fs, ys, gs, fams = [], [], [], []
    n_ex = n_match = 0
    gid = 0
    for f in sorted(glob.glob(src_glob)):
        for line in open(f):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            for t in r['theorems']:
                ex = t.get('aux_example')
                if not ex or len(ex['aux_pts']) != 1:
                    continue
                n_ex += 1
                coords = {k: complex(*v) for k, v in ex['coords'].items()}
                sub = Problem('sub', ex['statement'] + ' ? ' + ex['goal'])
                Psub = {k: coords[k] for k in sub.points}
                za = coords[ex['aux_pts'][0]]
                try:
                    C = generate(sub, Psub, quota=quota, seed=gid)
                except Exception as e:
                    print('gen error', repr(e), file=sys.stderr)
                    continue
                y = np.array([abs(c['z'] - za) < 1e-6 for c in C], np.int8)
                if y.sum() == 0:
                    # positive not representable in a quota-capped pool: add it from the full pool
                    Cfull = generate(sub, Psub, quota=None)
                    pos = [c for c in Cfull if abs(c['z'] - za) < 1e-6]
                    if not pos:
                        continue
                    C = C + pos[:1]
                    y = np.array([abs(c['z'] - za) < 1e-6 for c in C], np.int8)
                n_match += 1
                Fs.append(features(C, sub, Psub)); ys.append(y); gs.append(np.full(len(y), gid))
                fams.append([c['family'] for c, yy in zip(C, y) if yy][0])
                gid += 1
    print('aux examples', n_ex, 'representable in candidate space', n_match)
    if Fs:
        np.savez_compressed(out, F=np.vstack(Fs), y=np.concatenate(ys), g=np.concatenate(gs))
        json.dump(dict(n_examples=n_ex, n_representable=n_match, pos_families=fams),
                  open(out.replace('.npz', '_meta.json'), 'w'))


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
