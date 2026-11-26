"""Mutation controls for the independent checker.

For every proof: (1) perturb a statement premise (rename a point in a premise
step), (2) forge a premise (insert a numerically-true non-statement fact as
'premise' and cite it), (3) delete a cited step, (4) swap the goal for another
fact of the proof, (5) drop one citation of a DD/AR step, (6) corrupt the
statement text (change the goal), (7) alter one coordinate of a point.
Each mutant must be rejected (except genuinely redundant citations in (5), which
are counted separately)."""
import copy
import glob
import json
import random
import sys

from checker import check_proof, CheckError, read_defs

DEFS = read_defs()


def accepted(pf):
    try:
        check_proof(pf, DEFS)
        return True
    except (CheckError, KeyError, IndexError, ValueError, ZeroDivisionError):
        return False


def mutants(pf, rng):
    steps = pf['steps']
    ids = [s['id'] for s in steps]
    out = []
    prem = [i for i, s in enumerate(steps) if s['rule'] == 'premise']
    pts = sorted(pf['coords'])
    # 1 perturb a premise
    if prem:
        m = copy.deepcopy(pf)
        i = rng.choice(prem)
        s = m['steps'][i]
        j = rng.randrange(len(s['args']))
        s['args'][j] = rng.choice([p for p in pts if p not in s['args']])
        out.append(('perturb_premise', m, False))
    # 2 forge premise: turn a derived (non-premise) step into a 'premise'
    der = [i for i, s in enumerate(steps) if s['rule'] not in ('premise',)]
    if der:
        m = copy.deepcopy(pf)
        i = rng.choice(der)
        m['steps'][i]['rule'] = 'premise'
        m['steps'][i]['prem'] = []
        out.append(('forge_premise', m, False))
    # 3 delete a cited step
    cited = sorted(set(p for s in steps for p in s['prem']))
    if cited:
        m = copy.deepcopy(pf)
        d = rng.choice(cited)
        m['steps'] = [s for s in m['steps'] if s['id'] != d]
        out.append(('delete_step', m, False))
    # 4 swap goal: final step replaced by another (non-goal) fact
    others = [s for s in steps[:-1] if s['kind'] != steps[-1]['kind'] or s['args'] != steps[-1]['args']]
    if others:
        m = copy.deepcopy(pf)
        o = rng.choice(others)
        m['goal_step'] = o['id']
        out.append(('swap_goal', m, False))
    # 5 drop a citation (from a step with >=1 premises)
    withp = [i for i, s in enumerate(steps) if s['prem']]
    if withp:
        m = copy.deepcopy(pf)
        i = rng.choice(withp)
        k = rng.randrange(len(m['steps'][i]['prem']))
        m['steps'][i]['prem'].pop(k)
        out.append(('drop_citation', m, None))
    # 6 change the goal in the statement text (replace one goal point by another point)
    m = copy.deepcopy(pf)
    body, goal = m['statement'].split('?')
    g = goal.split()
    j = rng.randrange(1, len(g))
    g[j] = rng.choice([p for p in pts if p not in g])
    m['statement'] = body + '? ' + ' '.join(g)
    m['goal'] = g
    out.append(('statement_goal', m, False))
    # 7 move one point numerically (facts become false)
    m = copy.deepcopy(pf)
    p = rng.choice(pts)
    m['coords'][p] = [m['coords'][p][0] + 0.137, m['coords'][p][1] - 0.071]
    out.append(('move_point', m, False))
    return out


def main(files, out_json):
    rng = random.Random(0)
    res = {}
    base_ok = 0
    for f in files:
        pf = json.load(open(f))
        if not accepted(pf):
            print('BASE REJECTED', f)
            continue
        base_ok += 1
        for rep in range(3):
            for name, m, expect in mutants(pf, rng):
                r = res.setdefault(name, dict(n=0, rejected=0))
                r['n'] += 1
                acc = accepted(m)
                r['rejected'] += (not acc)
                if acc:
                    r.setdefault('accepted_examples', []).append(f.split('/')[-2] + '/' + f.split('/')[-1])
    print('base proofs accepted', base_ok, '/', len(files))
    for k, v in res.items():
        print('%-16s %4d mutants, %4d rejected (%.1f%%)' % (k, v['n'], v['rejected'], 100 * v['rejected'] / max(1, v['n'])))
    json.dump(dict(base=base_ok, n_files=len(files), mutants=res), open(out_json, 'w'), indent=1)


if __name__ == '__main__':
    main(sys.argv[2:], sys.argv[1])
