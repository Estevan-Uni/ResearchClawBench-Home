"""Shared utilities: dataset parser, Adam optimiser, plotting style."""
import ast, json, os
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, 'data', 'M-AI-Synth__Materials_AI_Dataset_.txt')
OUT = os.path.join(ROOT, 'outputs')
IMG = os.path.join(ROOT, 'report', 'images')
os.makedirs(OUT, exist_ok=True); os.makedirs(IMG, exist_ok=True)


def load_dataset(path=DATA):
    """Parse the three '#'-headed blocks; each non-empty line is a Python list."""
    blocks, cur = {}, None
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('#'):
                cur = line.split(':')[-1].strip().split()[0]  # e.g. property_prediction.py
                blocks[cur] = []
            else:
                blocks[cur].append(np.array(ast.literal_eval(line), dtype=float))
    assert [len(blocks[k]) for k in blocks] == [4, 2, 6], {k: len(v) for k, v in blocks.items()}
    pp, sg, ao = (blocks[k] for k in ['property_prediction.py', 'structure_generation.py',
                                       'autonomous_optimization.py'])
    return dict(
        atomic_numbers=pp[0].astype(int), node_features=pp[1], edge_list=pp[2].astype(int),
        targets=pp[3], lattice_a=sg[0], lattice_b=sg[1],
        T_range=ao[0], t_range=ao[1], T_opt=float(ao[2][0]), t_opt=float(ao[3][0]),
        noise=float(ao[4][0]), n_iter=int(ao[5][0]))


class Adam:
    def __init__(self, params, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8, wd=0.0):
        self.p, self.lr, self.b1, self.b2, self.eps, self.wd = params, lr, b1, b2, eps, wd
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def step(self, grads):
        self.t += 1
        for k in self.p:
            g = grads[k] + self.wd * self.p[k]
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * g * g
            mh = self.m[k] / (1 - self.b1 ** self.t); vh = self.v[k] / (1 - self.b2 ** self.t)
            self.p[k] -= self.lr * mh / (np.sqrt(vh) + self.eps)


def grad_check(loss_fn, grad_fn, params, n=25, h=1e-6, seed=0):
    """Central-difference check on n random entries; returns max relative error."""
    rng = np.random.default_rng(seed)
    g = grad_fn(params)
    errs = []
    for k in params:
        flat = params[k].reshape(-1)
        for i in rng.choice(flat.size, size=min(n, flat.size), replace=False):
            old = flat[i]
            flat[i] = old + h; lp = loss_fn(params)
            flat[i] = old - h; lm = loss_fn(params)
            flat[i] = old
            num = (lp - lm) / (2 * h); ana = g[k].reshape(-1)[i]
            errs.append(abs(num - ana) / max(1e-6, abs(num) + abs(ana)))
    return float(np.max(errs))


def save_json(obj, name):
    with open(os.path.join(OUT, name), 'w') as f:
        json.dump(obj, f, indent=2, default=lambda o: o.tolist() if hasattr(o, 'tolist') else float(o))


def style():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'figure.dpi': 130, 'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'axes.grid': True, 'grid.alpha': 0.3,
                         'savefig.bbox': 'tight'})
    return plt

C = ['#2b6cb0', '#dd6b20', '#38a169', '#805ad5', '#c53030', '#718096']
