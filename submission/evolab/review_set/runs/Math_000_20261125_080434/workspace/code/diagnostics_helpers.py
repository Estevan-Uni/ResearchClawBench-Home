import numpy as np
def occ_frame(gtb, res=2.0):
    n = len(gtb); occ = np.zeros(n)
    for i, b in enumerate(gtb):
        front = (gtb[:, 3] > b[3])
        ov = front & (gtb[:, 0] < b[2]) & (gtb[:, 2] > b[0]) & (gtb[:, 1] < b[3]) & (gtb[:, 3] > b[1])
        if not ov.any(): continue
        xs = np.arange(b[0], b[2], res) + res / 2; ys = np.arange(b[1], b[3], res) + res / 2
        X, Y = np.meshgrid(xs, ys); cov = np.zeros(X.shape, bool)
        for c in gtb[ov]: cov |= (X >= c[0]) & (X <= c[2]) & (Y >= c[1]) & (Y <= c[3])
        occ[i] = cov.mean()
    return occ
