import numpy as np


class MetaModel:
    """input: unit-cube parameters; output: V(tau) [V], dT(tau) [K], t_end [s]"""
    def __init__(self, nV, nT, w_tend=10.0):
        self.nV, self.nT, self.w_tend = nV, nT, w_tend

    def _Y(self, V, T, te):
        return np.hstack([V, T, np.log(te)[:, None]])

    def fit_scaler(self, Y):
        self.mu = Y.mean(0); self.sd = Y.std(0) + 1e-12
        # t_end column up-weighted (it is 1 of ~170 outputs but carries the capacity information)
        self.w = np.ones(Y.shape[1]); self.w[-1] = self.w_tend

    def enc(self, Y):
        return (Y - self.mu) / self.sd * self.w

    def dec(self, Z):
        return Z / self.w * self.sd + self.mu

    def predict(self, U):
        Y = self.dec(self.net.predict(np.atleast_2d(U)))
        return Y[:, :self.nV], Y[:, self.nV:self.nV + self.nT], np.exp(Y[:, -1])


