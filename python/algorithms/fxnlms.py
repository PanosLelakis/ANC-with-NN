import numpy as np

class FxNLMS:
    def __init__(self, L, mu, w, dtype=np.float64):
        self.L = int(L)
        self.mu = float(mu)
        self.w = np.array(w, copy=True, dtype=dtype)

        self.u_x  = np.zeros(self.L, dtype=dtype)  # raw x buffer (for y)
        self.u_xf = np.zeros(self.L, dtype=dtype)  # filtered-x buffer (for update)

    def predict(self, x):
        self.u_x[1:] = self.u_x[:-1]
        self.u_x[0] = x
        return float(np.dot(self.w, self.u_x))

    def adapt(self, error, x_f):
        self.u_xf[1:] = self.u_xf[:-1]
        self.u_xf[0] = x_f
        norm = float(np.dot(self.u_xf, self.u_xf) + 1e-8)
        self.w += 2 * (self.mu / norm) * float(error) * self.u_xf