import numpy as np

class FxLMS:
    def __init__(self, L, mu, w, dtype=np.float32):
        self.L = int(L)
        self.mu = float(mu)
        self.w = np.array(w, dtype=dtype, copy=True)

        self.u_x  = np.zeros(self.L, dtype=dtype)   # raw-x buffer
        self.u_xf = np.zeros(self.L, dtype=dtype)   # filtered-x buffer

    def predict(self, x):
        self.u_x[1:] = self.u_x[:-1]
        self.u_x[0] = x
        return np.dot(self.w, self.u_x)

    def adapt(self, error, x_f):
        self.u_xf[1:] = self.u_xf[:-1]
        self.u_xf[0] = x_f
        # LMS update (no normalization)
        self.w += 2 * self.mu * float(error) * self.u_xf