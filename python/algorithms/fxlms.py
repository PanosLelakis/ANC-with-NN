import numpy as np

class FxLMS:
    def __init__(self, L, mu, w, dtype=np.float64):
        self.L = L
        self.mu = mu
        self.w = np.array(w, dtype=dtype, copy=True)
        self.u = np.zeros(self.L, dtype=dtype)

    def predict(self, x):
        self.u[1:] = self.u[:-1]
        self.u[0] = x
        return np.dot(self.w, self.u)

    def adapt(self, error):
        self.w += self.mu * error * self.u

    def estimate(self, x, d):
        y = self.predict(x)
        e = d - y
        self.adapt(e)
        return e, y