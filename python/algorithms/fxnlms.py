import numpy as np

class FxNLMS:
    def __init__(self, L, mu, w):
        self.L = L
        self.mu = mu
        self.w = w
        self.u = np.zeros(self.L)

    def predict(self, x):
        self.u[1:] = self.u[:-1]
        self.u[0] = x
        return np.dot(self.w, self.u)

    def adapt(self, error):
        norm_factor = np.dot(self.u, self.u) + 1e-8
        self.w += (self.mu / norm_factor) * error * self.u

    def estimate(self, x, d):
        y = self.predict(x)
        e = d - y
        self.adapt(e)
        return e, y