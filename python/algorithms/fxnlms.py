import numpy as np

class FxNLMS:
    def __init__(self, L, mu):
        self.L = L
        self.mu = mu
        self.w = np.zeros(self.L)
        self.u = np.zeros(self.L)

    def predict(self, x):
        self.u[1:] = self.u[:-1]
        self.u[0] = x
        return np.dot(self.w, self.u)

    def adapt(self, error, filtered_x):
        norm_factor = np.dot(filtered_x, filtered_x) + 1e-8
        self.w += (self.mu / norm_factor) * error * filtered_x

    def estimate(self, x, d):
        y = self.predict(x)
        e = d - y
        self.adapt(e, x)
        return e, y