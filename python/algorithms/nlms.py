import numpy as np

class NLMS:
    def __init__(self, L, mu, w):
        self.L = L
        self.mu = mu
        self.w = w  # Filter weights
        self.u = np.zeros(self.L)  # Input buffer

    def predict(self, x):
        self.u[1:] = self.u[:-1]
        self.u[0] = x
        return np.dot(self.w, self.u)

    def adapt(self, error):
        norm_factor = np.dot(self.u, self.u) + 1e-8
        self.w += 2 * self.mu * error * self.u / norm_factor

    def estimate(self, x, d):
        y = self.predict(x)
        e = d - y
        self.adapt(e)
        return e, y