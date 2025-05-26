import numpy as np

class FxLMS:
    def __init__(self, L, mu, w):
        self.L = L
        self.mu = mu
        self.w = w
        self.u = np.zeros(self.L)

    def predict(self, x):
        self.u[1:] = self.u[:-1]
        self.u[0] = x
        return np.dot(self.w, self.u)

    def adapt(self, error, x):
        self.w += 2 * self.mu * error * x

    def estimate(self, x, d):
        y = self.predict(x)
        e = d - y
        self.adapt(e, x)
        return e, y