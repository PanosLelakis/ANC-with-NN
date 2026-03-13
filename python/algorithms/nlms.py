import numpy as np

class NLMS:
    def __init__(self, L, mu, w, dtype=np.float32):
        self.L = L
        self.mu = mu
        self.w = np.array(w, dtype=dtype, copy=True)  # Filter weights
        self.u = np.zeros(self.L, dtype=dtype)  # Input buffer
    
    def predict(self, x):
        self.u[1:] = self.u[:-1]
        self.u[0] = x
        return float(np.dot(self.w, self.u))

    def adapt(self, error):
        norm_factor = float(np.dot(self.u, self.u) + 1e-8)
        self.w += 2 * (self.mu / norm_factor) * float(error) * self.u

    def estimate(self, x, d):
        y = self.predict(x)
        e = d - y
        self.adapt(e)
        return e, y