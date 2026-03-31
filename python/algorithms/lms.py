import numpy as np

class LMS:
    def __init__(self, L, mu, w, dtype=np.float32):
        self.L = int(L)
        self.mu = float(mu)
        self.w = np.array(w, dtype=dtype, copy=True)  # Filter weights
        self.u = np.zeros(self.L, dtype=dtype)  # Input buffer
    
    def predict(self, x):
        self.u[1:] = self.u[:-1]  # Shift buffer
        self.u[0] = x  # Insert new input
        return np.dot(self.w, self.u)

    def adapt(self, error):
        self.w += 2 * self.mu * float(error) * self.u

    def estimate(self, x, d):
        y = self.predict(x)
        if not np.isfinite(y):
            y = 0.0
        y = float(y)
        e = d - y  # Compute error
        self.adapt(e)  # Update weights
        return e, y