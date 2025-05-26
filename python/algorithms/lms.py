import numpy as np

class LMS:
    def __init__(self, L, mu, w):
        self.L = L
        self.mu = mu
        self.w = w  # Filter weights
        self.u = np.zeros(self.L)  # Input buffer

    def predict(self, x):
        self.u[1:] = self.u[:-1]  # Shift buffer
        self.u[0] = x  # Insert new input
        return np.dot(self.w, self.u)

    def adapt(self, error):
        self.w += 2 * self.mu * error * self.u

    def estimate(self, x, d):
        y = self.predict(x)
        e = d - y  # Compute error
        self.adapt(e)  # Update weights
        return e, y