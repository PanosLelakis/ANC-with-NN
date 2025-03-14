import numpy as np

class FxLMS:
    def __init__(self, L, mu, secondary_path_impulse_response):
        self.L = L
        self.mu = mu
        self.w = np.zeros(self.L)
        self.u = np.zeros(self.L)
        self.secondary_path_impulse_response = secondary_path_impulse_response

    def predict(self, x):
        self.u[1:] = self.u[:-1]
        self.u[0] = x
        return np.dot(self.w, self.u)

    def adapt(self, error, filtered_x):
        self.w = self.w - self.mu * error * filtered_x

    def estimate(self, x, d):
        y = self.predict(x)
        filtered_x = np.convolve(x, self.secondary_path_impulse_response, mode='full')[-self.L:]
        e = d - y
        self.adapt(e, filtered_x)
        return e, y