import numpy as np

def convert_to_dbfs(signal, max_val):
    #max_val = np.max(np.abs(signal))
    #if max_val == 0:
        #return np.full_like(signal, -np.inf)  # Avoid log of zero
    return 20 * np.log10((np.abs(signal) / (max_val + 1e-8)) + 1e-8)