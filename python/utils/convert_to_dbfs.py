import numpy as np

def convert_to_dbfs(signal, max_val):
    #max_val = np.max(np.abs(signal))
    #if max_val == 0:
        #return np.full_like(signal, -np.inf)  # Avoid log of zero
    x = np.asarray(signal, dtype=np.float64)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    dbfs = 20.0 * np.log10((np.abs(x) / (max_val + 1e-12)) + 1e-12)
    dbfs = np.nan_to_num(dbfs, nan=-120.0, posinf=120.0, neginf=-120.0)
    return dbfs