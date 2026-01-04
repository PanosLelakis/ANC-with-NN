import numpy as np

def convert_to_dbfs(signal, max_val):
    #max_val = np.max(np.abs(signal))
    #if max_val == 0:
        #return np.full_like(signal, -np.inf)  # Avoid log of zero
    x = np.asarray(signal, dtype=np.float64)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    mv = float(max(max_val, 1e-12))
    db = 20.0 * np.log10((np.abs(x) / mv) + 1e-8)
    db = np.nan_to_num(db, nan=-120.0, posinf=120.0, neginf=-120.0)
    return db