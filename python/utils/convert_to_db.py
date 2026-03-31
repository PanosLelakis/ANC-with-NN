import numpy as np

# Convert signal to dB
def val_to_db(signal):
    return 20.0 * np.log10(np.abs(signal) + 1e-12)

# Convert signal to dBr
def val_to_dbr(signal, max_val):
    #x = np.asarray(signal, dtype=np.float64)
    #x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    #dbr = 20.0 * np.log10((np.abs(x) / (max_val + 1e-12)) + 1e-12)
    #dbr = np.nan_to_num(dbr, nan=-120.0, posinf=120.0, neginf=-120.0)
    return 20.0 * np.log10((np.abs(signal) / (max_val + 1e-12)) + 1e-12)

# Convert from dB to dBr
def db_to_dbr(signal, max_val):
    return signal - 20.0 * np.log10(max_val + 1e-12)