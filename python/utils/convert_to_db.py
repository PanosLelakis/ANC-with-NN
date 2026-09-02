import numpy as np

# Convert signal to dBr
def val_to_dbr(signal, max_val):
    return 20.0 * np.log10((np.abs(signal) / (max_val + 1e-12)) + 1e-12)