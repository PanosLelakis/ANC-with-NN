from scipy.io import loadmat
import mat73
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

primary_path = mat73.loadmat("python/primary_path_new.mat")['sim_imp'].flatten()[:4000]
secondary_path = loadmat("python/secondary_path_new.mat")['sim_imp'].flatten()[:2000]

plt.figure()

plt.subplot(2,1,1)
plt.plot(primary_path)
plt.grid()

plt.subplot(2,1,2)
plt.plot(secondary_path)
plt.grid()

plt.show()