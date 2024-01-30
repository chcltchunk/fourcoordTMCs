import numpy as np
import matplotlib.pyplot as plt

targets = np.load("fcTMCml/features/regression_rff_selection/regression_targets.npy")

plt.hist(targets, bins=100)
plt.show()
