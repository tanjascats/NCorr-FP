import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib import cm
plt.rcParams['axes.grid'] = True
sns.set_style("whitegrid")
cmap = cm.get_cmap('plasma') # autumn
colors = [cmap(i*(1/5)) for i in range(4)]

x = 100 - np.flip(np.array([0, 0.02, 0.06, 0.1, 0.14, 0.18, 0.22, 0.26, 0.3, 0.5])) * 100
# we omit further examples because changing more than 60% of data set's values is arguably of no
# significant value for the data user
y = [None, None, None, None]  # detected fingerprints
# baselines (random fp) --> fingerprinting_toolbox/evaluation/robustness/flipping/
y_b = [None, None, None, None]
x_b = [50, 60, 70, 80, 90, 100]
n_exp = 50

# --------------------------------------- #
# gamma = 1
# --------------------------------------- #
y[0] = n_exp - (np.array([3, 6, 6, 10, 10, 10, 10, 10, 10, 10]) + np.array([2, 24, 30, 38, 37, 40, 40, 40, 40, 40]))
y_b[0] = np.array([0.58, 0.23, 0.03, 0.01, 0.0, 0])
# --------------------------------------- #
# gamma = 2
# --------------------------------------- #
y[1] = n_exp - (np.array([1, 4, 6, 7, 6, 9, 10, 10, 10, 10]) + np.array([1, 14, 14, 30, 27, 34, 38, 37, 39, 40]))
y_b[1] = np.array([0.98, 0.87, 0.65, 0.25, 0.09, 0])
# --------------------------------------- #
# gamma = 3
# --------------------------------------- #
y[2] = n_exp - (np.array([1, 5, 4, 6, 5, 8, 9, 9, 10, 10]) + np.array([1, 6, 6, 13, 16, 27, 33, 32, 39, 40]))
y_b[2] = np.array([1.0,0.98, 0.92, 0.73,  0.56,0])
# --------------------------------------- #
# gamma = 5
# --------------------------------------- #
y[3] = n_exp - np.array([0, 1, 0, 4, 0, 3, 5, 6, 10, 10] + np.array([0, 1, 1, 3, 7, 11, 16, 18, 25, 40]))
y_b[3] = np.array([1.0, 1.0,  0.98,  0.91,  0.6, 0])

fig, ax = plt.subplots(1, 1, sharex='col', sharey='row')
ax.set_xlabel("Portion of the unchanged data(%)", size=14)
ax.set_ylabel("False Miss", size=14)
ax.set_prop_cycle(color=colors)
ax.plot(x_b, y_b[0], label='baselines (random FP)', color=colors[0], linestyle=(0, (5, 5)), linewidth=1)
ax.plot(x, y[0]/n_exp, label='$\gamma$ = 1 (NCorr-FP)')#, c='0.15')
ax.plot(x, y[1]/n_exp, label='$\gamma$ = 2')#, c='0.35')
ax.plot(x_b, y_b[1], color=colors[1], linestyle=(0, (5, 5)), linewidth=1)
ax.plot(x, y[2]/n_exp, label='$\gamma$ = 3')#, c='0.65')
ax.plot(x_b, y_b[2], color=colors[2], linestyle=(0, (5, 5)), linewidth=1)
ax.plot(x, y[3]/n_exp, label='$\gamma$ = 5')#, c='0.85')
ax.plot(x_b, y_b[3], color=colors[3], linestyle=(0, (5, 5)), linewidth=1)
ax.legend()
plt.show()