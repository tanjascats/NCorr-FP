import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib import cm
plt.rcParams['axes.grid'] = True
sns.set_style("whitegrid")
cmap = cm.get_cmap('plasma') # autumn
colors = [cmap(i*(1/5)) for i in range(4)]

x = 100 - np.flip(np.array([0, 0.02, 0.06, 0.1, 0.14, 0.18, 0.22, 0.26, 0.3, 0.5])) * 100
# we omit further examples because changing more than 50% of data set's values is arguably of no
# significant value for the data user
y = [None, None, None, None]  # detected fingerprints
# baseline (random fp) --> fingerprinting_toolbox/evaluation/robustness/flipping/
x_b = 100 - np.array([0, 0.02, 0.06, 0.1, 0.14, 0.18, 0.22, 0.26, 0.3, 0.5])*100
y_b = [None, None, None, None]

n_exp = 20

# --------------------------------------- #
# gamma = 5
# --------------------------------------- #
y[0] = 20 - np.array([20, 20, 20, 20, 20, 20, 20, 20, 20, 20])
y_b[0] = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
# --------------------------------------- #
# gamma = 10
# --------------------------------------- #
y[1] = 100 - (np.array([12, 17, 20, 20, 20, 20, 20, 20, 20, 20]) +
              np.array([61, 78, 80, 80, 80, 80, 80, 80, 80, 80]))
y_b[1] = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
# --------------------------------------- #
# gamma = 20
# --------------------------------------- #
y[2] = 100 - (np.array([1, 3, 10, 13, 12, 19, 16, 19, 19, 20]) +
              np.array([5, 45, 59, 72, 76, 80, 80, 80, 80, 80]))  # [3, 10, 13, 12, 19, 16, 19, 19]
y_b[2] = np.flip([0.28, 0, 0, 0, 0, 0, 0, 0, 0, 0])
# --------------------------------------- #
# gamma = 40
# --------------------------------------- #
y[3] = 100 - (np.array([0, 1, 5, 8, 11, 15, 17, 19, 20, 20]) +
              np.array([0, 6, 6, 22, 31, 51, 67, 76, 80, 80]))
y_b[3] = np.flip([0.92, 0.36, 0.28, 0.20, 0, 0, 0, 0, 0, 0])

fig, ax = plt.subplots(1, 1, sharex='col', sharey='row')
ax.set_xlabel("Portion of the unchanged data(%)", size=14)
ax.set_ylabel("False Miss", size=14)
ax.set_prop_cycle(color=colors)
ax.plot(x_b, y_b[0], label='baselines (random FP)', color=colors[0], linestyle=(0, (5, 5)), linewidth=1)
ax.plot(x, y[0]/n_exp, label='$\gamma$ = 5 (NCorr-FP)')#, c='0.15')
ax.plot(x, y[1]/100, label='$\gamma$ = 10')#, marker='o', c='0.35')
ax.plot(x_b, y_b[1], color=colors[1], linestyle=(0, (5, 5)), linewidth=1)
ax.plot(x, y[2]/100, label='$\gamma$ = 20')#, c='0.65')
ax.plot(x_b, y_b[2], color=colors[2], linestyle=(0, (5, 5)), linewidth=1)
ax.plot(x, y[3]/100, label='$\gamma$ = 40')#, c='0.85')
ax.plot(x_b, y_b[3], color=colors[3], linestyle=(0, (5, 5)), linewidth=1)
ax.legend()
plt.show()