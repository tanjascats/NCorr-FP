from matplotlib import pyplot as plt
from matplotlib import cm
import numpy as np
import seaborn as sns

#plt.rcParams['axes.grid'] = True
#plt.style.use('seaborn-colorblind')
sns.set_style("whitegrid")
cmap = cm.get_cmap('plasma') # autumn
colors = [cmap(i*(1/5)) for i in range(4)]

x = np.array([i for i in range(11)])  # -> number of columns released (the results are in ascending order of
                                        # number of columns DELETED)
y = [None, None, None, None]
# baselines (random fp) --> fingerprinting_toolbox/evaluation/robustness/vertical/
y_b = [None, None, None, None]

n_exp = 50
# gamma = 1
y[0] = n_exp - (np.array([0, 4, 4, 4, 7, 8, 13, 17, 16, 20, 20]) + np.array([0, 6, 7, 13, 15, 20, 21, 27, 30, 30, 30]))
y_b[0] = np.array([1, 1, 1.0, 0.97, 0.89, 0.69, 0.27, 0.04, 0.0, 0, 0])
# ----------------------------------------------------------------------------- #
# ----------------------------------------------------------------------------- #
# gamma = 2
y[1] = n_exp - (np.array([0, 0, 3, 2, 3, 3, 7, 7, 8, 9, 20]) + np.array([0, 2, 3, 4, 5, 9, 7, 14, 16, 24, 30]))
y_b[1] = np.array([1, 1, 1.0, 1.0, 0.96, 0.9, 0.59, 0.24,  0.07,  0.0,0])
# ----------------------------------------------------------------------------- #
# ----------------------------------------------------------------------------- #
# gamma = 3
y[2] = n_exp - (np.array([0, 0, 0, 0, 0, 0, 2, 5, 7, 6, 20]) + np.array([0, 1, 0, 0, 2, 2, 3, 2, 3, 8, 30]))
y_b[2] = np.array([1, 1, 1.0,  1.0, 1.0, 1.0, 1.0, 0.93, 0.8,  0.5,0])
# ----------------------------------------------------------------------------- #
# ----------------------------------------------------------------------------- #
# gamma = 5
y[3] = n_exp - (np.array([0, 0, 0, 0, 0, 0, 0, 1, 2, 1, 20]) + np.array([0, 0, 0, 0, 0, 0, 0, 1, 2, 5, 30]))
y_b[3] = np.array([1, 1, 1.0, 1.0, 1.0, 1.0,  1.0,  1.0, 0.82,  0.44,0])
# ----------------------------------------------------------------------------- #
# ----------------------------------------------------------------------------- #


# --------------------------------------------------------------------- #
# ----------------- ALL IN ONE FIGURE --------------------------------- #
# --------------------------------------------------------------------- #
fig2, ax2 = plt.subplots(1, 1, sharex='col', sharey='row')
ax2.set_prop_cycle(color=colors)
ax2.plot(x, y_b[0], label='baselines (random FP)', color=colors[0], linestyle=(0, (5, 5)), linewidth=1)
ax2.plot(x, y[0]/n_exp, label='$\gamma$ = 1 (NCorr-FP)')#, c='0.15')
ax2.plot(x, y[1]/n_exp, label='$\gamma$ = 2')#, c='0.35')
ax2.plot(x, y_b[1], color=colors[1], linestyle=(0, (5, 5)), linewidth=1)
ax2.plot(x, y[2]/n_exp, label='$\gamma$ = 3')#, c='0.65')
ax2.plot(x, y_b[2], color=colors[2], linestyle=(0, (5, 5)), linewidth=1)
ax2.plot(x, y[3]/n_exp, label='$\gamma$ = 5')#, c='0.85')
ax2.plot(x, y_b[3], color=colors[3], linestyle=(0, (5, 5)), linewidth=1)
# ax2.plot(x, y[3])
plt.xticks(np.arange(0, 11, step=1), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 'ALL'])
fig2.text(0.5, 0.02, 'Number of columns released', ha='center', size=14)
fig2.text(0.04, 0.5, 'False Miss(%)', va='center', rotation='vertical', size=14)
ax2.legend()
plt.show()


