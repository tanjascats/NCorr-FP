from matplotlib import pyplot as plt
from matplotlib import cm
import numpy as np
import seaborn as sns

sns.set_style("whitegrid")
cmap = cm.get_cmap('plasma') # autumn
colors = [cmap(i*(1/5)) for i in range(4)]

x = np.array([0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.40, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85,
              0.9, 0.95, 1])
x = x*100
n_exp = 100
y = [None, None, None, None]
# baselines (random fp) --> fingerprinting_toolbox/evaluation/robustness/horizontal/
x_b = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])*100
y_b = [None, None, None, None]


# RESULTS FOR L=64
# robustness_analysis/results/horizontal/adult
# gamma = 5
y[0] = n_exp - (np.flip([20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 19, 19, 13, 0, 0]) +
                np.array([0, 2, 65, 75, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80]))
y_b[0] = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
# ----------------------------------------------------------------------------- #
# ----------------------------------------------------------------------------- #
# gamma = 10
y[1] = n_exp - (np.array([0, 0, 0, 6, 16, 17, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20]) +
                np.array([0, 0, 4, 37, 69, 79, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80]))
y_b[1] = np.array([1, 0.16, 0.0, 0, 0, 0, 0, 0, 0, 0, 0])
# ----------------------------------------------------------------------------- #
# ----------------------------------------------------------------------------- #
# gamma = 20
y[2] = n_exp - (np.array([0, 0, 0, 0, 1, 5, 8, 12, 16, 17, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20]) +
                np.array([0, 0, 0, 1, 9, 29, 53, 63, 71, 76, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80]))
y_b[2] = np.array([1, 0.94, 0.28, 0.0, 0, 0, 0, 0, 0, 0, 0])
# ----------------------------------------------------------------------------- #
# ----------------------------------------------------------------------------- #
# gamma = 40
y[3] = n_exp - (np.array([0, 0, 0, 0, 0, 0, 0, 1, 2, 5, 6, 9, 12, 16, 18, 18, 17, 17, 20, 20, 20]) +
                np.array([0, 0, 0, 0, 0, 0, 1, 6, 23, 37, 53, 56, 70, 72, 73, 78, 78, 80, 80, 80, 80]))
y_b[3] = np.array([1, 1, 0.88, 0.48,  0.14, 0.08,  0.0, 0, 0, 0, 0])

fig2, ax2 = plt.subplots(1, 1, sharex='col', sharey='row')
ax2.set_prop_cycle(color=colors)
ax2.plot(x_b, y_b[0], label='baselines (random FP)', color=colors[0], linestyle=(0, (5, 5)), linewidth=1)
ax2.plot(x, y[0]/n_exp, label='$\gamma$ = 5 (NCorr-FP)')  #, c='0.15')
ax2.plot(x, y[1]/n_exp, label='$\gamma$ = 10')  #, c='0.35')
ax2.plot(x_b, y_b[1], color=colors[1], linestyle=(0, (5, 5)), linewidth=1)
ax2.plot(x, y[2]/n_exp, label='$\gamma$ = 20')  #, c='0.65')
ax2.plot(x_b, y_b[2], color=colors[2], linestyle=(0, (5, 5)), linewidth=1)
ax2.plot(x, y[3]/n_exp, label='$\gamma$ = 40')  #, c='0.85')
ax2.plot(x_b, y_b[3], color=colors[3], linestyle=(0, (5, 5)), linewidth=1)
fig2.text(0.5, 0.02, 'Size of the subset released(%)', ha='center', size=14)
fig2.text(0.04, 0.5, 'False Miss', va='center', rotation='vertical', size=14)
ax2.legend()
plt.show()

