from matplotlib import pyplot as plt
from matplotlib import cm
import numpy as np
import seaborn as sns

#plt.rcParams['axes.grid'] = True
#plt.rcParams["legend.loc"] = 'upper right'
#plt.style.use('seaborn-colorblind')
sns.set_style("whitegrid")
cmap = cm.get_cmap('plasma') # autumn
colors = [cmap(i*(1/5)) for i in range(4)]

# fig, ax = plt.subplots(2, 2, sharex='col', sharey='row')

x = np.array([0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.40, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85,
              0.9, 0.95, 1])
x = x*100
n_exp = 1000
y = [None, None, None, None]
# baselines (random fp) --> fingerprinting_toolbox/evaluation/robustness/horizontal/
x_b = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])*100
y_b = [None, None, None, None]

# gamma = 1
y[0] = n_exp - np.array([0, 4, 135, 381, 648, 786, 897, 955, 978, 993, 999, 1000, 999, 1000, 1000, 1000, 1000, 1000,
                 1000, 1000, 1000])
y_b[0] = np.array([1, 0.76, 0.14, 0, 0, 0.0, 0, 0, 0, 0, 0])
# ----------------------------------------------------------------------------- #
# ----------------------------------------------------------------------------- #
# gamma = 2
y[1] = n_exp - np.array([0, 0, 17, 115, 223, 460, 584, 720, 777, 876, 915, 954, 961, 991, 988, 992, 999, 1000, 1000, 1000,
                 1000])
y_b[1] = np.array([1,  0.98, 0.67, 0.29, 0.15, 0.05, 0.01, 0.0, 0, 0, 0])
# ----------------------------------------------------------------------------- #
# ----------------------------------------------------------------------------- #
# gamma = 3
y[2] = n_exp - np.array([0, 0, 9, 28, 119, 203, 360, 492, 608, 676, 777, 833, 890, 904, 952, 972, 990, 996, 998, 1000,
                 1000])
y_b[2] = np.array([1, 1.0, 0.94, 0.62, 0.35, 0.13, 0.08, 0.01, 0.0, 0, 0])
# ----------------------------------------------------------------------------- #
# ----------------------------------------------------------------------------- #
# gamma = 5
y[3] = n_exp - np.array([0, 0, 0, 3, 10, 52, 75, 186, 271, 345, 454, 577, 620, 677, 785, 820, 892, 917, 961, 986, 1000])
y_b[3] = np.array([1, 1.0, 1.0,  0.87,  0.61, 0.39, 0.11, 0.05,  0.01,  0.0, 0])
# ----------------------------------------------------------------------------- #
# ----------------------------------------------------------------------------- #
# gamma = 7 - not 100% successful anymore

# todo test how they look all in one plot
#ax[0, 0].plot(x, y[0])
#ax[0, 0].plot(x, y[1])
# todo end of test

#fig.text(0.5, 0.02, 'Size of the subset(%)', ha='center')
#fig.text(0.04, 0.5, 'Detected fingerprints(%)', va='center', rotation='vertical')

#plt.show()

fig2, ax2 = plt.subplots(1, 1, sharex='col', sharey='row')
ax2.set_prop_cycle(color=colors)
ax2.plot(x_b, y_b[0], label='baselines (random FP)', color=colors[0], linestyle=(0, (5, 5)), linewidth=1)
ax2.plot(x, y[0]/n_exp, label='$\gamma$ = 1 (NCorr-FP)')

ax2.plot(x, y[1]/n_exp, label='$\gamma$ = 2')
ax2.plot(x_b, y_b[1], color=colors[1], linestyle=(0, (5, 5)), linewidth=1)

ax2.plot(x, y[2]/n_exp, label='$\gamma$ = 3')
ax2.plot(x_b, y_b[2], color=colors[2], linestyle=(0, (5, 5)), linewidth=1)
ax2.plot(x, y[3]/n_exp, label='$\gamma$ = 5')
ax2.plot(x_b, y_b[3], color=colors[3], linestyle=(0, (5, 5)), linewidth=1)
# ax2.plot(x, y[3])
fig2.text(0.5, 0.02, 'Size of the subset released(%)', ha='center', size=14)
fig2.text(0.04, 0.5, 'False Miss', va='center', rotation='vertical', size=14)
ax2.legend()
plt.show()

