from matplotlib import pyplot as plt
from matplotlib import cm
import numpy as np
import seaborn as sns

#plt.rcParams['axes.grid'] = True
#plt.rcParams["legend.loc"] = 'lower right'
#plt.style.use('seaborn-colorblind')
sns.set_style("whitegrid")
cmap = cm.get_cmap('plasma') # autumn
colors = [cmap(i*(1/5)) for i in range(4)]

fig, ax = plt.subplots(2, 2, sharex='col', sharey='row')

x = np.array([0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.40, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85,
              0.9, 0.95, 1])
x = x*100
n_exp = 500 + 500
y_1 = np.array([0, 8, 253, 429, 420, 496, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500])
y_2 = np.array([0, 3, 225, 390, 469, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500])
y = [None, None, None, None]
y[0] = y_1 + y_2
y[0] = (y[0]/n_exp)*100

n_exp = 1000
# !!!! RESULTS FOR L=16 !!!!
# gamma = 2
y[0] = n_exp - np.array([0, 2, 385, 842, 961, 991, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000,
                 1000, 1000])
# ----------------------------------------------------------------------------- #
# ----------------------------------------------------------------------------- #
# gamma = 5
y[1] = n_exp - np.array([0, 0, 1, 50, 232, 437, 651, 805, 878, 937, 970, 978, 990, 997, 997, 997, 1000, 1000, 1000, 1000, 1000])
# ----------------------------------------------------------------------------- #
# ----------------------------------------------------------------------------- #
# gamma = 7
y[2] = n_exp - np.array([0, 0, 0, 4, 44, 127, 297, 478, 591, 749, 814, 885, 914, 955, 974, 964, 994, 997, 998, 1000, 1000])
# ----------------------------------------------------------------------------- #
# ----------------------------------------------------------------------------- #
# gamma = 10
y[3] = n_exp - np.array([0, 0, 0, 1, 3, 22, 62, 135, 240, 314, 486, 568, 634, 759, 826, 889, 904, 932, 963, 988, 1000])
# ----------------------------------------------------------------------------- #
# ----------------------------------------------------------------------------- #

for i in range(2):
    for j in range(2):
        y[2*i+j] = (y[2*i+j] / n_exp)
        ax[i, j].plot(x, y[2*i+j])
        plt.grid()

# todo test how they look all in one plot
ax[0, 0].plot(x, y[1])
ax[0, 0].plot(x, y[2])
ax[0, 0].plot(x, y[3])
# todo end of test

fig.text(0.5, 0.02, 'Size of the subset(%)', ha='center')
fig.text(0.04, 0.5, 'Detected fingerprints(%)', va='center', rotation='vertical')

#plt.show()

fig2, ax2 = plt.subplots(1, 1, sharex='col', sharey='row')
ax2.set_prop_cycle(color=colors)
ax2.plot(x, y[0], label='$\gamma$ = 2')
ax2.plot(x, y[1], label='$\gamma$ = 5')
ax2.plot(x, y[2], label='$\gamma$ = 7')
ax2.plot(x, y[3], label='$\gamma$ = 10')
# ax2.plot(x, y[3])
fig2.text(0.5, 0.02, 'Size of the subset released(%)', ha='center', size=14)
fig2.text(0.04, 0.5, 'False miss', va='center', rotation='vertical', size=14)
ax2.legend()
plt.show()

