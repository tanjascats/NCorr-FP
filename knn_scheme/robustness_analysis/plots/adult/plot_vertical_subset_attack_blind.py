from matplotlib import pyplot as plt
from matplotlib import cm
import numpy as np
import seaborn as sns

#plt.rcParams['axes.grid'] = True
#plt.style.use('seaborn-colorblind')
sns.set_style("whitegrid")
cmap = cm.get_cmap('plasma') # autumn
colors = [cmap(i*(1/5)) for i in range(4)]

fig, ax = plt.subplots(2, 2, sharex='col', sharey='row')

x = np.array([i for i in range(16)])  # -> number of columns released (the results are in ascending order of
                                        # number of columns DELETED)
y = [None, None, None, None]

n_exp = 20
# !!!! RESULTS FOR ADULT L=64!!!!
# gamma = 1
y[0] = n_exp - np.array([0, 17, 17, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20])
# ----------------------------------------------------------------------------- #
# ----------------------------------------------------------------------------- #
# gamma = 2
y[1] = n_exp - np.array([0, 10, 13, 17, 16, 16, 14, 17, 20, 20, 20, 20, 20, 20, 20, 20])
# ----------------------------------------------------------------------------- #
# ----------------------------------------------------------------------------- #
# gamma = 3
y[2] = n_exp - np.array([0, 0, 3, 7, 6, 8, 5, 9, 18, 12, 17, 15, 20, 20, 20, 20])
# ----------------------------------------------------------------------------- #
# ----------------------------------------------------------------------------- #
# gamma = 5
y[3] = n_exp+n_exp - (np.array([0, 0, 0, 2, 2, 1, 2, 2, 7, 3, 4, 4, 10, 7, 12, 20]) + np.array([0, 0, 1, 1, 0, 0, 2, 3, 3, 5, 6, 7, 6, 12, 19, 20]))
# ----------------------------------------------------------------------------- #
# ----------------------------------------------------------------------------- #

for i in range(2):
    for j in range(2):
        y[2*i+j] = (y[2*i+j] / n_exp) * 100
        ax[i, j].plot(x, y[2*i+j])
        plt.grid()

# todo test how they look all in one plot
ax[0, 0].plot(x, y[1])
ax[0, 0].plot(x, y[2])
ax[0, 0].plot(x, y[3])
# todo end of test

fig.text(0.5, 0.02, 'Number of columns released', ha='center')
fig.text(0.04, 0.5, 'Detected fingerprints(%)', va='center', rotation='vertical')

# --------------------------------------------------------------------- #
# ----------------- ALL IN ONE FIGURE --------------------------------- #
# --------------------------------------------------------------------- #
fig2, ax2 = plt.subplots(1, 1, sharex='col', sharey='row')
ax2.set_prop_cycle(color=colors)
ax2.plot(x, y[0]/100, label='$\gamma$ = 5')#, c='0.15')
ax2.plot(x, y[1]/100, label='$\gamma$ = 10')#, c='0.35')
ax2.plot(x, y[2]/100, label='$\gamma$ = 20')#, c='0.65')
ax2.plot(x, y[3]/100, label='$\gamma$ = 40')#, c='0.85')
# ax2.plot(x, y[3])
plt.xticks(np.arange(0, 16, step=1), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 'ALL'])
fig2.text(0.5, 0.02, 'Number of columns released', ha='center', size=14)
fig2.text(0.04, 0.5, 'False Miss(%)', va='center', rotation='vertical', size=14)
ax2.legend()
plt.show()


