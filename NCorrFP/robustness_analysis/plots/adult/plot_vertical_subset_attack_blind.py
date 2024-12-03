from matplotlib import pyplot as plt
from matplotlib import cm
import numpy as np
import seaborn as sns

#plt.rcParams['axes.grid'] = True
#plt.style.use('seaborn-colorblind')
sns.set_style("whitegrid")
cmap = cm.get_cmap('plasma') # autumn
colors = [cmap(i*(1/5)) for i in range(4)]

#fig, ax = plt.subplots(2, 2, sharex='col', sharey='row')

x = np.array([i for i in range(16)])  # -> number of columns released (the results are in ascending order of
                                        # number of columns DELETED)
x_b = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15])
y = [None, None, None, None]
y_b = [None, None, None, None]

n_exp = 40
# RESULTS FOR ADULT L=64
# gamma = 5
y[0] = 100 - (np.array([0, 17, 17, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20]) +
                np.array([0, 17, 17, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20]) +
                np.array([0, 56, 58, 59, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60]))
y_b[0] = np.array([1, 1.0, 1.0, 0.38, 0.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
#----------------------------------------------------------------------------- #
# ----------------------------------------------------------------------------- #
# gamma = 10
y[1] = n_exp - np.array(np.array([0, 10, 13, 17, 16, 16, 14, 17, 20, 20, 20, 20, 20, 20, 20, 20]) + np.array([0, 14, 14, 15, 16, 18, 18, 18, 19, 20, 20, 20, 20, 20, 20, 20]))
y_b[1] = np.array([1, 1.0, 1.0, 0.98,  0.44, 0.08, 0.0, 0, 0, 0, 0, 0, 0, 0, 0])
# ----------------------------------------------------------------------------- #
# ----------------------------------------------------------------------------- #
# gamma = 20
y[2] = n_exp - np.array(np.array([0, 0, 3, 7, 6, 8, 5, 9, 18, 12, 17, 15, 20, 20, 20, 20]) + np.array([0, 4, 7, 6, 7, 10, 8, 13, 14, 20, 20, 20, 20, 20, 20, 20]))
#y_b[3] = np.array([])
y_b[2] = np.array([1, 1.0, 1.0,0.98,  0.92,  0.66,0.30,0.04,  0.0, 0, 0, 0, 0, 0, 0])
# ----------------------------------------------------------------------------- #
# ----------------------------------------------------------------------------- #
# gamma = 40
y[3] = 100 - ((np.array([0, 0, 0, 2, 2, 1, 2, 2, 7, 3, 4, 4, 10, 7, 12, 20]) +
                 np.array([0, 0, 1, 1, 0, 0, 2, 3, 3, 5, 6, 7, 6, 12, 19, 20])) +
                np.array([0, 3, 2, 6, 1, 15, 11, 17, 21, 31, 35, 39, 53, 57, 60, 60]))
y_b[3] = np.array([1, 1.0, 1.0, 1.0, 1.0,  1.0, 0.94, 0.8,  0.58, 0.24, 0.12, 0.02, 0.0, 0, 0])
# ----------------------------------------------------------------------------- #
# ----------------------------------------------------------------------------- #


#fig.text(0.5, 0.02, 'Number of columns released', ha='center')
#fig.text(0.04, 0.5, 'Detected fingerprints(%)', va='center', rotation='vertical')

# --------------------------------------------------------------------- #
# ----------------- ALL IN ONE FIGURE --------------------------------- #
# --------------------------------------------------------------------- #
fig2, ax2 = plt.subplots(1, 1, sharex='col', sharey='row')
ax2.set_prop_cycle(color=colors)
print(type(x))
print(type(y[0]/100))
ax2.plot(x_b, y_b[0], label='baselines (random FP)', color=colors[0], linestyle=(0, (5, 5)), linewidth=1)
ax2.plot(x, y[0]/100, label='$\gamma$ = 5')#, c='0.15')
ax2.plot(x_b, y_b[1], color=colors[1], linestyle=(0, (5, 5)), linewidth=1)
ax2.plot(x, y[1]/n_exp, label='$\gamma$ = 10')#, c='0.35')
ax2.plot(x_b, y_b[2], color=colors[2], linestyle=(0, (5, 5)), linewidth=1)
ax2.plot(x, y[2]/n_exp, label='$\gamma$ = 20')#, c='0.65')
ax2.plot(x_b, y_b[3], color=colors[3], linestyle=(0, (5, 5)), linewidth=1)
ax2.plot(x, y[3]/100, label='$\gamma$ = 40')#, c='0.85')
# ax2.plot(x, y[3])
plt.xticks(np.arange(0, 16, step=1), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 'ALL'])
fig2.text(0.5, 0.02, 'Number of columns released', ha='center', size=14)
fig2.text(0.04, 0.5, 'False Miss(%)', va='center', rotation='vertical', size=14)
ax2.legend()
plt.show()


