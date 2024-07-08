import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib import cm
plt.rcParams['axes.grid'] = True
sns.set_style("whitegrid")
cmap = cm.get_cmap('plasma') # autumn
colors = [cmap(i*(1/5)) for i in range(4)]

x = 100 - np.flip(np.array([0, 0.02, 0.06, 0.1, 0.14, 0.18, 0.22, 0.26, 0.3])) * 100
# we omit further examples because changing more than 60% of data set's values is arguably of no
# significant value for the data user
y = [None, None, None, None]  # detected fingerprints

n_exp = 20

# !!!! RESULTS FOR BREAST CANCER L=8!!!!
# --------------------------------------- #
# gamma = 5
# --------------------------------------- #
y[0] = 20 - np.array([20, 20, 20, 20, 20, 20, 20, 20, 20])
# --------------------------------------- #
# gamma = 10
# --------------------------------------- #
y[1] = 20 - np.array([17, 20, 20, 20, 20, 20, 20, 20, 20])
# --------------------------------------- #
# gamma = 20
# --------------------------------------- #
y[2] = 20 - np.array([3, 10, 13, 12, 19, 16, 19, 19, 20]) # [3, 10, 13, 12, 19, 16, 19, 19]
# --------------------------------------- #
# gamma = 40
# --------------------------------------- #
y[3] = 20 - np.array([1, 5, 8, 11, 15, 17, 19, 20, 20])

fig, ax = plt.subplots(1, 1, sharex='col', sharey='row')
ax.set_xlabel("Portion of the unchanged data(%)", size=14)
ax.set_ylabel("False Miss", size=14)
ax.set_prop_cycle(color=colors)
ax.plot(x, y[0]/n_exp, label='$\gamma$ = 5')#, c='0.15')
ax.plot(x, y[1]/n_exp, label='$\gamma$ = 10')#, c='0.35')
ax.plot(x, y[2]/n_exp, label='$\gamma$ = 20')#, c='0.65')
ax.plot(x, y[3]/n_exp, label='$\gamma$ = 40')#, c='0.85')
ax.legend()
plt.show()