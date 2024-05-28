from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns

#plt.rcParams['axes.grid'] = True
#plt.rcParams["legend.loc"] = 'upper right'
# plt.style.use('seaborn-colorblind')
sns.set_style("whitegrid")

fig, ax = plt.subplots(2, 2, sharex='col', sharey='row')

x = np.array([0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.40, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85,
              0.9, 0.95, 1])
x = x*100
n_exp = 20
y = [None, None, None, None]

# !!!! RESULTS FOR L=8 !!!!
# gamma = 5
y[0] = n_exp - np.flip([20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 19, 19, 13, 0, 0])
# ----------------------------------------------------------------------------- #
# ----------------------------------------------------------------------------- #
# gamma = 10
y[1] = n_exp - np.array([0, 0, 0, 6, 16, 17, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20])
# ----------------------------------------------------------------------------- #
# ----------------------------------------------------------------------------- #
# gamma = 20
y[2] = n_exp - np.array([0, 0, 0, 0, 1, 5, 8, 12, 16, 17, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20])
# ----------------------------------------------------------------------------- #
# ----------------------------------------------------------------------------- #
# gamma = 40
y[3] = n_exp - np.array([0, 0, 0, 0, 0, 0, 0, 1, 2, 5, 6, 9, 12, 16, 18, 18, 17, 17, 20, 20, 20])
# ----------------------------------------------------------------------------- #
# ----------------------------------------------------------------------------- #
# gamma = 7 - not 100% successful anymore

# todo test how they look all in one plot
ax[0, 0].plot(x, y[0])
#ax[0, 0].plot(x, y[1])
# todo end of test

fig.text(0.5, 0.02, 'Size of the subset(%)', ha='center')
fig.text(0.04, 0.5, 'Detected fingerprints(%)', va='center', rotation='vertical')

#plt.show()

fig2, ax2 = plt.subplots(1, 1, sharex='col', sharey='row')
ax2.plot(x, y[0]/n_exp, label='$\gamma$ = 5')#, c='0.15')
ax2.plot(x, y[1]/n_exp, label='$\gamma$ = 10')#, c='0.35')
ax2.plot(x, y[2]/n_exp, label='$\gamma$ = 20')#, c='0.65')
ax2.plot(x, y[3]/n_exp, label='$\gamma$ = 40')#, c='0.85')
# ax2.plot(x, y[3])
fig2.text(0.5, 0.02, 'Size of the subset released(%)', ha='center', size=14)
fig2.text(0.04, 0.5, 'False Miss', va='center', rotation='vertical', size=14)
ax2.legend()
plt.show()

