import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import rc
rc('text', usetex=True)
matplotlib.rcParams['text.latex.preamble'].join([r'\usepackage{amsfonts}', r'\usepackage{amsmath}', r'\usepackage{amssymb}'])

# parameters
ql = [1.0, 5.0, 10.0, 50.0, 100.0, 500.0]
z_0_set = 0.1  
eta = 0.0
theta = 45
t_final = 100.0
factor_steps = 401

labelsize = 14
ticksize = 13

prefix = "E:/Dropbox/data/dipoles/repo/data/"
fig = plt.figure(1, figsize=(8, 6))

for k, q_set in enumerate(ql):

    # load data
    data = np.load(prefix + "nonreciprocal_dynamics_pair_energy_q=" + str(q_set).replace(".", "-") + "_z0=" + str(z_0_set).replace(".", "-") + "_theta=" + str(theta) + "_eta=" + str(eta).replace(".", "-") + "_tf=" + str(t_final).replace(".", "-") + "_f_steps=" + str(factor_steps) + ".npz")

    ear = data["en"]

    # fill upper empty half in ear data
    # upper left quarter
    ear[0:(factor_steps - 1) // 2 + 1, (factor_steps - 1) // 2 + 1:factor_steps] = ear[(factor_steps - 1) // 2:factor_steps, 1:((factor_steps - 1) // 2 + 1)]

    # upper right quarter
    ear[(factor_steps - 1) // 2 + 1:factor_steps, (factor_steps - 1) // 2 + 1:factor_steps] = ear[1:(factor_steps - 1) // 2 + 1, 1:((factor_steps - 1) // 2 + 1)]

    # transform ear to plus/minus variables
    ear_pm = np.zeros((factor_steps, factor_steps))

    for i in range(factor_steps):
        for j in range(factor_steps):
            ear_pm[(i + j) % factor_steps, (i - j) % factor_steps] = ear[i, j]


    plt.subplot(2, 3, k + 1)

    # plot ear
    plot = plt.pcolormesh(np.linspace(0., 2.0 + 2 / (factor_steps - 1), factor_steps + 1) * np.pi, np.linspace(0., 2.0 + 2 / (factor_steps - 1), factor_steps + 1) * np.pi, 2 * np.sqrt(np.roll(np.roll(ear_pm, 100, axis=0), 200, axis=1) / t_final).T, 
    cmap="gnuplot", norm=colors.LogNorm(vmin=1.e-2, vmax=1.e2), zorder=1)

    if k >= 3:
        plt.xticks([0., np.pi, 2 * np.pi], [r"$-\pi / 2$", r"$\pi / 2$", r"$3\pi / 2$"], fontsize=ticksize)
        plt.xlabel(r"$\varphi_{+}$", fontsize=labelsize)

    else:
        plt.xticks([0., np.pi, 2 * np.pi], [], fontsize=ticksize)

    if k == 0 or k == 3:
        plt.yticks([0., np.pi, 2 * np.pi], [r"$-\pi$", r"$0$", r"$\pi$"], fontsize=ticksize)
        plt.ylabel(r"$\varphi_{-}$", fontsize=labelsize)

    else:
        plt.yticks([0., np.pi, 2 * np.pi], [], fontsize=ticksize)

    plt.xlim(0., 2.0 * np.pi)
    plt.ylim(0., 2.0 * np.pi)
    plt.title(r"$q=" + str(int(q_set)) + r"$", fontsize=labelsize)

# colorbar
ax = fig.add_axes([0.07, 0.04, 0.9, 0.02])
cb = fig.colorbar(plot, cax=ax, orientation="horizontal")

cb.ax.tick_params(labelsize=ticksize)
plt.subplots_adjust(wspace=0.25, hspace=0.2, left=0.07, right=0.97, bottom=0.16, top=0.95)
plt.savefig("Fig6.png", format="png", dpi=1200)
plt.show()