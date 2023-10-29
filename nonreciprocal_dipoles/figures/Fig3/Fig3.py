import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.patches as patches
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from matplotlib import rc
import matplotlib.lines as lines
rc('text', usetex=True)
matplotlib.rcParams['text.latex.preamble'].join([r'\usepackage{amsfonts}', r'\usepackage{amsmath}', r'\usepackage{amssymb}'])

# parameters
eta = 0.0
q_set = 1.0
z_0_set = 0.1  # plate distance
theta = 45

labelsize = 11
ticksize = 11
titlesize = 12
legendsize = 10
linewidth = 1

circlecol = "grey"
linecol = "grey"

prefix = "E:/Dropbox/data/dipoles/repo/data/"
fig = plt.figure(1, figsize=(7.5, 7.5))

# phi_+ = 1.1 pi, phi_- = 1.7 pi <-> phi_1 = 0.8 pi, phi_2 = 1.4 pi
factor1 = 0.8
factor2 = 1.4

data = np.load(prefix + "nonreciprocal_dynamics_pair_q=" + str(q_set).replace(".", "-") + "/nonreciprocal_dynamics_pair_q=" + str(q_set).replace(".", "-") + "_z0=" + str(z_0_set).replace(".", "-") + "_theta=" + str(theta) + "_eta=" + str(eta).replace(".", "-")
+ "_f1=" + str(round(factor1, 2)).replace(".", "-") + "_f2=" + str(round(factor2, 2)).replace(".", "-") + ".npz")

tl = data["t"]
yl = data["y"]

steps = len(tl)
phi_1 = yl[0, :]
phi_2 = yl[1, :]
phi_p = np.mod(phi_1 + phi_2, 2 * np.pi)
phi_m = np.mod(phi_1 - phi_2, 2 * np.pi)

omega_1 = yl[2, :]
omega_2 = yl[3, :]
omega_p = omega_1 + omega_2
omega_m = omega_1 - omega_2

# plot angular orbit and initial and final state
plt.subplot(2, 2, 1)
plt.plot(tl, phi_m, color="black", label=r"$\varphi_{-}$", lw=linewidth, zorder=2)
plt.plot(tl, phi_p,  color="red", label=r"$\varphi_{+}$", lw=linewidth, zorder=2)


plt.legend(fontsize=legendsize, loc="upper right", handlelength=0.75)
plt.title(r"$\varphi_{\pm}$", fontsize=titlesize)
plt.xticks(fontsize=ticksize)
plt.yticks([0., np.pi, 2 * np.pi], [r"$0$", r"$\pi$", r"$2\pi$"], fontsize=ticksize)

# plot angular orbit and initial and final state
plt.subplot(2, 2, 2)
plt.plot(tl, omega_m, color="black", label=r"$\omega_{-}$", lw=linewidth, zorder=2)
plt.plot(tl, omega_p,  color="red", label=r"$\omega_{+}$", lw=linewidth, zorder=2)


plt.xticks(fontsize=ticksize)
plt.yticks(fontsize=ticksize)


# plt.grid(zorder=1)
plt.legend(fontsize=legendsize, loc="upper right", handlelength=0.75)
plt.title(r"$\omega_{\pm}$", fontsize=titlesize)


# phi_+ = 0.5 pi, phi_- = 0.75 pi <-> phi_1 = 0.625 pi, phi_2 = 1.875 pi
factor1 = 0.65
factor2 = 1.85

data = np.load(prefix + "nonreciprocal_dynamics_pair_q=" + str(q_set).replace(".", "-") + "/nonreciprocal_dynamics_pair_q=" + str(q_set).replace(".", "-") + "_z0=" + str(z_0_set).replace(".", "-") + "_theta=" + str(theta) + "_eta=" + str(eta).replace(".", "-")
+ "_f1=" + str(round(factor1, 2)).replace(".", "-") + "_f2=" + str(round(factor2, 2)).replace(".", "-") + ".npz")

tl = data["t"]
yl = data["y"]

steps = len(tl)
phi_1 = yl[0, :]
phi_2 = yl[1, :]
phi_p = np.mod(phi_1 + phi_2, 2 * np.pi)
phi_m = np.mod(phi_1 - phi_2, 2 * np.pi)

omega_1 = yl[2, :]
omega_2 = yl[3, :]
omega_p = omega_1 + omega_2
omega_m = omega_1 - omega_2

plt.subplot(2, 2, 3)
plt.plot(tl, phi_m, color="black", label=r"$\varphi_{-}$", lw=linewidth, zorder=2)
plt.plot(tl, phi_p,  color="red", label=r"$\varphi_{+}$", lw=linewidth, zorder=2)


plt.legend(fontsize=legendsize, loc="upper right", handlelength=0.75)
plt.xlabel(r"$\tau$", fontsize=labelsize)
plt.xticks(fontsize=ticksize)
plt.yticks([0., np.pi, 2 * np.pi], [r"$0$", r"$\pi$", r"$2\pi$"], fontsize=ticksize)

# plot angular orbit and initial and final state
plt.subplot(2, 2, 4)
plt.plot(tl, omega_m, color="black", label=r"$\omega_{-}$", lw=linewidth, zorder=2)
plt.plot(tl, omega_p,  color="red", label=r"$\omega_{+}$", lw=linewidth, zorder=2)

# plt.grid(zorder=1)
plt.legend(fontsize=legendsize, loc="upper right", handlelength=0.75)
plt.xlabel(r"$\tau$", fontsize=labelsize)
plt.xticks(fontsize=ticksize)
plt.yticks(fontsize=ticksize)

plt.subplots_adjust(wspace=0.2, hspace=0.2, top=0.95, bottom=0.075, left=0.075, right=0.975)
plt.savefig("Fig3.pdf", format="pdf")
plt.show()