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
q_set = 1.0
z_0_set = 0.1  
eta = 0.0
theta = 45
t_final = 100.0
factor_steps = 401

labelsize = 13
labelsize_orbit = 12
titlesize = 11

ticksize = 12
ticksize_orbit = 12

circlecol = "grey"
linecol = "grey"

prefix = "E:/Dropbox/data/dipoles/repo/data/"                       # repo data directory
fig = plt.figure(1, figsize=(7.5, 7.5))

# load data
data = np.load(prefix + "nonreciprocal_dynamics_pair_q=" + str(q_set).replace(".", "-") + "/nonreciprocal_dynamics_pair_energy_q=" + str(q_set).replace(".", "-") + "_z0=" + str(z_0_set).replace(".", "-")
    + "_theta=" + str(theta) + "_eta=" + str(eta).replace(".", "-") + "_tf=" + str(t_final).replace(".", "-") + "_f_steps=" + str(factor_steps) + ".npz")

# data = np.load(prefix + "nonreciprocal_dynamics_pair_q=" + str(q_set).replace(".", "-") + "/nonreciprocal_dynamics_with_sia_pair_energy_q=" + str(q_set).replace(".", "-") + "_z0=" + str(z_0_set).replace(".", "-")
#     + "_theta=" + str(theta) + "_eta=" + str(eta).replace(".", "-") + "_tf=" + str(t_final).replace(".", "-") + "_t_steps=" + str(10000) + "_f_steps=" + str(factor_steps) + ".npz")

ear = data["en"]
K_xx = data["K_xx"]
K_yy = data["K_yy"]
K_xy = data["K_xy"]

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



# main axis for ear
ax = plt.axes([0.075, 0.075, 0.6, 0.55])

# plot ear
plt.pcolormesh(np.linspace(0., 2.0 + 2 / (factor_steps - 1), factor_steps + 1) * np.pi, np.linspace(0., 2.0 + 2 / (factor_steps - 1), factor_steps + 1) * np.pi, 2 * np.sqrt(ear_pm / t_final).T, 
cmap="gnuplot", norm=colors.LogNorm(vmin=1.e-2, vmax=1.e2), zorder=1, antialiased=True)

cb = plt.colorbar()
cb.ax.tick_params(labelsize=ticksize - 1)

plt.xlim(0., 2.0 * np.pi)
plt.ylim(0., 2.0 * np.pi)

plt.xticks([0., 0.5 * np.pi, np.pi, 1.5 * np.pi, 2 * np.pi], [r"$0$", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"], fontsize=ticksize)
plt.yticks([0., 0.5 * np.pi, np.pi, 1.5 * np.pi, 2 * np.pi], [r"$0$", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"], fontsize=ticksize)

plt.xlabel(r"$\varphi_{+}$", fontsize=labelsize)
plt.ylabel(r"$\varphi_{-}$", fontsize=labelsize, labelpad=-6)

circle1 = patches.Circle((0.5 * np.pi, 0.8 * np.pi), radius=0.1, edgecolor=circlecol, fill=False, transform=ax.transData)
ax.add_patch(circle1)

circle2 = patches.Circle((0.2 * np.pi, 1.4 * np.pi), radius=0.1, edgecolor=circlecol, fill=False, transform=ax.transData)
ax.add_patch(circle2)

circle3 = patches.Circle((1 * np.pi, 1.6 * np.pi), radius=0.1, edgecolor=circlecol, fill=False, transform=ax.transData)
ax.add_patch(circle3)

circle4 = patches.Circle((1.5 * np.pi, 0.9 * np.pi), radius=0.1, edgecolor=circlecol, fill=False, transform=ax.transData)
ax.add_patch(circle4)

circle5 = patches.Circle((1.75 * np.pi, 0.05 * np.pi), radius=0.1, edgecolor=circlecol, fill=False, transform=ax.transData)
ax.add_patch(circle5)

# orbits
cmap_scatter = cm.get_cmap('RdYlGn')

# phi_+ = 0.5 pi, phi_- = 0.75 pi <-> phi_1 = 0.625 pi, phi_2 = 1.875 pi
factor1 = 0.65
factor2 = 1.85

data = np.load(prefix + "nonreciprocal_dynamics_pair_q=" + str(q_set).replace(".", "-") + "/nonreciprocal_dynamics_pair_q=" + str(q_set).replace(".", "-") + "_z0=" + str(z_0_set).replace(".", "-") + "_theta=" + str(theta) + "_eta=" + str(eta).replace(".", "-")
+ "_f1=" + str(round(factor1, 2)).replace(".", "-") + "_f2=" + str(round(factor2, 2)).replace(".", "-") + ".npz")

tl = data["t"]
yl = data["y"]
K_xx = data["K_xx"]
K_yy = data["K_yy"]
K_xy = data["K_xy"]

steps = len(tl)
phi1_sim = np.mod(yl[0, :], 2 * np.pi)
phi2_sim = np.mod(yl[1, :], 2 * np.pi)
phip_sim = np.mod(yl[0, :] + yl[1, :], 2 * np.pi)
phim_sim = np.mod(yl[0, :] - yl[1, :], 2 * np.pi)

ax = plt.axes([0.075, 0.725, 0.25, 0.25])
max_val = 100.0
plt.scatter(phip_sim, phim_sim, s=1, marker="o", color=cmap_scatter(np.linspace(0., 1., len(tl))))


plt.xlim(0., 2.0 * np.pi)
plt.ylim(0., 2.0 * np.pi)

plt.xticks([0., np.pi, 2 * np.pi], [r"$0$", r"$\pi$", r"$2\pi$"], fontsize=ticksize_orbit)
plt.yticks([0., np.pi, 2 * np.pi], [r"$0$", r"$\pi$", r"$2\pi$"], fontsize=ticksize_orbit)

# plt.xlabel(r"$\varphi_{+}$", fontsize=labelsize_orbit)
plt.ylabel(r"$\varphi_{-}$", fontsize=labelsize_orbit)
plt.title(r"$a$", fontsize=titlesize)


# phi_+ = 1.1 pi, phi_- = 1.7 pi <-> phi_1 = 0.8 pi, phi_2 = 1.4 pi
factor1 = 0.8
factor2 = 1.4

data = np.load(prefix + "nonreciprocal_dynamics_pair_q=" + str(q_set).replace(".", "-") + "/nonreciprocal_dynamics_pair_q=" + str(q_set).replace(".", "-") + "_z0=" + str(z_0_set).replace(".", "-") + "_theta=" + str(theta) + "_eta=" + str(eta).replace(".", "-")
+ "_f1=" + str(round(factor1, 2)).replace(".", "-") + "_f2=" + str(round(factor2, 2)).replace(".", "-") + ".npz")

tl = data["t"]
yl = data["y"]
K_xx = data["K_xx"]
K_yy = data["K_yy"]
K_xy = data["K_xy"]

steps = len(tl)
phi1_sim = np.mod(yl[0, :], 2 * np.pi)
phi2_sim = np.mod(yl[1, :], 2 * np.pi)
phip_sim = np.mod(yl[0, :] + yl[1, :], 2 * np.pi)
phim_sim = np.mod(yl[0, :] - yl[1, :], 2 * np.pi)

ax = plt.axes([0.4, 0.725, 0.25, 0.25])
max_val = 100.0
plt.scatter(phip_sim, phim_sim, s=1, marker="o", color=cmap_scatter(np.linspace(0., 1., len(tl))))


plt.xlim(0., 2.0 * np.pi)
plt.ylim(0., 2.0 * np.pi)

plt.xticks([0., np.pi, 2 * np.pi], [r"$0$", r"$\pi$", r"$2\pi$"], fontsize=ticksize_orbit)
plt.yticks([0., np.pi, 2 * np.pi], [r"$0$", r"$\pi$", r"$2\pi$"], fontsize=ticksize_orbit)

# plt.xlabel(r"$\varphi_{+}$", fontsize=labelsize_orbit)
# plt.ylabel(r"$\varphi_{-}$", fontsize=labelsize_orbit)

plt.title(r"$b$", fontsize=titlesize)

# phi_+ = 1 pi, phi_- = 1.5 pi <-> phi_1 = 1.25 pi, phi_2 = 1.75 pi
factor1 = 1.3
factor2 = 1.7

data = np.load(prefix + "nonreciprocal_dynamics_pair_q=" + str(q_set).replace(".", "-") + "/nonreciprocal_dynamics_pair_q=" + str(q_set).replace(".", "-") + "_z0=" + str(z_0_set).replace(".", "-") + "_theta=" + str(theta) + "_eta=" + str(eta).replace(".", "-")
+ "_f1=" + str(round(factor1, 2)).replace(".", "-") + "_f2=" + str(round(factor2, 2)).replace(".", "-") + ".npz")

tl = data["t"]
yl = data["y"]
K_xx = data["K_xx"]
K_yy = data["K_yy"]
K_xy = data["K_xy"]

steps = len(tl)
phi1_sim = np.mod(yl[0, :], 2 * np.pi)
phi2_sim = np.mod(yl[1, :], 2 * np.pi)
phip_sim = np.mod(yl[0, :] + yl[1, :], 2 * np.pi)
phim_sim = np.mod(yl[0, :] - yl[1, :], 2 * np.pi)

ax = plt.axes([0.725, 0.725, 0.25, 0.25])
max_val = 100.0
plt.scatter(phip_sim, phim_sim, s=1, marker="o", color=cmap_scatter(np.linspace(0., 1., len(tl))))


plt.xlim(0., 2.0 * np.pi)
plt.ylim(0., 2.0 * np.pi)

plt.xticks([0., np.pi, 2 * np.pi], [r"$0$", r"$\pi$", r"$2\pi$"], fontsize=ticksize_orbit)
plt.yticks([0., np.pi, 2 * np.pi], [r"$0$", r"$\pi$", r"$2\pi$"], fontsize=ticksize_orbit)

# plt.xlabel(r"$\varphi_{+}$", fontsize=labelsize_orbit)
# plt.ylabel(r"$\varphi_{-}$", fontsize=labelsize_orbit)
plt.title(r"$c$", fontsize=titlesize)


# phi_+ = 1.5 pi, phi_- = 1 pi <-> phi_1 = 1.25 pi, phi_2 = 0.25 pi
factor1 = 1.2
factor2 = 0.3

data = np.load(prefix + "nonreciprocal_dynamics_pair_q=" + str(q_set).replace(".", "-") + "/nonreciprocal_dynamics_pair_q=" + str(q_set).replace(".", "-") + "_z0=" + str(z_0_set).replace(".", "-") + "_theta=" + str(theta) + "_eta=" + str(eta).replace(".", "-")
+ "_f1=" + str(round(factor1, 2)).replace(".", "-") + "_f2=" + str(round(factor2, 2)).replace(".", "-") + ".npz")

tl = data["t"]
yl = data["y"]
K_xx = data["K_xx"]
K_yy = data["K_yy"]
K_xy = data["K_xy"]

steps = len(tl)
phi1_sim = np.mod(yl[0, :], 2 * np.pi)
phi2_sim = np.mod(yl[1, :], 2 * np.pi)
phip_sim = np.mod(yl[0, :] + yl[1, :], 2 * np.pi)
phim_sim = np.mod(yl[0, :] - yl[1, :], 2 * np.pi)

ax = plt.axes([0.725, 0.4, 0.25, 0.25])
max_val = 100.0
plt.scatter(phip_sim, phim_sim, s=1, marker="o", color=cmap_scatter(np.linspace(0., 1., len(tl))))


plt.xlim(0., 2.0 * np.pi)
plt.ylim(0., 2.0 * np.pi)

plt.xticks([0., np.pi, 2 * np.pi], [r"$0$", r"$\pi$", r"$2\pi$"], fontsize=ticksize_orbit)
plt.yticks([0., np.pi, 2 * np.pi], [r"$0$", r"$\pi$", r"$2\pi$"], fontsize=ticksize_orbit)

# plt.xlabel(r"$\varphi_{+}$", fontsize=labelsize_orbit)
# plt.ylabel(r"$\varphi_{-}$", fontsize=labelsize_orbit)
plt.title(r"$d$", fontsize=titlesize)


# phi_+ = 1.75 pi, phi_- = 0 pi <-> phi_1 = 0.875 pi, phi_2 = 0.875 pi
factor1 = 0.9
factor2 = 0.85

data = np.load(prefix + "nonreciprocal_dynamics_pair_q=" + str(q_set).replace(".", "-") + "/nonreciprocal_dynamics_pair_q=" + str(q_set).replace(".", "-") + "_z0=" + str(z_0_set).replace(".", "-") + "_theta=" + str(theta) + "_eta=" + str(eta).replace(".", "-")
+ "_f1=" + str(round(factor1, 2)).replace(".", "-") + "_f2=" + str(round(factor2, 2)).replace(".", "-") + ".npz")

tl = data["t"]
yl = data["y"]
K_xx = data["K_xx"]
K_yy = data["K_yy"]
K_xy = data["K_xy"]

steps = len(tl)
phi1_sim = np.mod(yl[0, :], 2 * np.pi)
phi2_sim = np.mod(yl[1, :], 2 * np.pi)
phip_sim = np.mod(yl[0, :] + yl[1, :], 2 * np.pi)
phim_sim = np.mod(yl[0, :] - yl[1, :], 2 * np.pi)

ax = plt.axes([0.725, 0.075, 0.25, 0.25])
max_val = 100.0
plt.scatter(phip_sim, phim_sim, s=1, marker="o", color=cmap_scatter(np.linspace(0., 1., len(tl))))


plt.xlim(0., 2.0 * np.pi)
plt.ylim(0., 2.0 * np.pi)

plt.xticks([0., np.pi, 2 * np.pi], [r"$0$", r"$\pi$", r"$2\pi$"], fontsize=ticksize_orbit)
plt.yticks([0., np.pi, 2 * np.pi], [r"$0$", r"$\pi$", r"$2\pi$"], fontsize=ticksize_orbit)

plt.xlabel(r"$\varphi_{+}$", fontsize=labelsize_orbit)
# plt.ylabel(r"$\varphi_{-}$", fontsize=labelsize_orbit)
plt.title(r"$e$", fontsize=titlesize)

fig.add_artist(lines.Line2D([0.195, 0.195], [0.305, 0.675], color=linecol, lw=1))
fig.add_artist(lines.Line2D([0.132, 0.475], [0.4625, 0.675], color=linecol, lw=1))
fig.add_artist(lines.Line2D([0.325, 0.7], [0.52, 0.7], color=linecol, lw=1))
fig.add_artist(lines.Line2D([0.445, 0.7], [0.325, 0.5], color=linecol, lw=1))
fig.add_artist(lines.Line2D([0.505, 0.675], [0.09, 0.2], color=linecol, lw=1))

plt.savefig("Fig4.png", format="png", dpi=1200)
plt.show()