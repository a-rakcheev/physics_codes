# plot path in (partial) phase space with the acceleration function in the background
# simulation with damping

import sys
sys.path.append("E:/Dropbox/data/dipoles/repo/code")
import dipole_functions as dpl
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rc
rc('text', usetex=True)
matplotlib.rcParams['text.latex.preamble'].join([r'\usepackage{amsfonts}', r'\usepackage{amsmath}', r'\usepackage{amssymb}'])
import matplotlib.colors as colors
from matplotlib import cm

# parameters
q_set = 10.0                            # qa factor
z_0_set = 0.1                           # plate distance
eta = 1.0                               # damping 
theta = 45                              # angle of dipole relative to x-axis
factor1 = 0.5                           # initial condition for phi1 - phi1(0) = f1 * pi
factor2 = 0.                            # initial condition for phi2 - phi2(0) = f2 * pi
t_final = 100.0                         # simulation time

L = 2                                   # number of dipoles
r_max = 20                              # maximal radius
r_res = 21                              # radial res
gc_order = 10000                        # gauss chebyshev order

labelsize = 14
ticksize = 13

prefix = "E:/Dropbox/data/dipoles/repo/data/"

fig = plt.figure(1, figsize=(4, 8))

# load couplings
if q_set == 0:

        K_p_xx = np.zeros((r_res, 360))
        K_p_yy = np.zeros((r_res, 360))
        K_p_xy = np.zeros((r_res, 360))

        K_m_xx = np.zeros((r_res, 360))
        K_m_yy = np.zeros((r_res, 360))
        K_m_xy = np.zeros((r_res, 360))


        rl = np.linspace(0, r_max, r_res)
        dtheta = 2 * np.pi / 360
        thetal = np.linspace(0., 2. * np.pi - dtheta, 360)

        # dipole-dipole interaction
        Fxl = np.zeros((r_res, 360))
        Fyl = np.zeros((r_res, 360))
        Fxyl = np.zeros((r_res, 360))

        for i, r_set in enumerate(rl):
            for j, theta_set in enumerate(thetal):

                lattice_vector = np.array([np.cos(theta_set), np.sin(theta_set), 0.])

                if r_set == 0.:

                    continue

                else:
                    mat = dpl.dipole_field_matrix(r_set * lattice_vector)
                    Fxl[i, j] = mat[0, 0]
                    Fyl[i, j] = mat[1, 1]
                    Fxyl[i, j] = mat[0, 1]


        C_p_xx = K_p_xx + Fxl
        C_p_yy = K_p_yy + Fyl
        C_p_xy = K_p_xy + Fxyl


else:
    name = prefix + "couplings_polar_nonreciprocal_q=" + str(q_set).replace(".", "-") + "_z0=" + str(z_0_set).replace(".", "-") + "_rmax=" + str(r_max) + "_rres=" + str(r_res) + "_gc_order=" + str(gc_order)

    data = np.load(name + ".npz")
    K_p_xx_part = data["K_p_xx"]
    K_p_yy_part = data["K_p_yy"]
    K_p_xy_part = data["K_p_xy"]

    K_m_xx_part = data["K_m_xx"]
    K_m_yy_part = data["K_m_yy"]
    K_m_xy_part = data["K_m_xy"]


    # fill by symmetry
    K_p_xx = np.zeros((r_res, 360))
    K_p_yy = np.zeros((r_res, 360))
    K_p_xy = np.zeros((r_res, 360))

    K_m_xx = np.zeros((r_res, 360))
    K_m_yy = np.zeros((r_res, 360))
    K_m_xy = np.zeros((r_res, 360))


    # plus part
    K_p_xx[:, 0:90] = K_p_xx_part
    K_p_xx[:, 90:180] = K_p_xx_part[:, ::-1]
    K_p_xx[:, 180:270] = K_p_xx_part
    K_p_xx[:, 270:360] = K_p_xx_part[:, ::-1]

    K_p_yy[:, 0:90] = K_p_yy_part
    K_p_yy[:, 90:180] = K_p_yy_part[:, ::-1]
    K_p_yy[:, 180:270] = K_p_yy_part
    K_p_yy[:, 270:360] = K_p_yy_part[:, ::-1]

    K_p_xy[:, 0:90] = K_p_xy_part
    K_p_xy[:, 90:180] = -K_p_xy_part[:, ::-1]
    K_p_xy[:, 180:270] = K_p_xy_part
    K_p_xy[:, 270:360] = -K_p_xy_part[:, ::-1]


    # minus part
    K_m_xx[:, 0:90] = K_m_xx_part
    K_m_xx[:, 90:180] = -K_m_xx_part[:, ::-1]
    K_m_xx[:, 180:270] = -K_m_xx_part
    K_m_xx[:, 270:360] = K_m_xx_part[:, ::-1]

    K_m_yy[:, 0:90] = K_m_yy_part
    K_m_yy[:, 90:180] = -K_m_yy_part[:, ::-1]
    K_m_yy[:, 180:270] = -K_m_yy_part
    K_m_yy[:, 270:360] = K_m_yy_part[:, ::-1]

    K_m_xy[:, 0:90] = K_m_xy_part
    K_m_xy[:, 90:180] = K_m_xy_part[:, ::-1]
    K_m_xy[:, 180:270] = -K_m_xy_part
    K_m_xy[:, 270:360] = -K_m_xy_part[:, ::-1]


    rl = np.linspace(0, r_max, r_res)
    dtheta = 2 * np.pi / 360
    thetal = np.linspace(0., 2. * np.pi - dtheta, 360)

    # dipole-dipole interaction
    Fxl = np.zeros((r_res, 360))
    Fyl = np.zeros((r_res, 360))
    Fxyl = np.zeros((r_res, 360))

    for i, r_set in enumerate(rl):
        for j, theta_set in enumerate(thetal):

            lattice_vector = np.array([np.cos(theta_set), np.sin(theta_set), 0.])

            if r_set == 0.:

                continue

            else:
                mat = dpl.dipole_field_matrix(r_set * lattice_vector)
                Fxl[i, j] = mat[0, 0]
                Fyl[i, j] = mat[1, 1]
                Fxyl[i, j] = mat[0, 1]


    C_p_xx = K_p_xx + Fxl
    C_p_yy = K_p_yy + Fyl
    C_p_xy = K_p_xy + Fxyl


    # pair couplings
    K_xx = np.zeros((L, 2))
    K_yy = np.zeros((L, 2))
    K_xy = np.zeros((L, 2))

    K_xx[:, 0] = C_p_xx[0:L, theta] + K_m_xx[0:L, theta]
    K_xx[:, 1] = C_p_xx[0:L, theta + 180] + K_m_xx[0:L, theta + 180]

    K_yy[:, 0] = C_p_yy[0:L, theta] + K_m_yy[0:L, theta]
    K_yy[:, 1] = C_p_yy[0:L, theta + 180] + K_m_yy[0:L, theta + 180]

    K_xy[:, 0] = C_p_xy[0:L, theta] + K_m_xy[0:L, theta]
    K_xy[:, 1] = C_p_xy[0:L, theta + 180] + K_m_xy[0:L, theta + 180]

# load simulation data
data = np.load(prefix + "nonreciprocal_dynamics_pair_q=" + str(q_set).replace(".", "-") + "_z0=" + str(z_0_set).replace(".", "-") + "_theta=" + str(theta) + "_eta=" + str(eta).replace(".", "-") + "_f1=" + str(round(factor1, 2)).replace(".", "-") 
+ "_f2=" + str(round(factor2, 2)).replace(".", "-") + "_T=" + str(t_final)  + ".npz")

tl = data["tl"]
phip_sim = np.mod(data["phi_p"], 2 * np.pi)
phim_sim = np.mod(data["phi_m"], 2 * np.pi)


# functions appearing in the eom for first and second dipole - use function of phi_{+} and phi_{-}
def eom_function_p(phi_p, phi_m):
    val = (K_xy[1, 1] + K_xy[1, 0]) * np.cos(phi_p) + 0.5 * (K_yy[1, 1] - K_xx[1, 1] + K_yy[1, 0] - K_xx[1, 0]) * np.sin(phi_p) + 0.5 * (K_yy[1, 0] + K_xx[1, 0] - K_yy[1, 1] - K_xx[1, 1]) * np.sin(phi_m)
    return val

def eom_function_m(phi_p, phi_m):
    val = (K_xy[1, 1] - K_xy[1, 0]) * np.cos(phi_p) + 0.5 * (K_yy[1, 1] - K_xx[1, 1] - K_yy[1, 0] + K_xx[1, 0]) * np.sin(phi_p) - 0.5 * (K_yy[1, 0] + K_xx[1, 0] + K_yy[1, 1] + K_xx[1, 1]) * np.sin(phi_m)
    return val


# value of eom function for background
steps = 501
phi = np.linspace(0., 2 * np.pi, steps)
K_phi_p = np.zeros((steps, steps))
K_phi_m = np.zeros((steps, steps))

for i, phi1 in enumerate(phi):
    for j, phi2 in enumerate(phi):

        K_phi_p[i, j] = eom_function_p(phi1, phi2)
        K_phi_m[i, j] = eom_function_m(phi1, phi2)


max_val = 100.0
cmap = "Spectral_r"                     # cmap for background
cmap_scatter = cm.get_cmap('Greys')     # cmap for path


plt.subplot(2, 1, 1)
# plot eom background
plt.pcolormesh(phi, phi, K_phi_p.T, cmap=cmap, vmin=-max_val, vmax=max_val)

# plot angular path
plt.scatter(phip_sim, phim_sim, s=5, marker="o", color=cmap_scatter(np.linspace(0., 1., len(tl))))

plt.xlim(0.45 * np.pi, 0.475 * np.pi)
plt.xticks([0.45 * np.pi, 0.475 * np.pi], [r"$\frac{9\pi}{20}$", r"$\frac{19\pi}{40}$"], fontsize=ticksize)

plt.ylim(0., 2.0 * np.pi)
plt.yticks([0., 0.5 * np.pi, np.pi, 1.5 * np.pi, 2 * np.pi], [r"$0$", r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{2}$", r"$2\pi$"], fontsize=ticksize)

plt.ylabel(r"$\varphi_{-}$", fontsize=labelsize)
plt.title(r"$\ddot{\varphi}_{+}$", fontsize=labelsize + 1)


plt.subplot(2, 1, 2)
# plot eom background
plot = plt.pcolormesh(phi, phi, K_phi_m.T, cmap=cmap, vmin=-max_val, vmax=max_val)

# plot angular path
plt.scatter(phip_sim, phim_sim, s=5, marker="o", color=cmap_scatter(np.linspace(0., 1., len(tl))))

plt.xlim(0.45 * np.pi, 0.475 * np.pi)
plt.xticks([0.45 * np.pi, 0.475 * np.pi], [r"$\frac{9\pi}{20}$", r"$\frac{19\pi}{40}$"], fontsize=ticksize)

plt.ylim(0., 2.0 * np.pi)
plt.yticks([0., 0.5 * np.pi, np.pi, 1.5 * np.pi, 2 * np.pi], [r"$0$", r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{2}$", r"$2\pi$"], fontsize=ticksize)

plt.xlabel(r"$\varphi_{+}$", fontsize=labelsize)
plt.ylabel(r"$\varphi_{-}$", fontsize=labelsize)
plt.title(r"$\ddot{\varphi}_{-}$", fontsize=labelsize + 1)

# colorbar axis
ax = fig.add_axes([0.125, 0.035, 0.825, 0.015])
cb = fig.colorbar(plot, cax=ax, orientation="horizontal")
cb.ax.tick_params(labelsize=ticksize - 1)

plt.subplots_adjust(hspace=0.2, bottom=0.15, top=0.95, left=0.125, right=0.95)

plt.savefig("Fig9.png", dpi=1200)
# plt.show()