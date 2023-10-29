
import sys
sys.path.append("E:/Dropbox/data/dipoles/repo/code/")
import numpy as np
import dipole_functions as dpl
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import rc
rc('text', usetex=True)
matplotlib.rcParams['text.latex.preamble'].join([r'\usepackage{amsfonts}', r'\usepackage{amsmath}', r'\usepackage{amssymb}'])


fig = plt.figure(1, figsize=(4.5, 5))
z_0_set = 0.1  # plate distance
r_max = 20  # maximal radius
r_res = 21  # radial
r_cut = 10
theta_res = 90
gc_order = 10000
thresh = 1.e-2
fontsize = 15
labelsize = 11


for q_set in [10.0]:
    plt.clf()

    if q_set == 0.:

        # fill by symmetry
        K_p_xx = np.zeros((r_res, 360))
        K_p_yy = np.zeros((r_res, 360))
        K_p_zz = np.zeros((r_res, 360))
        K_p_xy = np.zeros((r_res, 360))
        K_p_xz = np.zeros((r_res, 360))
        K_p_yz = np.zeros((r_res, 360))

        K_m_xx = np.zeros((r_res, 360))
        K_m_yy = np.zeros((r_res, 360))
        K_m_zz = np.zeros((r_res, 360))
        K_m_xy = np.zeros((r_res, 360))
        K_m_xz = np.zeros((r_res, 360))
        K_m_yz = np.zeros((r_res, 360))


    else:

        prefix = "E:/Dropbox/data/dipoles/repo/data/"                       # repo data directory

        name = prefix + "couplings_polar_nonreciprocal_q=" + str(q_set).replace(".", "-") + "_z0=" + str(z_0_set).replace(".", "-") + "_rmax=" + str(r_max) + "_rres=" + str(r_res) + "_gc_order=" + str(gc_order)

        data = np.load(name + ".npz")
        K_p_xx_part = data["K_p_xx"]
        K_p_yy_part = data["K_p_yy"]
        K_p_zz_part = data["K_p_zz"]
        K_p_xy_part = data["K_p_xy"]
        K_p_xz_part = data["K_p_xz"]
        K_p_yz_part = data["K_p_yz"]

        K_m_xx_part = data["K_m_xx"]
        K_m_yy_part = data["K_m_yy"]
        K_m_zz_part = data["K_m_zz"]
        K_m_xy_part = data["K_m_xy"]
        K_m_xz_part = data["K_m_xz"]
        K_m_yz_part = data["K_m_yz"]


        # fill by symmetry
        K_p_xx = np.zeros((r_res, 360))
        K_p_yy = np.zeros((r_res, 360))
        K_p_zz = np.zeros((r_res, 360))
        K_p_xy = np.zeros((r_res, 360))
        K_p_xz = np.zeros((r_res, 360))
        K_p_yz = np.zeros((r_res, 360))

        K_m_xx = np.zeros((r_res, 360))
        K_m_yy = np.zeros((r_res, 360))
        K_m_zz = np.zeros((r_res, 360))
        K_m_xy = np.zeros((r_res, 360))
        K_m_xz = np.zeros((r_res, 360))
        K_m_yz = np.zeros((r_res, 360))


        # plus part
        K_p_xx[:, 0:90] = K_p_xx_part
        K_p_xx[:, 90:180] = K_p_xx_part[:, ::-1]
        K_p_xx[:, 180:270] = K_p_xx_part
        K_p_xx[:, 270:360] = K_p_xx_part[:, ::-1]

        K_p_yy[:, 0:90] = K_p_yy_part
        K_p_yy[:, 90:180] = K_p_yy_part[:, ::-1]
        K_p_yy[:, 180:270] = K_p_yy_part
        K_p_yy[:, 270:360] = K_p_yy_part[:, ::-1]

        K_p_zz[:, 0:90] = K_p_zz_part
        K_p_zz[:, 90:180] = K_p_zz_part[:, ::-1]
        K_p_zz[:, 180:270] = K_p_zz_part
        K_p_zz[:, 270:360] = K_p_zz_part[:, ::-1]

        K_p_xy[:, 0:90] = K_p_xy_part
        K_p_xy[:, 90:180] = -K_p_xy_part[:, ::-1]
        K_p_xy[:, 180:270] = K_p_xy_part
        K_p_xy[:, 270:360] = -K_p_xy_part[:, ::-1]

        K_p_xz[:, 0:90] = K_p_xz_part
        K_p_xz[:, 90:180] = K_p_xz_part[:, ::-1]
        K_p_xz[:, 180:270] = K_p_xz_part
        K_p_xz[:, 270:360] = K_p_xz_part[:, ::-1]

        K_p_yz[:, 0:90] = K_p_yz_part
        K_p_yz[:, 90:180] = -K_p_yz_part[:, ::-1]
        K_p_yz[:, 180:270] = K_p_yz_part
        K_p_yz[:, 270:360] = -K_p_yz_part[:, ::-1]


        # minus part
        K_m_xx[:, 0:90] = K_m_xx_part
        K_m_xx[:, 90:180] = -K_m_xx_part[:, ::-1]
        K_m_xx[:, 180:270] = -K_m_xx_part
        K_m_xx[:, 270:360] = K_m_xx_part[:, ::-1]

        K_m_yy[:, 0:90] = K_m_yy_part
        K_m_yy[:, 90:180] = -K_m_yy_part[:, ::-1]
        K_m_yy[:, 180:270] = -K_m_yy_part
        K_m_yy[:, 270:360] = K_m_yy_part[:, ::-1]

        K_m_zz[:, 0:90] = K_m_zz_part
        K_m_zz[:, 90:180] = K_m_zz_part[:, ::-1]
        K_m_zz[:, 180:270] = -K_m_zz_part
        K_m_zz[:, 270:360] = -K_m_zz_part[:, ::-1]

        K_m_xy[:, 0:90] = K_m_xy_part
        K_m_xy[:, 90:180] = K_m_xy_part[:, ::-1]
        K_m_xy[:, 180:270] = -K_m_xy_part
        K_m_xy[:, 270:360] = -K_m_xy_part[:, ::-1]

        K_m_xz[:, 0:90] = K_m_xz_part
        K_m_xz[:, 90:180] = -K_m_xz_part[:, ::-1]
        K_m_xz[:, 180:270] = -K_m_xz_part
        K_m_xz[:, 270:360] = K_m_xz_part[:, ::-1]

        K_m_yz[:, 0:90] = K_m_yz_part
        K_m_yz[:, 90:180] = -K_m_yz_part[:, ::-1]
        K_m_yz[:, 180:270] = -K_m_yz_part
        K_m_yz[:, 270:360] = K_m_yz_part[:, ::-1]


    rl = np.linspace(0, r_cut, r_cut + 1)
    dtheta = 2 * np.pi / 360
    thetal = np.linspace(0., 2. * np.pi - dtheta, 360)

    # theta_bounds = np.linspace(2. * np.pi - dtheta / 2, 4. * np.pi - dtheta / 2, theta_res + 1)
    theta_bounds = np.linspace(0., 2. * np.pi, 360 + 1)
    r, theta = np.meshgrid(rl, theta_bounds)


    # dipole-dipole interaction
    Fxl = np.zeros((r_cut + 1, 360))
    Fyl = np.zeros((r_cut + 1, 360))
    Fzl = np.zeros((r_cut + 1, 360))
    Fxyl = np.zeros((r_cut + 1, 360))

    for i, r_set in enumerate(rl):
        for j, theta_set in enumerate(thetal):

            lattice_vector = np.array([np.cos(theta_set), np.sin(theta_set), 0.])

            if r_set == 0.:
                
                if q_set == 0.:
                    continue

                else:
                    # single-ion anisotropy
                    A = np.load(prefix + "sia_couplings_z_0=" + str(z_0_set).replace(".", "-") + "_q=" + str(q_set).replace(".", "-") + ".npy")
                    
                    Fxl[0, :] = np.full(360, A[0])
                    Fyl[0, :] = np.full(360, A[1])
                    Fzl[0, :] = np.full(360, A[2])


            else:
                mat = dpl.dipole_field_matrix(r_set * lattice_vector)
                Fxl[i, j] = mat[0, 0]
                Fyl[i, j] = mat[1, 1]
                Fzl[i, j] = mat[2, 2]
                Fxyl[i, j] = mat[0, 1]

    C_p_xx = K_p_xx[0:r_cut, :] + Fxl[0:r_cut, :]
    C_p_yy = K_p_yy[0:r_cut, :] + Fyl[0:r_cut, :]
    C_p_zz = K_p_zz[0:r_cut, :] + Fzl[0:r_cut, :]
    C_p_xy = K_p_xy[0:r_cut, :] + Fxyl[0:r_cut, :]

    p = 2
    cmap = "Spectral_r"

    plt.subplot(3, 3, 1, projection='polar', frame_on=True)
    plt.pcolormesh(theta, r, C_p_xx[0:r_cut, :].T + K_m_xx[0:r_cut, :].T, cmap=cmap, norm=colors.SymLogNorm(vmin=-(10 ** p), vmax=10 ** p, linthresh=thresh, linscale=1.5), zorder=2)

    plt.grid(False)
    plt.xticks([], [])
    plt.yticks([], [])

    plt.title(r"$K_{xx}$", fontsize=fontsize, loc="left")


    plt.subplot(3, 3, 2, projection='polar')
    plt.pcolormesh(theta, r, C_p_xy[0:r_cut, :].T + K_m_xy[0:r_cut, :].T, cmap=cmap, norm=colors.SymLogNorm(vmin=-(10 ** p), vmax=10 ** p, linthresh=thresh, linscale=1.5), zorder=2)

    plt.grid(False)
    plt.xticks([], [])
    plt.yticks([], [])

    plt.title(r"$K_{xy}$", fontsize=fontsize, loc="left")


    plt.subplot(3, 3, 3, projection='polar')
    plt.pcolormesh(theta, r, K_p_xz[0:r_cut, :].T + K_m_xz[0:r_cut, :].T, cmap=cmap, norm=colors.SymLogNorm(vmin=-(10 ** p), vmax=10 ** p, linthresh=thresh, linscale=1.5), zorder=2)

    plt.grid(False)
    plt.xticks([], [])
    plt.yticks([], [])

    plt.title(r"$K_{xz}$", fontsize=fontsize, loc="left")


    plt.subplot(3, 3, 4, projection='polar')
    plt.pcolormesh(theta, r, C_p_xy[0:r_cut, :].T + K_m_xy[0:r_cut, :].T, cmap=cmap, norm=colors.SymLogNorm(vmin=-(10 ** p), vmax=10 ** p, linthresh=thresh, linscale=1.5), zorder=2)

    plt.grid(False)
    plt.xticks([], [])
    plt.yticks([], [])

    plt.title(r"$K_{yx}$", fontsize=fontsize, loc="left")


    plt.subplot(3, 3, 5, projection='polar')
    plt.pcolormesh(theta, r, C_p_yy[0:r_cut, :].T + K_m_yy[0:r_cut, :].T, cmap=cmap, norm=colors.SymLogNorm(vmin=-(10 ** p), vmax=10 ** p, linthresh=thresh, linscale=1.5), zorder=2)

    plt.grid(False)
    plt.xticks([], [])
    plt.yticks([], [])

    plt.title(r"$K_{yy}$", fontsize=fontsize, loc="left")


    plt.subplot(3, 3, 6, projection='polar')
    plot = plt.pcolormesh(theta, r, K_p_yz[0:r_cut, :].T+ K_m_yz[0:r_cut, :].T, cmap=cmap, norm=colors.SymLogNorm(vmin=-(10 ** p), vmax=10 ** p, linthresh=thresh, linscale=1.5), zorder=2)

    plt.grid(False)
    plt.xticks([], [])
    plt.yticks([], [])

    plt.title(r"$K_{yz}$", fontsize=fontsize, loc="left")


    plt.subplot(3, 3, 7, projection='polar', frame_on=True)
    plt.pcolormesh(theta, r, -K_p_xz[0:r_cut, :].T - K_m_xz[0:r_cut, :].T, cmap=cmap, norm=colors.SymLogNorm(vmin=-(10 ** p), vmax=10 ** p, linthresh=thresh, linscale=1.5), zorder=2)

    plt.grid(False)
    plt.xticks([], [])
    plt.yticks([], [])

    plt.title(r"$K_{zx}$", fontsize=fontsize, loc="left")


    plt.subplot(3, 3, 8, projection='polar')
    plt.pcolormesh(theta, r, -K_p_yz[0:r_cut, :].T - K_m_yz[0:r_cut, :].T, cmap=cmap, norm=colors.SymLogNorm(vmin=-(10 ** p), vmax=10 ** p, linthresh=thresh, linscale=1.5), zorder=2)

    plt.grid(False)
    plt.xticks([], [])
    plt.yticks([], [])

    plt.title(r"$K_{zy}$", fontsize=fontsize, loc="left")


    plt.subplot(3, 3, 9, projection='polar')
    plt.pcolormesh(theta, r, C_p_zz[0:r_cut, :].T + K_m_zz[0:r_cut, :].T, cmap=cmap, norm=colors.SymLogNorm(vmin=-(10 ** p), vmax=10 ** p, linthresh=thresh, linscale=1.5), zorder=2)
    
    plt.grid(False)
    plt.xticks([], [])
    plt.yticks([], [])


    plt.title(r"$K_{zz}$", fontsize=fontsize, loc="left")

    # colorbar axis
    ax = fig.add_axes([0.025, 0.05, 0.95, 0.025])
    cb = fig.colorbar(plot, cax=ax, orientation="horizontal")
    cb.ax.tick_params(labelsize=labelsize)

    plt.subplots_adjust(wspace=0.05, hspace=0.25, left=0.025, right=1.0, bottom=0.1, top=0.95)
    plt.savefig("Fig2b.png", format="png", dpi=1200)
    # plt.show()