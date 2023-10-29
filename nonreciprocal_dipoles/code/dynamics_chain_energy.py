# compute the dynamics of a pair (or chain) for various initial conditions
# only the values of the energy at the end of the evolution are needed
# no single-ion anisotropy, damping can be set

# path to repo directory
dirname = "E:/Dropbox/dipoles_private/paper_new/repo"

import numpy as np
import numba as nb
import scipy.integrate as spint
import dipole_functions as dpl

# eom functions
# formulated as first order ode for angles phi and angular velocities omega instead of second order ode for phi
# time derivative of (phi, omega)
@nb.jit(nopython=True)
def deriv(t, y):

    a = np.zeros(2 * L)

    # derivative of phi_n is omega_n
    a[0:L] = y[L:2 * L]

    # derivative of omega_n
    for n in range(L):

        # dissipation
        a[L + n] = - eta * y[L + n]

        for m in range(L):

            r_nm = abs(n - m)

            # depending on whether m > n or m < n one needs to choose the appropriate column in the couplings
            if m < n:
                a[L + n] += K_xy[r_nm, 0] * np.cos(y[n] + y[m]) + K_yy[r_nm, 0] * np.sin(y[m]) * np.cos(y[n]) - K_xx[r_nm, 0] * np.sin(y[n]) * np.cos(y[m]) 

            elif m > n:
                a[L + n] += K_xy[r_nm, 1] * np.cos(y[n] + y[m]) + K_yy[r_nm, 1] * np.sin(y[m]) * np.cos(y[n]) - K_xx[r_nm, 1] * np.sin(y[n]) * np.cos(y[m])

    return a


# parameters
L = 2                                   # number of dipoles
q_set = 0.1                             # qa factor
z_0_set = 0.1                           # plate distance
theta = 45                              # angle of dipole relative to x-axis
eta = 0.0                               # damping constant
r_max = 20                              # maximal radius
r_res = 21                              # radial res
r_cut = 10                              # plot up to r_cut
theta_res = 90                          # angular res
gc_order = 10000                        # gauss chebyshev order
factor_steps = 401                      # res for initial conditions 
t_final = 100.0                         # simulation time

prefix = dirname + "/data/"

# coupling matrix

# if qa = 0 this is just the dipole dipole matrix
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


# load for finite qa
else:
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

    # add dipole-dipole interaction
    rl = np.linspace(0, r_cut, r_cut + 1)
    dtheta = 2 * np.pi / 360
    thetal = np.linspace(0., 2. * np.pi - dtheta, 360)

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

                continue

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


# solve eom

# initial values for angles and angular velocities 
# initial condition for phi_n - phi_n(0) = f_n * pi
# velocities are set to zero
# phi_2 only needs half of the values due to symmetry (see the Fig python scripts for description)

factors1 = np.linspace(0.0, 2.0, factor_steps)
factors2 = np.linspace(0.0, 1.0, (factor_steps - 1) // 2 + 1)

# energy absorption rate at the end of the protocol - from this the average angular acceleration can be computed
ear = np.zeros((factor_steps, factor_steps))


for i, factor1 in enumerate(factors1):
        for j, factor2 in enumerate(factors2):

            y0 = np.array([factor1 * np.pi, factor2 * np.pi, 0., 0.])
            
            # solve ode 
            sol = spint.solve_ivp(deriv, args=(K_xx, K_yy, K_xy), t_span=[0., t_final], t_eval=np.linspace(0., t_final, 2), y0=y0, atol=1.e-12, rtol=1.e-9)

            # angular velocities
            omega1 = sol.y[2, :]
            omega2 = sol.y[3, :]

            # energy absorption rate (density)
            ear[i, j] = 0.5 * (omega1[-1] ** 2 + omega2[-1] ** 2) / (L * t_final)


# save results
np.savez_compressed(prefix + "nonreciprocal_dynamics_pair_q=" + str(q_set).replace(".", "-") + "/nonreciprocal_dynamics_pair_energy_q=" + str(q_set).replace(".", "-") + "_z0=" + str(z_0_set).replace(".", "-") + "_theta=" + str(theta) 
+ "_eta=" + str(eta).replace(".", "-") + "_tf=" + str(t_final).replace(".", "-") + "_f_steps=" + str(factor_steps) + ".npz", en=ear, K_xx=K_xx, K_yy=K_yy, K_xy=K_xy)     