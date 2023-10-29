# compute the dynamics of a pair (or chain) from an initial condition
# with single-ion anisotropy, damping can be set

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
def deriv(t, y, K_xx, K_yy, K_xy):

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

            # single-ion anisotropy
            else:
                a[L + n] += (K_yy[r_nm, 0] - K_xx[r_nm, 0]) * np.sin(2 * y[n])
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
factor1 = 0.5                           # initial condition for phi1 - phi1(0) = f1 * pi
factor2 = 1.75                          # initial condition for phi2 - phi2(0) = f2 * pi
t_final = 100.0                         # simulation time
t_steps = 10001                         # timesteps for analysis (the ode solver will choose the stepsize for computations internally)

# nodes and weight for computation of the single-ion anisotropy
gc_nw = dpl.gauss_chebyshev_nodes_and_weights(gc_order)
gl_nw = dpl.gauss_laguerre_nodes_and_weights(180)

prefix = dirname + "/data/"

# if qa = 0 this is just the dipole dipole matrix
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

# load for finite qa
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

if q_set > 0.0:

    # add SIA
    A_x, A_y, A_z = dpl.sia_couplings(q_set, z_0_set, gc_nw, gl_nw)

    K_xx[0, 0] = K_xx[0, 1] = A_x
    K_yy[0, 0] = K_yy[0, 1] = A_y


# solve eom

# initial values for angles and angular velocities 
# velocities are set to zero
y0 = np.array([factor1 * np.pi, factor2 * np.pi, 0., 0.])

# solve ode 
sol = spint.solve_ivp(deriv, t_span=[0., t_final], t_eval=np.linspace(0., t_final, t_steps), y0=y0, atol=1.e-12, rtol=1.e-9)

# save results
# not all the saved data in /data saves t_final and t_steps in the filename - in that case one can check what was used by looking at the array for t
# in the data filenames t_final is abbreviated tf or T

# save results
np.savez_compressed(prefix + "nonreciprocal_dynamics_pair_q=" + str(q_set).replace(".", "-") + "/nonreciprocal_dynamics_with_sia_pair_q=" + str(q_set).replace(".", "-") + "_z0=" + str(z_0_set).replace(".", "-") + "_theta=" + str(theta) 
+ "_eta=" + str(eta).replace(".", "-") + "_tf=" + str(t_final).replace(".", "-") + "_t_steps=" + str(t_steps) + "_f1=" + str(round(factor1, 2)).replace(".", "-") + "_f2=" + str(round(factor2, 2)).replace(".", "-") + ".npz", t=sol.t, y=sol.y, K_xx=K_xx, K_yy=K_yy, K_xy=K_xy)   