# dynamics for long times and multiple qa values
# no single-ion anisotropy, damping can be set

# path to repo directory
dirname = "E:/Dropbox/dipoles_private/paper_new/repo"

import numpy as np
import scipy.integrate as spint
import dipole_functions as dpl
import numba as nb

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

    return a


# parameters
L = 2                                   # number of dipoles
z_0_set = 0.1                           # plate distance
theta = 45                              # angle of dipole relative to x-axis
eta = 0.0                               # damping constant
r_max = 20                              # maximal radius
r_res = 21                              # radial res
theta_res = 90                          # angular res
gc_order = 10000                        # gauss chebyshev order
t_final = 100.0                         # simulation time

# qa factors
ql = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0, 900.0, 1000.0]


prefix = dirname + "/data/"

# various initial conditions
for factor1 in [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75]:
    for factor2 in [0.0, 0.25, 0.5, 0.75, 1.0]:

        ear = np.zeros(len(ql))

        for l, q_set in enumerate(ql):

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


            # solve eom

            # initial values for angles and angular velocities 
            # velocities are set to zero
            y0 = np.array([factor1 * np.pi, factor2 * np.pi, 0., 0.])

            # solve ode 
            sol = spint.solve_ivp(deriv, args=(K_xx, K_yy, K_xy), t_span=[0., t_final], t_eval=np.linspace(0., t_final, 2), y0=y0, atol=1.e-12, rtol=1.e-9)

            omega1 = sol.y[2, :]
            omega2 = sol.y[3, :]

            ear[l] = 0.5 * (omega1[-1] ** 2 + omega2[-1] ** 2) / (L * t_final)

        # save results
        np.savez_compressed(prefix + "nonreciprocal_dynamics_pair_energy_vary_q_z0=" + str(z_0_set).replace(".", "-") + "_theta=" + str(theta) + "_eta=" + str(eta) 
        + "_tf=" + str(t_final).replace(".", "-") + "_f1=" + str(round(factor1, 2)) + "_f2=" + str(round(factor2, 2)) + ".npz", en=ear, ql=ql)    