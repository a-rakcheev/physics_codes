import sys
import numpy as np
import scipy.integrate as spint
import dipole_functions as dpl
import numba as nb

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

            # SIA
            else:
                a[L + n] += (K_yy[r_nm, 0] - K_xx[r_nm, 0]) * np.sin(2 * y[n])
    return a


# parameters
L = int(sys.argv[1])
theta = int(sys.argv[2])
q_set = float(sys.argv[3])
eta = float(sys.argv[4])
t_final = float(sys.argv[5])
factor_steps = int(sys.argv[6])
t_steps = int(sys.argv[7])

# L = 2
# theta = 30
# q_set = 100.0
# eta = 0.0
# t_final = 100.0
# t_steps = 5000
# factor_steps = 41

z_0_set = 0.1  # plate distance
r_max = 20  # maximal radius
r_res = 21  # radial
gc_order = 10000

gc_nw = dpl.gauss_chebyshev_nodes_and_weights(gc_order)
gl_nw = dpl.gauss_laguerre_nodes_and_weights(180)

# prefix = "E:/Dropbox/data/dipoles/"
# prefix = "/data/Dropbox/data/dipoles/"
prefix = ""


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

if q_set > 0.0:

    # add SIA
    A_x, A_y, A_z = dpl.sia_couplings(q_set, z_0_set, gc_nw, gl_nw)

    K_xx[0, 0] = K_xx[0, 1] = A_x
    K_yy[0, 0] = K_yy[0, 1] = A_y


# initial values
factors1 = np.linspace(0.0, 2.0, factor_steps)
factors2 = np.linspace(0.0, 1.0, (factor_steps - 1) // 2 + 1)
ear = np.zeros((factor_steps, factor_steps))


for i, factor1 in enumerate(factors1):
        print(i, flush=True)
        for j, factor2 in enumerate(factors2):

            y0 = np.array([factor1 * np.pi, factor2 * np.pi, 0., 0.])
            
            # solve ode 
            sol = spint.solve_ivp(deriv, args=(K_xx, K_yy, K_xy), t_span=[0., t_final], t_eval=np.linspace(0., t_final, t_steps), y0=y0, atol=1.e-12, rtol=1.e-9)

            omega1 = sol.y[2, :]
            omega2 = sol.y[3, :]

            ear[i, j] = np.mean(np.diff(0.5 * (omega1 ** 2 + omega2 ** 2)) / (L * (t_final / t_steps)))



# save results
np.savez_compressed(prefix + "nonreciprocal_dynamics_with_sia_hamiltonian_pair_energy_q=" + str(q_set).replace(".", "-") + "_z0=" + str(z_0_set).replace(".", "-") + "_theta=" + str(theta) 
+ "_eta=" + str(eta) + "_tf=" + str(t_final).replace(".", "-") + "_t_steps=" + str(t_steps) + "_f_steps=" + str(factor_steps) + ".npz", en=ear, K_xx=K_xx, K_yy=K_yy, K_xy=K_xy)    



