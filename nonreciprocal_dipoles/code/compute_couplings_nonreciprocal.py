# compute couplings as a function of r, theta
# these computations were performed on a cluster

import numpy as np
import numba as nb
import dipole_functions as dpl


q_set = 1.0                             # qa factor
z_0_set = 0.1                           # plate distance
R = 10                                  # maximal radius
radial_res = 11                         # radial resolution
angular_res = 360                       # angular resolution - can be reduced by using symmetries as discussed in the Appendix
gc_order = 10000                        # gauss chebyshev order

# factors for adaptive integration - see Appendix for details
freq_res = 100
bc_factor = 10

# gauss chebyshev nodes and weights
gc_nw = dpl.gauss_chebyshev_nodes_and_weights(gc_order)

@nb.njit(parallel=True)
def couplings_polar(q, z_0, r_max, r_res, theta_res, nodes_and_weights, frequency_res, boundary_factor):

    K_plus_xx = np.zeros((r_res, theta_res))
    K_plus_yy = np.zeros((r_res, theta_res))
    K_plus_zz = np.zeros((r_res, theta_res))
    K_plus_xy = np.zeros((r_res, theta_res))
    K_plus_xz = np.zeros((r_res, theta_res))
    K_plus_yz = np.zeros((r_res, theta_res))

    K_minus_xx = np.zeros((r_res, theta_res))
    K_minus_yy = np.zeros((r_res, theta_res))
    K_minus_zz = np.zeros((r_res, theta_res))
    K_minus_xy = np.zeros((r_res, theta_res))
    K_minus_xz = np.zeros((r_res, theta_res))
    K_minus_yz = np.zeros((r_res, theta_res))


    rl = np.linspace(0, r_max, r_res)
    dtheta = 2. * np.pi / theta_res
    thetal = np.linspace(0., 2. * np.pi - dtheta, theta_res)

    # prange can parallelize on shared memory (omp) setups
    for i in nb.prange(r_res):

        r_set = rl[i]

        for j in range(theta_res):

            theta_set = thetal[j]
            if r_set == 0.:

                K_plus_xx[i, j] = 0
                K_plus_yy[i, j] = 0
                K_plus_zz[i, j] = 0
                K_plus_xy[i, j] = 0
                K_plus_xz[i, j] = 0
                K_plus_yz[i, j] = 0

                K_minus_xx[i, j] = 0
                K_minus_yy[i, j] = 0
                K_minus_zz[i, j] = 0
                K_minus_xy[i, j] = 0
                K_minus_xz[i, j] = 0
                K_minus_yz[i, j] = 0

            else:

                K_p_xx, K_p_yy, K_p_zz, K_p_xy, K_p_xz, K_p_yz, K_m_xx, K_m_yy, K_m_zz, K_m_xy, K_m_xz, K_m_yz = dpl.adaptive_integrals(q, z_0, r_set, theta_set, frequency_res, boundary_factor, nodes_and_weights[:, 0], nodes_and_weights[:, 1])
                
                
                K_plus_xx[i, j] = K_p_xx
                K_plus_yy[i, j] = K_p_yy
                K_plus_zz[i, j] = K_p_zz
                K_plus_xy[i, j] = K_p_xy
                K_plus_xz[i, j] = K_p_xz
                K_plus_yz[i, j] = K_p_yz

                K_minus_xx[i, j] = K_m_xx
                K_minus_yy[i, j] = K_m_yy
                K_minus_zz[i, j] = K_m_zz
                K_minus_xy[i, j] = K_m_xy
                K_minus_xz[i, j] = K_m_xz
                K_minus_yz[i, j] = K_m_yz

    return K_plus_xx, K_plus_yy, K_plus_zz, K_plus_xy, K_plus_xz, K_plus_yz, K_minus_xx, K_minus_yy, K_minus_zz, K_minus_xy, K_minus_xz, K_minus_yz 


K_plus_xx_int, K_plus_yy_int, K_plus_zz_int, K_plus_xy_int, K_plus_xz_int, K_plus_yz_int, K_minus_xx_int, K_minus_yy_int, K_minus_zz_int, K_minus_xy_int, K_minus_xz_int, K_minus_yz_int = couplings_polar(q_set, z_0_set, R, radial_res, angular_res, gc_nw, freq_res, bc_factor)

name = "couplings_polar_nonreciprocal_q=" + str(q_set).replace(".", "-") + "_z0=" + str(z_0_set).replace(".", "-") + "_rmax=" + str(R) + "_rres=" + str(radial_res) + "_gc_order=" + str(gc_order)
np.savez_compressed(name + ".npz", K_p_xx=K_plus_xx_int, K_p_yy=K_plus_yy_int, K_p_zz=K_plus_zz_int, K_p_xy=K_plus_xy_int, K_p_xz=K_plus_xz_int, K_p_yz=K_plus_yz_int, K_m_xx=K_minus_xx_int, K_m_yy=K_minus_yy_int, K_m_zz=K_minus_zz_int, K_m_xy=K_minus_xy_int, K_m_xz=K_minus_xz_int, K_m_yz=K_minus_yz_int)
