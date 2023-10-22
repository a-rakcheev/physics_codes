import numpy as np

# parameters
L = 12                                             # number of spins
res_kappa = 100                                        # number of grid points on x axis
res_g = 100                                        # number of grid points on y axis
l = 7                                             # range cutoff for variational strings

kappal = np.linspace(1.e-2, 1.5, res_kappa)
gl = np.linspace(1.e-2, 1.5, res_g)

name = "optimization_matrices_annni_L=" + str(L) + "_l=" + str(l) + ".npz"
data = np.load(name)

R_X_ZZ = data["R_X_ZZ"]
R_X_Z1Z = data["R_X_Z1Z"]
R_X_X = data["R_X_X"]
R_Z1Z_X = data["R_Z1Z_X"]
R_Z1Z_Z1Z = data["R_Z1Z_Z1Z"]
R_Z1Z_ZZ = data["R_Z1Z_ZZ"]

P_X_X = data["P_X_X"]
P_X_ZZ = data["P_X_ZZ"]
P_Z1Z_ZZ = data["P_Z1Z_ZZ"]
P_Z1Z_X = data["P_Z1Z_X"]
P_Z1Z_Z1Z = data["P_Z1Z_Z1Z"]
P_ZZ_ZZ = data["P_ZZ_ZZ"]

data = None
var_order = len(R_Z1Z_Z1Z)

# find optimal coefficients
coefficients_kappa = np.zeros((res_kappa, res_g, var_order))
coefficients_g = np.zeros((res_kappa, res_g, var_order))

for i, kappa in enumerate(kappal):
    for j, g in enumerate(gl):

        print("kappa, g:", kappa, g)
        P = P_ZZ_ZZ + g * g * P_X_X + kappa * kappa * P_Z1Z_Z1Z \
            + g * (P_X_ZZ + P_X_ZZ.T) - kappa * (P_Z1Z_ZZ + P_Z1Z_ZZ.T) - kappa * g * (P_Z1Z_X + P_Z1Z_X.T)

        P_inv = np.linalg.inv(P)
        P = None

        R_kappa = -R_Z1Z_ZZ - g * R_Z1Z_X + kappa * R_Z1Z_Z1Z
        R_g = R_X_ZZ + g * R_X_X - kappa * R_X_Z1Z

        coefficients_kappa[i, j, :] = 0.5 * np.dot(P_inv, R_kappa)
        coefficients_g[i, j, :] = 0.5 * np.dot(P_inv, R_g)

name = "optimize_agp_TPFY_annni_precomputed_L" + str(L) + "_l" + str(l) + "_res_kappa" + str(
    res_kappa) + "_res_g" + str(res_g) + ".npz"
np.savez_compressed(name, kappal=kappal, gl=gl, ckappa=coefficients_kappa, cg=coefficients_g)

