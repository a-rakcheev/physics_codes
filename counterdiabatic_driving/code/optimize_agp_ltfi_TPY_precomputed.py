import numpy as np

# parameters
L = 12                                             # number of spins
res_h = 50                                        # number of grid points on x axis
res_g = 25                                        # number of grid points on y axis
l = 7                                             # range cutoff for variational strings

hl = np.linspace(1.e-6, 1.5, res_h)
gl = np.linspace(1.e-6, 0.75, res_g)

name = "optimization_matrices_parallel_ltfi_L=" + str(L) + "_l=" + str(l) + ".npz"
# name = "optimization_matrices_ltfi_L=" + str(L) + "_l=" + str(l) + ".npz"

data = np.load(name)

R_X_ZZ = data["R_X_ZZ"]
R_X_Z = data["R_X_Z"]
R_X_X = data["R_X_X"]
R_Z_X = data["R_Z_X"]
R_Z_Z = data["R_Z_Z"]
R_Z_ZZ = data["R_Z_ZZ"]

P_X_X = data["P_X_X"]
P_X_ZZ = data["P_X_ZZ"]
P_Z_ZZ = data["P_Z_ZZ"]
P_Z_X = data["P_Z_X"]
P_Z_Z = data["P_Z_Z"]
P_ZZ_ZZ = data["P_ZZ_ZZ"]

data = None
var_order = len(R_Z_Z)

# find optimal coefficients
coefficients_h = np.zeros((res_h, res_g, var_order))
coefficients_g = np.zeros((res_h, res_g, var_order))

for i, h in enumerate(hl):
    for j, g in enumerate(gl):

        print("h, g:", h, g)
        P = P_ZZ_ZZ + g * g * P_X_X + h * h * P_Z_Z \
            - g * (P_X_ZZ + P_X_ZZ.T) - h * (P_Z_ZZ + P_Z_ZZ.T) + h * g * (P_Z_X + P_Z_X.T)

        P_inv = np.linalg.inv(P)
        P = None

        R_h = -R_Z_ZZ + g * R_Z_X + h * R_Z_Z
        R_g = -R_X_ZZ + g * R_X_X + h * R_X_Z

        coefficients_h[i, j, :] = 0.5 * np.dot(P_inv, R_h)
        coefficients_g[i, j, :] = 0.5 * np.dot(P_inv, R_g)

name = "optimize_agp_TPY_ltfi_precomputed_L" + str(L) + "_l" + str(l) + "_res_h" + str(
    res_h) + "_res_g" + str(res_g) + ".npz"
np.savez_compressed(name, hl=hl, gl=gl, ch=coefficients_h, cg=coefficients_g)

