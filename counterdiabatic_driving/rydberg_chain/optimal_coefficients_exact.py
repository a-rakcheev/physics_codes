# compute optimal coefficients for the truncated Rydberg chain in the TDL
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as linalg
import tqdm
import sys

# # parameters
# l = int(sys.argv[1])                                             # range cutoff for variational strings
# L = int(sys.argv[2])                                             # number of spins
# res_h = int(sys.argv[3])                                         # number of grid points on x axis
# res_g = int(sys.argv[4])                                         # number of grid points on y axis

# parameters
L = 4
l = L                                            # range cutoff for variational strings
res_g = 9                                        # logarithmic resolution
res_h = 9                                        # logarithmic resolution

gl = np.concatenate((np.linspace(1, 9, res_g) * 10 ** -5, np.linspace(1, 9, res_g) * 10 ** -4, np.linspace(1, 9, res_g) * 10 ** -3, np.linspace(1, 9, res_g) * 10 ** -2, np.linspace(1, 9, res_g) * 10 ** -1, np.linspace(1, 5, (res_g + 1) // 2) * 10 ** 0))
hl = np.concatenate((np.linspace(1, 9, res_h) * 10 ** -5, np.linspace(1, 9, res_h) * 10 ** -4, np.linspace(1, 9, res_h) * 10 ** -3, np.linspace(1, 9, res_h) * 10 ** -2, np.linspace(1, 9, res_h) * 10 ** -1, np.linspace(1, 5, (res_h + 1) // 2) * 10 ** 0))


# read in R and P matrices for the given operators
# R matrices
name = "optimization_matrices_TPY/optimization_matrices_R_exact_X_X_TPY_l=" + str(l) + ".npz"
data = np.load(name)
R_exact_X_X = data["R"]
    
name = "optimization_matrices_TPY/optimization_matrices_R_exact_X_Z_TPY_l=" + str(l) + ".npz"
data = np.load(name)
R_exact_X_Z = data["R"]

name = "optimization_matrices_TPY/optimization_matrices_R_exact_X_ZZ_TPY_l=" + str(l) + ".npz"
data = np.load(name)
R_exact_X_ZZ = data["R"]

name = "optimization_matrices_TPY/optimization_matrices_R_exact_X_Z1Z_TPY_l=" + str(l) + ".npz"
data = np.load(name)
R_exact_X_Z1Z = data["R"]

name = "optimization_matrices_TPY/optimization_matrices_R_exact_X_Z11Z_TPY_l=" + str(l) + ".npz"
data = np.load(name)
R_exact_X_Z11Z = data["R"]

name = "optimization_matrices_TPY/optimization_matrices_R_exact_Z_X_TPY_l=" + str(l) + ".npz"
data = np.load(name)
R_exact_Z_X = data["R"]
    
name = "optimization_matrices_TPY/optimization_matrices_R_exact_Z_Z_TPY_l=" + str(l) + ".npz"
data = np.load(name)
R_exact_Z_Z = data["R"]

name = "optimization_matrices_TPY/optimization_matrices_R_exact_Z_ZZ_TPY_l=" + str(l) + ".npz"
data = np.load(name)
R_exact_Z_ZZ = data["R"]

name = "optimization_matrices_TPY/optimization_matrices_R_exact_Z_Z1Z_TPY_l=" + str(l) + ".npz"
data = np.load(name)
R_exact_Z_Z1Z = data["R"]

name = "optimization_matrices_TPY/optimization_matrices_R_exact_Z_Z11Z_TPY_l=" + str(l) + ".npz"
data = np.load(name)
R_exact_Z_Z11Z = data["R"]

# combine ZdZ Hamiltonian
R_exact_X_ZdZ = R_exact_X_ZZ + (1 / 2 ** 6) * R_exact_X_Z1Z + (1 / 3 ** 6) * R_exact_X_Z11Z
R_exact_Z_ZdZ = R_exact_Z_ZZ + (1 / 2 ** 6) * R_exact_Z_Z1Z + (1 / 3 ** 6) * R_exact_Z_Z11Z


# P matrices
# from X
name = "optimization_matrices_TPY/optimization_matrices_P_exact_X_X_TPY_l=" + str(l) + ".npz"
P_exact_X_X = sp.load_npz(name)
name = "optimization_matrices_TPY/optimization_matrices_P_exact_diag_X_X_TPY_l=" + str(l) + ".npz"
P_exact_X_X_diag = sp.load_npz(name)
P_exact_X_X = P_exact_X_X + P_exact_X_X.T + P_exact_X_X_diag

name = "optimization_matrices_TPY/optimization_matrices_P_exact_X_Z_TPY_l=" + str(l) + ".npz"
P_exact_X_Z = sp.load_npz(name)

name = "optimization_matrices_TPY/optimization_matrices_P_exact_X_ZZ_TPY_l=" + str(l) + ".npz"
P_exact_X_ZZ = sp.load_npz(name)

name = "optimization_matrices_TPY/optimization_matrices_P_exact_X_Z1Z_TPY_l=" + str(l) + ".npz"
P_exact_X_Z1Z = sp.load_npz(name)

name = "optimization_matrices_TPY/optimization_matrices_P_exact_X_Z11Z_TPY_l=" + str(l) + ".npz"
P_exact_X_Z11Z = sp.load_npz(name)


# from Z
name = "optimization_matrices_TPY/optimization_matrices_P_exact_Z_Z_TPY_l=" + str(l) + ".npz"
P_exact_Z_Z = sp.load_npz(name)
name = "optimization_matrices_TPY/optimization_matrices_P_exact_diag_Z_Z_TPY_l=" + str(l) + ".npz"
P_exact_Z_Z_diag = sp.load_npz(name)
P_exact_Z_Z = P_exact_Z_Z + P_exact_Z_Z.T + P_exact_Z_Z_diag

name = "optimization_matrices_TPY/optimization_matrices_P_exact_Z_ZZ_TPY_l=" + str(l) + ".npz"
P_exact_Z_ZZ = sp.load_npz(name)

name = "optimization_matrices_TPY/optimization_matrices_P_exact_Z_Z1Z_TPY_l=" + str(l) + ".npz"
P_exact_Z_Z1Z = sp.load_npz(name)

name = "optimization_matrices_TPY/optimization_matrices_P_exact_Z_Z11Z_TPY_l=" + str(l) + ".npz"
P_exact_Z_Z11Z = sp.load_npz(name)

# from ZZ
name = "optimization_matrices_TPY/optimization_matrices_P_exact_ZZ_ZZ_TPY_l=" + str(l) + ".npz"
P_exact_ZZ_ZZ = sp.load_npz(name)
name = "optimization_matrices_TPY/optimization_matrices_P_exact_diag_ZZ_ZZ_TPY_l=" + str(l) + ".npz"
P_exact_ZZ_ZZ_diag = sp.load_npz(name)
P_exact_ZZ_ZZ = P_exact_ZZ_ZZ + P_exact_ZZ_ZZ.T + P_exact_ZZ_ZZ_diag

name = "optimization_matrices_TPY/optimization_matrices_P_exact_ZZ_Z1Z_TPY_l=" + str(l) + ".npz"
P_exact_ZZ_Z1Z = sp.load_npz(name)

name = "optimization_matrices_TPY/optimization_matrices_P_exact_ZZ_Z11Z_TPY_l=" + str(l) + ".npz"
P_exact_ZZ_Z11Z = sp.load_npz(name)


# from Z1Z
name = "optimization_matrices_TPY/optimization_matrices_P_exact_Z1Z_Z1Z_TPY_l=" + str(l) + ".npz"
P_exact_Z1Z_Z1Z = sp.load_npz(name)
name = "optimization_matrices_TPY/optimization_matrices_P_exact_diag_Z1Z_Z1Z_TPY_l=" + str(l) + ".npz"
P_exact_Z1Z_Z1Z_diag = sp.load_npz(name)
P_exact_Z1Z_Z1Z = P_exact_Z1Z_Z1Z + P_exact_Z1Z_Z1Z.T + P_exact_Z1Z_Z1Z_diag

name = "optimization_matrices_TPY/optimization_matrices_P_exact_Z1Z_Z11Z_TPY_l=" + str(l) + ".npz"
P_exact_Z1Z_Z11Z = sp.load_npz(name)


# from Z11Z
name = "optimization_matrices_TPY/optimization_matrices_P_exact_Z11Z_Z11Z_TPY_l=" + str(l) + ".npz"
P_exact_Z11Z_Z11Z = sp.load_npz(name)
name = "optimization_matrices_TPY/optimization_matrices_P_exact_diag_Z11Z_Z11Z_TPY_l=" + str(l) + ".npz"
P_exact_Z11Z_Z11Z_diag = sp.load_npz(name)
P_exact_Z11Z_Z11Z = P_exact_Z11Z_Z11Z + P_exact_Z11Z_Z11Z.T + P_exact_Z11Z_Z11Z_diag


# combine to ZdZ Hamiltonian
P_exact_X_ZdZ = P_exact_X_ZZ + (1 / 2 ** 6) * P_exact_X_Z1Z + (1 / 3 ** 6) * P_exact_X_Z11Z
P_exact_Z_ZdZ = P_exact_Z_ZZ + (1 / 2 ** 6) * P_exact_Z_Z1Z + (1 / 3 ** 6) * P_exact_Z_Z11Z

P_exact_ZdZ_ZdZ = P_exact_ZZ_ZZ + (1 / 2 ** 6) * (P_exact_ZZ_Z1Z + P_exact_ZZ_Z1Z.T) + (1 / 3 ** 6) * (P_exact_ZZ_Z11Z + P_exact_ZZ_Z11Z.T) \
    + (1 / 2 ** 6) ** 2 * P_exact_Z1Z_Z1Z + (1 / 2 ** 6) * (1 / 3 ** 6) * (P_exact_Z1Z_Z11Z + P_exact_Z1Z_Z11Z.T) + (1 / 3 ** 6) ** 2 * P_exact_Z11Z_Z11Z


# find optimal coefficients
var_order = len(R_exact_X_X)
coefficients_1 = np.zeros((len(gl), len(hl), var_order))
coefficients_2 = np.zeros((len(gl), len(hl), var_order))

for i, g in enumerate(tqdm.tqdm(gl)):
    for j, h in enumerate(hl):
        
        P_exact = P_exact_ZdZ_ZdZ + (g ** 2) * P_exact_X_X + (h ** 2) * P_exact_Z_Z + (-1) * g * (P_exact_X_ZdZ + P_exact_X_ZdZ.T) + (-1) * h * (P_exact_Z_ZdZ + P_exact_Z_ZdZ.T) + g * h * (P_exact_X_Z + P_exact_X_Z.T)

        R_exact_X = (-1) * (R_exact_X_ZdZ + (-1) * g * R_exact_X_X + (-1) * h * R_exact_X_Z)
        R_exact_Z = (-1) * (R_exact_Z_ZdZ + (-1) * g * R_exact_Z_X + (-1) * h * R_exact_Z_Z)

        coefficients_1[i, j, :] = linalg.spsolve(P_exact, 0.5 * R_exact_X)
        coefficients_2[i, j, :] = linalg.spsolve(P_exact, 0.5 * R_exact_Z)


name = "optimal_coefficients/optimal_coefficients_exact_rydberg_chain_truncated_TPY_L=" + str(L) + "_res_g=" + str(res_g) + "_res_h=" + str(res_h) + ".npz"
np.savez_compressed(name, gl=gl, hl=hl, c_g=coefficients_1, c_h=coefficients_2)


