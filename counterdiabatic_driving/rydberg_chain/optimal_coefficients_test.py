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
l = 4                                            # range cutoff for variational strings
res_g = 65                                        # logarithmic resolution
res_h = 65                                        # logarithmic resolution

gl = np.concatenate((np.linspace(1, 9, res_g) * 10 ** -5, np.linspace(1, 9, res_g) * 10 ** -4, np.linspace(1, 9, res_g) * 10 ** -3, np.linspace(1, 9, res_g) * 10 ** -2, np.linspace(1, 9, res_g) * 10 ** -1, np.linspace(1, 5, (res_g + 1) // 2) * 10 ** 0))
hl = np.concatenate((np.linspace(1, 9, res_h) * 10 ** -5, np.linspace(1, 9, res_h) * 10 ** -4, np.linspace(1, 9, res_h) * 10 ** -3, np.linspace(1, 9, res_h) * 10 ** -2, np.linspace(1, 9, res_h) * 10 ** -1, np.linspace(1, 5, (res_h + 1) // 2) * 10 ** 0))


# read in R and P matrices for the given operators
# R matrices
name = "optimization_matrices_TPY/optimization_matrices_R_X_X_TPY_l=" + str(l) + ".npz"
data = np.load(name)
R_X_X = data["R"]
    
name = "optimization_matrices_TPY/optimization_matrices_R_X_Z_TPY_l=" + str(l) + ".npz"
data = np.load(name)
R_X_Z = data["R"]

name = "optimization_matrices_TPY/optimization_matrices_R_X_ZZ_TPY_l=" + str(l) + ".npz"
data = np.load(name)
R_X_ZZ = data["R"]

name = "optimization_matrices_TPY/optimization_matrices_R_X_Z1Z_TPY_l=" + str(l) + ".npz"
data = np.load(name)
R_X_Z1Z = data["R"]

name = "optimization_matrices_TPY/optimization_matrices_R_X_Z11Z_TPY_l=" + str(l) + ".npz"
data = np.load(name)
R_X_Z11Z = data["R"]

name = "optimization_matrices_TPY/optimization_matrices_R_Z_X_TPY_l=" + str(l) + ".npz"
data = np.load(name)
R_Z_X = data["R"]
    
name = "optimization_matrices_TPY/optimization_matrices_R_Z_Z_TPY_l=" + str(l) + ".npz"
data = np.load(name)
R_Z_Z = data["R"]

name = "optimization_matrices_TPY/optimization_matrices_R_Z_ZZ_TPY_l=" + str(l) + ".npz"
data = np.load(name)
R_Z_ZZ = data["R"]

name = "optimization_matrices_TPY/optimization_matrices_R_Z_Z1Z_TPY_l=" + str(l) + ".npz"
data = np.load(name)
R_Z_Z1Z = data["R"]

name = "optimization_matrices_TPY/optimization_matrices_R_Z_Z11Z_TPY_l=" + str(l) + ".npz"
data = np.load(name)
R_Z_Z11Z = data["R"]

# combine ZdZ Hamiltonian
R_X_ZdZ = R_X_ZZ + (1 / 2 ** 6) * R_X_Z1Z + (1 / 3 ** 6) * R_X_Z11Z
R_Z_ZdZ = R_Z_ZZ + (1 / 2 ** 6) * R_Z_Z1Z + (1 / 3 ** 6) * R_Z_Z11Z


# P matrices
# from X
name = "optimization_matrices_TPY/optimization_matrices_P_X_X_TPY_l=" + str(l) + ".npz"
P_X_X = sp.load_npz(name)
name = "optimization_matrices_TPY/optimization_matrices_P_diag_X_X_TPY_l=" + str(l) + ".npz"
P_X_X_diag = sp.load_npz(name)
P_X_X = P_X_X + P_X_X.T + P_X_X_diag

name = "optimization_matrices_TPY/optimization_matrices_P_X_Z_TPY_l=" + str(l) + ".npz"
P_X_Z = sp.load_npz(name)

name = "optimization_matrices_TPY/optimization_matrices_P_X_ZZ_TPY_l=" + str(l) + ".npz"
P_X_ZZ = sp.load_npz(name)

name = "optimization_matrices_TPY/optimization_matrices_P_X_Z1Z_TPY_l=" + str(l) + ".npz"
P_X_Z1Z = sp.load_npz(name)

name = "optimization_matrices_TPY/optimization_matrices_P_X_Z11Z_TPY_l=" + str(l) + ".npz"
P_X_Z11Z = sp.load_npz(name)


# from Z
name = "optimization_matrices_TPY/optimization_matrices_P_Z_Z_TPY_l=" + str(l) + ".npz"
P_Z_Z = sp.load_npz(name)
name = "optimization_matrices_TPY/optimization_matrices_P_diag_Z_Z_TPY_l=" + str(l) + ".npz"
P_Z_Z_diag = sp.load_npz(name)
P_Z_Z = P_Z_Z + P_Z_Z.T + P_Z_Z_diag

name = "optimization_matrices_TPY/optimization_matrices_P_Z_ZZ_TPY_l=" + str(l) + ".npz"
P_Z_ZZ = sp.load_npz(name)

name = "optimization_matrices_TPY/optimization_matrices_P_Z_Z1Z_TPY_l=" + str(l) + ".npz"
P_Z_Z1Z = sp.load_npz(name)

name = "optimization_matrices_TPY/optimization_matrices_P_Z_Z11Z_TPY_l=" + str(l) + ".npz"
P_Z_Z11Z = sp.load_npz(name)

# from ZZ
name = "optimization_matrices_TPY/optimization_matrices_P_ZZ_ZZ_TPY_l=" + str(l) + ".npz"
P_ZZ_ZZ = sp.load_npz(name)
name = "optimization_matrices_TPY/optimization_matrices_P_diag_ZZ_ZZ_TPY_l=" + str(l) + ".npz"
P_ZZ_ZZ_diag = sp.load_npz(name)
P_ZZ_ZZ = P_ZZ_ZZ + P_ZZ_ZZ.T + P_ZZ_ZZ_diag

name = "optimization_matrices_TPY/optimization_matrices_P_ZZ_Z1Z_TPY_l=" + str(l) + ".npz"
P_ZZ_Z1Z = sp.load_npz(name)

name = "optimization_matrices_TPY/optimization_matrices_P_ZZ_Z11Z_TPY_l=" + str(l) + ".npz"
P_ZZ_Z11Z = sp.load_npz(name)


# from Z1Z
name = "optimization_matrices_TPY/optimization_matrices_P_Z1Z_Z1Z_TPY_l=" + str(l) + ".npz"
P_Z1Z_Z1Z = sp.load_npz(name)
name = "optimization_matrices_TPY/optimization_matrices_P_diag_Z1Z_Z1Z_TPY_l=" + str(l) + ".npz"
P_Z1Z_Z1Z_diag = sp.load_npz(name)
P_Z1Z_Z1Z = P_Z1Z_Z1Z + P_Z1Z_Z1Z.T + P_Z1Z_Z1Z_diag

name = "optimization_matrices_TPY/optimization_matrices_P_Z1Z_Z11Z_TPY_l=" + str(l) + ".npz"
P_Z1Z_Z11Z = sp.load_npz(name)


# from Z11Z
name = "optimization_matrices_TPY/optimization_matrices_P_Z11Z_Z11Z_TPY_l=" + str(l) + ".npz"
P_Z11Z_Z11Z = sp.load_npz(name)
name = "optimization_matrices_TPY/optimization_matrices_P_diag_Z11Z_Z11Z_TPY_l=" + str(l) + ".npz"
P_Z11Z_Z11Z_diag = sp.load_npz(name)
P_Z11Z_Z11Z = P_Z11Z_Z11Z + P_Z11Z_Z11Z.T + P_Z11Z_Z11Z_diag

# print("R_X_X:", R_X_X)
# print("R_X_Z:", R_X_Z)
# print("R_X_ZZ:", R_X_ZZ)
# print("R_X_Z1Z:", R_X_Z1Z)
# print("R_X_Z11Z:", R_X_Z11Z)

# print("R_Z_X:", R_Z_X)
# print("R_Z_Z:", R_Z_Z)
# print("R_Z_ZZ:", R_Z_ZZ)
# print("R_Z_Z1Z:", R_Z_Z1Z)
# print("R_Z_Z11Z:", R_Z_Z11Z)

# print("P_X_X:", P_X_X)
# print("P_X_Z:", P_X_Z)
# print("P_X_ZZ:", P_X_ZZ)
# print("P_X_Z1Z:", P_X_Z1Z)
# print("P_X_Z11Z:", P_X_Z11Z)

# print("P_Z_Z:", P_Z_Z)
# print("P_Z_ZZ:", P_Z_ZZ)
# print("P_Z_Z1Z:", P_Z_Z1Z)
# print("P_Z_Z11Z:", P_Z_Z11Z)

# print("P_ZZ_ZZ:", P_ZZ_ZZ)
# print("P_ZZ_Z1Z:", P_ZZ_Z1Z)
# print("P_ZZ_Z11Z:", P_ZZ_Z11Z)

# print("P_Z1Z_Z1Z:", P_Z1Z_Z1Z)
# print("P_Z1Z_Z11Z:", P_Z1Z_Z11Z)

# print("P_Z11Z_Z11Z:", P_Z11Z_Z11Z)

# combine to ZdZ Hamiltonian
P_X_ZdZ = P_X_ZZ + (1 / 2 ** 6) * P_X_Z1Z + (1 / 3 ** 6) * P_X_Z11Z
P_Z_ZdZ = P_Z_ZZ + (1 / 2 ** 6) * P_Z_Z1Z + (1 / 3 ** 6) * P_Z_Z11Z

P_ZdZ_ZdZ = P_ZZ_ZZ + (1 / 2 ** 6) * (P_ZZ_Z1Z + P_ZZ_Z1Z.T) + (1 / 3 ** 6) * (P_ZZ_Z11Z + P_ZZ_Z11Z.T) \
    + (1 / 2 ** 6) ** 2 * P_Z1Z_Z1Z + (1 / 2 ** 6) * (1 / 3 ** 6) * (P_Z1Z_Z11Z + P_Z1Z_Z11Z.T) + (1 / 3 ** 6) ** 2 * P_Z11Z_Z11Z

def alpha(a, g, h, p):
    factor = h ** 2 + g ** 2 + 2 * (1 + (1 / 4 ** p) + (1 / 9 ** p))
    return 0.5 * a / factor

def P_analytic(g, h, p):
    return 4 * h ** 2 + 4 * g ** 2 + 8 * (1 + (1 / 4 ** p) + (1 / 9 ** p))

# find optimal coefficients
var_order = len(R_X_X)
coefficients_1 = np.zeros((len(gl), len(hl), var_order))
coefficients_2 = np.zeros((len(gl), len(hl), var_order))

coefficients_1_analytic = np.zeros((len(gl), len(hl), var_order))
coefficients_2_analytic = np.zeros((len(gl), len(hl), var_order))

for i, g in enumerate(tqdm.tqdm(gl)):
    for j, h in enumerate(hl):
        
        P = P_ZdZ_ZdZ + (g ** 2) * P_X_X + (h ** 2) * P_Z_Z + (-1) * g * (P_X_ZdZ + P_X_ZdZ.T) + (-1) * h * (P_Z_ZdZ + P_Z_ZdZ.T) + g * h * (P_X_Z + P_X_Z.T)
        # P_a = P_analytic(g, h, 6)

        # print(P, P_a, abs(P_a - P[0, 0]))
        R_X = (-1) * (R_X_ZdZ + (-1) * g * R_X_X + (-1) * h * R_X_Z)
        R_Z = (-1) * (R_Z_ZdZ + (-1) * g * R_Z_X + (-1) * h * R_Z_Z)

        coefficients_1[i, j, :] = linalg.spsolve(P, 0.5 * R_X)
        coefficients_2[i, j, :] = linalg.spsolve(P, 0.5 * R_Z)

        coefficients_1_analytic[i, j, :] = alpha(h, g, h, 6)
        coefficients_2_analytic[i, j, :] = alpha(-g, g, h, 6)


print(np.linalg.norm(coefficients_1 - coefficients_1_analytic))
print(np.linalg.norm(coefficients_2 - coefficients_2_analytic))

name = "optimal_coefficients/optimal_coefficients_rydberg_chain_truncated_TPY_l" + str(l) + "_res_g=" + str(res_g) + "_res_h=" + str(res_h) + ".npz"
np.savez_compressed(name, gl=gl, hl=hl, c_g=coefficients_1, c_h=coefficients_2)