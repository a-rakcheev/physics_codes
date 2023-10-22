import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as linalg
import time
import sys

# # parameters
# l = int(sys.argv[1])                                             # range cutoff for variational strings
# L = int(sys.argv[2])                                             # number of spins
# res_h = int(sys.argv[3])                                         # number of grid points on x axis
# res_g = int(sys.argv[4])                                         # number of grid points on y axis

# parameters
l = 6                                             # range cutoff for variational strings
L = 13
res_kappa = 50                                        # number of grid points on x axis
res_g = 50                                       # number of grid points on y axis


kappal = np.linspace(1.e-6, 1.5, res_kappa)
gl = np.linspace(1.e-6, 0.75, res_g)

# R matrices
name = "optimization_matrices/annni_optimization_matrices_R_L=" + str(L) + "_l=" + str(l) + ".npz"
data = np.load(name)

R_X_ZZ = data["R_X_ZZ"]
R_X_Z1Z = data["R_X_Z1Z"]
R_X_X = data["R_X_X"]
R_Z1Z_X = data["R_Z1Z_X"]
R_Z1Z_ZZ = data["R_Z1Z_ZZ"]
R_Z1Z_Z1Z = data["R_Z1Z_Z1Z"]

var_order = len(R_X_X)

# P matrices

# symmetric
name = "optimization_matrices/optimization_matrices_P_X_X_TPFY_l=" + str(l) + ".npz"
P_X_X = sp.load_npz(name)

name = "optimization_matrices/optimization_matrices_P_Z1Z_Z1Z_TPFY_l=" + str(l) + ".npz"
P_Z1Z_Z1Z = 4 * sp.load_npz(name)

name = "optimization_matrices/optimization_matrices_P_ZZ_ZZ_TPFY_l=" + str(l) + ".npz"
P_ZZ_ZZ = sp.load_npz(name)

# full
name = "optimization_matrices/optimization_matrices_P_X_Z1Z_TPFY_l=" + str(l) + ".npz"
P_X_Z1Z = 2 * sp.load_npz(name)

name = "optimization_matrices/optimization_matrices_P_X_ZZ_TPFY_l=" + str(l) + ".npz"
P_X_ZZ = sp.load_npz(name)

name = "optimization_matrices/optimization_matrices_P_ZZ_Z1Z_TPFY_l=" + str(l) + ".npz"
P_ZZ_Z1Z = 2 * sp.load_npz(name)

# test
L_test = 12

# full
name = "optimization_matrices/optimization_matrices_annni_L=" + str(L_test) + "_l=" + str(l) + ".npz"
data = np.load(name)

R_X_Z1Z_test = data["R_X_Z1Z"] / L_test
R_X_ZZ_test = data["R_X_ZZ"] / L_test
R_X_X_test = data["R_X_X"] / L_test
R_Z1Z_X_test = data["R_Z1Z_X"] / L_test
R_Z1Z_Z1Z_test = data["R_Z1Z_Z1Z"] / L_test
R_Z1Z_ZZ_test = data["R_Z1Z_ZZ"] / L_test

P_X_X_test = data["P_X_X"] / L_test
P_X_ZZ_test = data["P_X_ZZ"] / L_test
P_Z1Z_X_test = data["P_Z1Z_X"] / L_test
P_Z1Z_ZZ_test = data["P_Z1Z_ZZ"] / L_test
P_ZZ_ZZ_test = data["P_ZZ_ZZ"] / L_test
P_Z1Z_Z1Z_test = data["P_Z1Z_Z1Z"] / L_test

print("R Matrices")
print("R_X_X")
print(np.linalg.norm(R_X_X - R_X_X_test))

print("R_X_Z")
print(np.linalg.norm(R_X_Z1Z - R_X_Z1Z_test))

print("R_X_ZZ")
print(np.linalg.norm(R_X_ZZ - R_X_ZZ_test))

print("R_Z_X")
print(np.linalg.norm(R_Z1Z_X - R_Z1Z_X_test))

print("R_Z_Z")
print(np.linalg.norm(R_Z1Z_ZZ - R_Z1Z_ZZ_test))

print("R_Z_ZZ")
print(np.linalg.norm(R_Z1Z_Z1Z - R_Z1Z_Z1Z_test))



print("P Matrices:")
print("P_X_X")
print(np.linalg.norm(P_X_X.todense() - P_X_X_test))
print(P_X_X - sp.coo_matrix(P_X_X_test))

print("P_ZZ_ZZ")
print(np.linalg.norm(P_ZZ_ZZ.todense() - P_ZZ_ZZ_test))
print(P_ZZ_ZZ - sp.coo_matrix(P_ZZ_ZZ_test))

print("P_Z1Z_Z1Z")
print(np.linalg.norm(P_Z1Z_Z1Z.todense() - P_Z1Z_Z1Z_test))
print(P_Z1Z_Z1Z - sp.coo_matrix(P_Z1Z_Z1Z_test))

print("P_X_ZZ")
print(np.linalg.norm(P_X_ZZ.todense() - P_X_ZZ_test))
print(P_X_ZZ - sp.coo_matrix(P_X_ZZ_test))

print("P_X_Z1Z")
print(np.linalg.norm(P_X_Z1Z.T.todense() - P_Z1Z_X_test))
print(P_X_Z1Z - sp.coo_matrix(P_Z1Z_X_test.T))

print("P_Z1Z_ZZ")
print(np.linalg.norm(P_ZZ_Z1Z.todense().T - P_Z1Z_ZZ_test))
print(P_ZZ_Z1Z.T - sp.coo_matrix(P_Z1Z_ZZ_test))


# start_time = time.time()
#
# # find optimal coefficients
# coefficients_h = np.zeros((res_h, res_g, var_order))
# coefficients_g = np.zeros((res_h, res_g, var_order))
#
# for i, h in enumerate(hl):
#     for j, g in enumerate(gl):
#
#         print("i, j:", i, j)
#         P = P_ZZ_ZZ + g * g * P_X_X + h * h * P_Z_Z \
#             - g * (P_X_ZZ + P_X_ZZ.T) - h * (P_Z_ZZ + P_Z_ZZ.T) + h * g * (P_X_Z + P_X_Z.T)
#
#         P_inv = linalg.inv(P)
#
#         R_h = -R_Z_ZZ + g * R_Z_X + h * R_Z_Z
#         R_g = -R_X_ZZ + g * R_X_X + h * R_X_Z
#
#         coefficients_h[i, j, :] = 0.5 * P_inv @ R_h
#         coefficients_g[i, j, :] = 0.5 * P_inv @ R_g
#
# end_time = time.time()
# print("Time:", end_time - start_time)
# #############################################################


start_time = time.time()

# find optimal coefficients
coefficients_kappa = np.zeros((res_kappa, res_g, var_order))
coefficients_g = np.zeros((res_kappa, res_g, var_order))
for i, kappa in enumerate(kappal):
    for j, g in enumerate(gl):
        print(i, j)
        # print("i, j:", i, j)
        print("kappa, g:", kappa, g)
        P = P_ZZ_ZZ + g * g * P_X_X + kappa * kappa * P_Z1Z_Z1Z \
            + g * (P_X_ZZ + P_X_ZZ.T) - kappa * (P_ZZ_Z1Z + P_ZZ_Z1Z.T) - kappa * g * (P_X_Z1Z + P_X_Z1Z.T)


        R_kappa = -R_Z1Z_ZZ - g * R_Z1Z_X + kappa * R_Z1Z_Z1Z
        R_g = R_X_ZZ + g * R_X_X - kappa * R_X_Z1Z

        coefficients_kappa[i, j, :] = linalg.spsolve(P, 0.5 * R_kappa)
        coefficients_g[i, j, :] = linalg.spsolve(P, 0.5 * R_g)

end_time = time.time()
print("Time:", end_time - start_time)

name = "annni_coefficients_l" + str(l) + "_res_h" + str(res_kappa) + "_res_g" + str(res_g) + ".npz"
np.savez_compressed(name, hl=kappal, gl=gl, ck=coefficients_kappa, cg=coefficients_g)