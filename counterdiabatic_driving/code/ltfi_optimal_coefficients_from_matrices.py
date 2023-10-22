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
l = 7                                             # range cutoff for variational strings
# L = 2 * l - 1                                             # number of spins
L = 13
res_h = 50                                        # number of grid points on x axis
res_g = 25                                        # number of grid points on y axis


hl = np.linspace(1.e-6, 1.5, res_h)
gl = np.linspace(1.e-6, 0.75, res_g)

# R matrices
name = "optimization_matrices/ltfi_optimization_matrices_R_L=" + str(L) + "_l=" + str(l) + ".npz"
data = np.load(name)

R_X_ZZ = data["R_X_ZZ"]
R_X_Z = data["R_X_Z"]
R_X_X = data["R_X_X"]
R_Z_X = data["R_Z_X"]
R_Z_Z = data["R_Z_Z"]
R_Z_ZZ = data["R_Z_ZZ"]

var_order = len(R_Z_Z)

# P matrices

# symmetric
name = "optimization_matrices/ltfi_optimization_matrices_P_X_X_mpi_L=" + str(L) + "_l=" + str(l) + ".npz"
P_X_X = sp.load_npz(name)

name = "optimization_matrices/ltfi_optimization_matrices_P_Z_Z_mpi_L=" + str(L) + "_l=" + str(l) + ".npz"
P_Z_Z = sp.load_npz(name)

name = "optimization_matrices/ltfi_optimization_matrices_P_ZZ_ZZ_mpi_L=" + str(L) + "_l=" + str(l) + ".npz"
P_ZZ_ZZ = sp.load_npz(name)

# full
name = "optimization_matrices/ltfi_optimization_matrices_P_X_Z_mpi_L=" + str(L) + "_l=" + str(l) + ".npz"
P_X_Z = sp.load_npz(name)

name = "optimization_matrices/ltfi_optimization_matrices_P_X_ZZ_mpi_L=" + str(L) + "_l=" + str(l) + ".npz"
P_X_ZZ = sp.load_npz(name)

name = "optimization_matrices/ltfi_optimization_matrices_P_Z_ZZ_mpi_L=" + str(L) + "_l=" + str(l) + ".npz"
P_Z_ZZ = sp.load_npz(name)

# test
L_test = 12
# full
name = "optimization_matrices/optimization_matrices_parallel_ltfi_L=" + str(L_test) + "_l=" + str(l) + ".npz"
data = np.load(name)

R_X_ZZ_test = data["R_X_ZZ"] / L_test
R_X_Z_test = data["R_X_Z"] / L_test
R_X_X_test = data["R_X_X"] / L_test
R_Z_X_test = data["R_Z_X"] / L_test
R_Z_Z_test = data["R_Z_Z"] / L_test
R_Z_ZZ_test = data["R_Z_ZZ"] / L_test

P_X_X_test = data["P_X_X"] / L_test
P_X_ZZ_test = data["P_X_ZZ"] / L_test
P_Z_ZZ_test = data["P_Z_ZZ"] / L_test
P_Z_X_test = data["P_Z_X"] / L_test
P_Z_Z_test = data["P_Z_Z"] / L_test
P_ZZ_ZZ_test = data["P_ZZ_ZZ"] / L_test

print("R Matrices")
print("R_X_X")
print(np.linalg.norm(R_X_X - R_X_X_test))

print("R_X_Z")
print(np.linalg.norm(R_X_Z - R_X_Z_test))

print("R_X_ZZ")
print(np.linalg.norm(R_X_ZZ - R_X_ZZ_test))

print("R_Z_X")
print(np.linalg.norm(R_Z_X - R_Z_X_test))

print("R_Z_Z")
print(np.linalg.norm(R_Z_Z - R_Z_Z_test))

print("R_Z_ZZ")
print(np.linalg.norm(R_Z_ZZ - R_Z_ZZ_test))



print("P Matrices:")
print("P_X_X")
print(np.linalg.norm(P_X_X.todense() - P_X_X_test))
print(P_X_X - sp.coo_matrix(P_X_X_test))

print("P_Z_Z")
print(np.linalg.norm(P_Z_Z.todense() - P_Z_Z_test))
print(P_Z_Z - sp.coo_matrix(P_Z_Z_test))

print("P_ZZ_ZZ")
print(np.linalg.norm(P_ZZ_ZZ.todense() - P_ZZ_ZZ_test))
print(P_ZZ_ZZ - sp.coo_matrix(P_ZZ_ZZ_test))

print("P_X_Z")
print(np.linalg.norm(P_X_Z.T.todense() - P_Z_X_test))
print(P_X_Z - sp.coo_matrix(P_Z_X_test.T))

print("P_X_ZZ")
print(np.linalg.norm(P_X_ZZ.todense() - P_X_ZZ_test))
print(P_X_ZZ - sp.coo_matrix(P_X_ZZ_test))

print("P_Z_ZZ")
print(np.linalg.norm(P_Z_ZZ.todense() - P_Z_ZZ_test))
print(P_Z_ZZ - sp.coo_matrix(P_Z_ZZ_test))


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
coefficients_h = np.zeros((res_h, res_g, var_order))
coefficients_g = np.zeros((res_h, res_g, var_order))
for i, h in enumerate(hl):
    for j, g in enumerate(gl):
        print(i, j)
        # print("i, j:", i, j)
        P = P_ZZ_ZZ + g * g * P_X_X + h * h * P_Z_Z \
            - g * (P_X_ZZ + P_X_ZZ.T) - h * (P_Z_ZZ + P_Z_ZZ.T) + h * g * (P_X_Z + P_X_Z.T)


        R_h = -R_Z_ZZ + g * R_Z_X + h * R_Z_Z
        R_g = -R_X_ZZ + g * R_X_X + h * R_X_Z

        coefficients_h[i, j, :] = linalg.spsolve(P, 0.5 * R_h)
        coefficients_g[i, j, :] = linalg.spsolve(P, 0.5 * R_g)

end_time = time.time()
print("Time:", end_time - start_time)

name = "ltfi_coefficients_L" + str(L) + "_l" + str(l) + "_res_h" + str(
    res_h) + "_res_g" + str(res_g) + ".npz"
np.savez_compressed(name, hl=hl, gl=gl, ch=coefficients_h, cg=coefficients_g)