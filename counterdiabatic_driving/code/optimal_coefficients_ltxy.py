# create optimal coefficients for AGP for a two parameter Hamiltonian
# H = s_0 * H_0 + p_1 * s_1 * H_1 + p_2 * s_2 * H_2
# with parameters p_1, p_2 and signs s_i which are passed as 0/1 => (-1)^0 , (-1)^1

# uses mpi for shared or distributed memory

import scipy.sparse.linalg as linalg
import sys

import numpy as np
import scipy.sparse as sp
import os.path
from os import path

from mpi4py import MPI

def parity(op_name):
    p = 0
    if op_name == op_name[::-1]:
        p = 1

    return p


# # parameters
# l = int(sys.argv[1])                                             # range cutoff for variational strings
# res_1 = int(sys.argv[3])                                         # number of grid points on x axis
# res_2 = int(sys.argv[4])                                         # number of grid points on y axis
#
# start1 = float(sys.argv[5])
# start2 = float(sys.argv[6])
# end_1 = float(sys.argv[7])
# end_2 = float(sys.argv[8])
#
# s_0 = int(sys.argv[9])
# s_1 = int(sys.argv[10])
# s_2 = int(sys.argv[11])


# parameters
l = 6  # range cutoff for variational strings
res_1 = 100  # number of grid points on x axis
res_2 = 100  # number of grid points on y axis

s_0 = 1  # signs of operators
s_1 = 1
s_2 = 1

start1 = 1.e-6
start2 = 1.e-6
end_1 = 2.0
end_2 = 2.0


# initialize
comm = MPI.COMM_WORLD

# get rank
rank = comm.Get_rank()

# size
number_of_processes = comm.Get_size()

# time each process
start_time = MPI.Wtime()

params1 = np.linspace(start1, end_1, res_1)
params2 = np.linspace(start2, end_2, res_2)

sign_0 = (-1) ** s_0
sign_1 = (-1) ** s_1
sign_2 = (-1) ** s_2

# adjust factors due to parity
# the parity of the operator for example zz leads to double counting
# # when creating 1zz1 + zz11 + 11zz + z11z + 1zz1 + 11zz + zz11 + z11z, in contrast to
# # 1xy1 + xy11 + 11xy + y11x + 1yx1 + 11yx + yx11 + x11y
# # the parity is 0, 1 in these cases

factor_XX = 0.5 * (2. - parity("xx"))
factor_YY = 0.5 * (2. - parity("yy"))
factor_X = 0.5 * (2. - parity("x"))
factor_Z = 0.5 * (2. - parity("z"))


# read in R and P matrices for the given operators

# R matrices
name = "optimization_matrices/optimization_matrices_R_X_XX_TPY_l=" + str(
    l) + ".npz"
data = np.load(name)
R_X_XX = data["R"] * factor_X * factor_XX

name = "optimization_matrices/optimization_matrices_R_X_YY_TPY_l=" + str(
    l) + ".npz"
data = np.load(name)
R_X_YY = data["R"] * factor_X * factor_YY

name = "optimization_matrices/optimization_matrices_R_X_X_TPY_l=" + str(
    l) + ".npz"
data = np.load(name)
R_X_X = data["R"] * factor_X * factor_X

name = "optimization_matrices/optimization_matrices_R_X_Z_TPY_l=" + str(
    l) + ".npz"
data = np.load(name)
R_X_Z = data["R"] * factor_X * factor_Z

R_1_0 = R_X_XX + R_X_YY
R_1_1 = R_X_X
R_1_2 = R_X_Z

name = "optimization_matrices/optimization_matrices_R_Z_XX_TPY_l=" + str(
    l) + ".npz"
data = np.load(name)
R_Z_XX = data["R"] * factor_Z * factor_XX

name = "optimization_matrices/optimization_matrices_R_Z_YY_TPY_l=" + str(
    l) + ".npz"
data = np.load(name)
R_Z_YY = data["R"] * factor_Z * factor_YY

name = "optimization_matrices/optimization_matrices_R_Z_X_TPY_l=" + str(
    l) + ".npz"
data = np.load(name)
R_Z_X = data["R"] * factor_Z * factor_X

name = "optimization_matrices/optimization_matrices_R_Z_Z_TPY_l=" + str(
    l) + ".npz"
data = np.load(name)
R_Z_Z = data["R"] * factor_Z * factor_Z

R_2_0 = R_Z_XX + R_Z_YY
R_2_1 = R_Z_X
R_2_2 = R_Z_Z



# P matrices
# symmetric
name = "optimization_matrices/optimization_matrices_P_XX_XX_TPY_l=" + str(
    l) + ".npz"
P_XX_XX = sp.load_npz(name) * factor_XX * factor_XX

name = "optimization_matrices/optimization_matrices_P_YY_YY_TPY_l=" + str(
    l) + ".npz"
P_YY_YY = sp.load_npz(name) * factor_YY * factor_YY

name = "optimization_matrices/optimization_matrices_P_X_X_TPY_l=" + str(
    l) + ".npz"
P_X_X = sp.load_npz(name) * factor_X * factor_X

name = "optimization_matrices/optimization_matrices_P_Z_Z_TPY_l=" + str(
    l) + ".npz"
P_Z_Z = sp.load_npz(name) * factor_Z * factor_Z

# not symmetric, need to check if operators are ordered correctly
if path.exists(
        "optimization_matrices/optimization_matrices_P_XX_YY_TPY_l=" + str(
                l) + ".npz"):
    name = "optimization_matrices/optimization_matrices_P_XX_YY_TPY_l=" + str(
        l) + ".npz"
else:
    name = "optimization_matrices/optimization_matrices_P_YY_XX_TPY_l=" + str(
        l) + ".npz"
P_XX_YY = sp.load_npz(name) * factor_XX * factor_YY
P_XX_YY = P_XX_YY + P_XX_YY.T

if path.exists(
        "optimization_matrices/optimization_matrices_P_XX_X_TPY_l=" + str(
                l) + ".npz"):
    name = "optimization_matrices/optimization_matrices_P_XX_X_TPY_l=" + str(
        l) + ".npz"
else:
    name = "optimization_matrices/optimization_matrices_P_X_XX_TPY_l=" + str(
        l) + ".npz"
P_XX_X = sp.load_npz(name) * factor_XX * factor_X
P_XX_X = P_XX_X + P_XX_X.T

if path.exists(
        "optimization_matrices/optimization_matrices_P_YY_X_TPY_l=" + str(
                l) + ".npz"):
    name = "optimization_matrices/optimization_matrices_P_YY_X_TPY_l=" + str(
        l) + ".npz"
else:
    name = "optimization_matrices/optimization_matrices_P_X_YY_TPY_l=" + str(
        l) + ".npz"
P_YY_X = sp.load_npz(name) * factor_YY * factor_X
P_YY_X = P_YY_X + P_YY_X.T

if path.exists(
        "optimization_matrices/optimization_matrices_P_XX_Z_TPY_l=" + str(
                l) + ".npz"):
    name = "optimization_matrices/optimization_matrices_P_XX_Z_TPY_l=" + str(
        l) + ".npz"
else:
    name = "optimization_matrices/optimization_matrices_P_Z_XX_TPY_l=" + str(
        l) + ".npz"
P_XX_Z = sp.load_npz(name) * factor_XX * factor_Z
P_XX_Z = P_XX_Z + P_XX_Z.T

if path.exists(
        "optimization_matrices/optimization_matrices_P_YY_Z_TPY_l=" + str(
                l) + ".npz"):
    name = "optimization_matrices/optimization_matrices_P_YY_Z_TPY_l=" + str(
        l) + ".npz"
else:
    name = "optimization_matrices/optimization_matrices_P_Z_YY_TPY_l=" + str(
        l) + ".npz"
P_YY_Z = sp.load_npz(name) * factor_YY * factor_Z
P_YY_Z = P_YY_Z + P_YY_Z.T

if path.exists(
        "optimization_matrices/optimization_matrices_P_X_Z_TPY_l=" + str(
                l) + ".npz"):
    name = "optimization_matrices/optimization_matrices_P_X_Z_TPY_l=" + str(
        l) + ".npz"
else:
    name = "optimization_matrices/optimization_matrices_P_Z_X_TPY_l=" + str(
        l) + ".npz"
P_X_Z = sp.load_npz(name) * factor_X * factor_Z
P_X_Z = P_X_Z + P_X_Z.T


P_0_0 = P_XX_XX + P_YY_YY + P_XX_YY
P_0_1 = P_XX_X + P_YY_X
P_0_2 = P_XX_Z + P_YY_Z
P_1_1 = P_X_X
P_1_2 = P_X_Z
P_2_2 = P_Z_Z


var_order = len(R_1_0)
step = res_1 // number_of_processes
rest = res_2 - number_of_processes * step

coeff_1 = []
coeff_2 = []

# time each process
start_time = MPI.Wtime()

for i in range(rank * step, (rank + 1) * step, 1):
    print(i, rank, flush=True)
    p_1 = params1[i]

    for j, p_2 in enumerate(params2):
        P = P_0_0 + (p_1 ** 2) * P_1_1 + (p_2 ** 2) * P_2_2 \
            + sign_0 * sign_1 * p_1 * P_0_1 \
            + sign_0 * sign_2 * p_2 * P_0_2 \
            + sign_1 * sign_2 * p_1 * p_2 * P_1_2

        R_1 = sign_1 * (sign_0 * R_1_0 + sign_1 * p_1 * R_1_1 + sign_2 * p_2 * R_1_2)
        R_2 = sign_2 * (sign_0 * R_2_0 + sign_1 * p_1 * R_2_1 + sign_2 * p_2 * R_2_2)

        coeff_1.append([i, j, linalg.spsolve(P, 0.5 * R_1)])
        coeff_2.append([i, j, linalg.spsolve(P, 0.5 * R_2)])

if rank < rest:
    i = number_of_processes * step + rank
    print(i, rank, flush=True)
    p_1 = params1[i]

    for j, p_2 in enumerate(params2):
        P = P_0_0 + (p_1 ** 2) * P_1_1 + (p_2 ** 2) * P_2_2 \
            + sign_0 * sign_1 * p_1 * P_0_1 \
            + sign_0 * sign_2 * p_2 * P_0_2 \
            + sign_1 * sign_2 * p_1 * p_2 * P_1_2

        R_1 = sign_1 * (sign_0 * R_1_0 + sign_1 * p_1 * R_1_1 + sign_2 * p_2 * R_1_2)
        R_2 = sign_2 * (sign_0 * R_2_0 + sign_1 * p_1 * R_2_1 + sign_2 * p_2 * R_2_2)

        coeff_1.append([i, j, linalg.spsolve(P, 0.5 * R_1)])
        coeff_2.append([i, j, linalg.spsolve(P, 0.5 * R_2)])

# gather list
process_list_1 = comm.gather(coeff_1, root=0)
process_list_2 = comm.gather(coeff_2, root=0)

end_time = MPI.Wtime()
print("Time:", end_time - start_time, rank)

if rank == 0:

    coefficients_1 = np.zeros((res_1, res_2, var_order))
    coefficients_2 = np.zeros((res_1, res_2, var_order))

    for p_list in process_list_1:
        for tri_list in p_list:
            i = tri_list[0]
            j = tri_list[1]
            coefficients_1[i, j, :] = tri_list[2]

    process_list_1 = None
    for p_list in process_list_2:
        for tri_list in p_list:
            i = tri_list[0]
            j = tri_list[1]
            coefficients_2[i, j, :] = tri_list[2]

    process_list_2 = None


    name = "optimal_coefficients_ltxy_TPY_l" + str(l) + "_s0=" + str(s_0) + "_s1=" + str(s_1) + "_s2=" + str(s_2) \
           + "_res_1=" + str(res_1) + "_res_2=" + str(res_2) + "_start1=" + str(start1).replace(".", "-") \
           + "_end1=" + str(end_1).replace(".", "-") + "_start2=" + str(start2).replace(".", "-") + "_end2=" \
           + str(end_2).replace(".", "-") + ".npz"

    np.savez_compressed(name, p1=params1, p2=params2, c1=coefficients_1, c2=coefficients_2)
