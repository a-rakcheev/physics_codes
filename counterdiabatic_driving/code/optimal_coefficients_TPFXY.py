# create optimal coefficients for AGP for a two parameter Hamiltonian
# H = s_0 * H_0 + p_1 * s_1 * H_1 + p_2 * s_2 * H_2
# with parameters p_1, p_2 and signs s_i which are passed as 0/1 => (-1)^0 , (-1)^1

import scipy.sparse.linalg as linalg
import time
import sys

import numpy as np
import scipy.sparse as sp
import os.path
from os import path



def parity(op_name):
    p = 0
    if op_name == op_name[::-1]:
        p = 1

    return p


# # parameters
# l = int(sys.argv[1])                                             # range cutoff for variational strings
# L = int(sys.argv[2])                                             # number of spins
# res_1 = int(sys.argv[3])                                         # number of grid points on x axis
# res_2 = int(sys.argv[4])                                         # number of grid points on y axis

# parameters
l = 3  # range cutoff for variational strings
res_1 = 5  # number of grid points on x axis
res_2 = 5  # number of grid points on y axis

s_0 = 0  # signs of operators
s_1 = 0
s_2 = 0

op_name_0 = "xx"  # operators in the hamiltonian
op_name_1 = "yy"
op_name_2 = "zz"


start1 = 1.e-6
start2 = 1.e-6
end_1 = 2.
end_2 = 2.

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

factor_0 = 0.5 * (2. - parity(op_name_0))
factor_1 = 0.5 * (2. - parity(op_name_1))
factor_2 = 0.5 * (2. - parity(op_name_2))

print(factor_0, factor_1, factor_2)

# read in R and P matrices for the given operators

# R matrices
name = "optimization_matrices/optimization_matrices_R_" + op_name_1.upper() + "_" + op_name_0.upper() + "_TPFXY_l=" + str(
    l) + ".npz"
data = np.load(name)
R_1_0 = data["R"] * factor_1 * factor_0

name = "optimization_matrices/optimization_matrices_R_" + op_name_1.upper() + "_" + op_name_1.upper() + "_TPFXY_l=" + str(
    l) + ".npz"
data = np.load(name)
R_1_1 = data["R"] * factor_1 * factor_1

name = "optimization_matrices/optimization_matrices_R_" + op_name_1.upper() + "_" + op_name_2.upper() + "_TPFXY_l=" + str(
    l) + ".npz"
data = np.load(name)
R_1_2 = data["R"] * factor_1 * factor_2

name = "optimization_matrices/optimization_matrices_R_" + op_name_2.upper() + "_" + op_name_0.upper() + "_TPFXY_l=" + str(
    l) + ".npz"
data = np.load(name)
R_2_0 = data["R"] * factor_2 * factor_0

name = "optimization_matrices/optimization_matrices_R_" + op_name_2.upper() + "_" + op_name_1.upper() + "_TPFXY_l=" + str(
    l) + ".npz"
data = np.load(name)
R_2_1 = data["R"] * factor_2 * factor_1

name = "optimization_matrices/optimization_matrices_R_" + op_name_2.upper() + "_" + op_name_2.upper() + "_TPFXY_l=" + str(
    l) + ".npz"
data = np.load(name)
R_2_2 = data["R"] * factor_2 * factor_2

# P matrices
# symmetric
name = "optimization_matrices/optimization_matrices_P_" + op_name_0.upper() + "_" + op_name_0.upper() + "_TPFXY_l=" + str(
    l) + ".npz"
P_0_0 = sp.load_npz(name) * factor_0 * factor_0

name = "optimization_matrices/optimization_matrices_P_" + op_name_1.upper() + "_" + op_name_1.upper() + "_TPFXY_l=" + str(
    l) + ".npz"
P_1_1 = sp.load_npz(name) * factor_1 * factor_1

name = "optimization_matrices/optimization_matrices_P_" + op_name_2.upper() + "_" + op_name_2.upper() + "_TPFXY_l=" + str(
    l) + ".npz"
P_2_2 = sp.load_npz(name) * factor_2 * factor_2

# not symmetric, need to check if operators are ordered correctly
if path.exists(
        "optimization_matrices/optimization_matrices_P_" + op_name_0.upper() + "_" + op_name_1.upper() + "_TPFXY_l=" + str(
                l) + ".npz"):
    name = "optimization_matrices/optimization_matrices_P_" + op_name_0.upper() + "_" + op_name_1.upper() + "_TPFXY_l=" + str(
        l) + ".npz"
else:
    name = "optimization_matrices/optimization_matrices_P_" + op_name_1.upper() + "_" + op_name_0.upper() + "_TPFXY_l=" + str(
        l) + ".npz"
P_0_1 = sp.load_npz(name) * factor_0 * factor_1
P_0_1 = P_0_1 + P_0_1.T

if path.exists(
        "optimization_matrices/optimization_matrices_P_" + op_name_0.upper() + "_" + op_name_2.upper() + "_TPFXY_l=" + str(
                l) + ".npz"):
    name = "optimization_matrices/optimization_matrices_P_" + op_name_0.upper() + "_" + op_name_2.upper() + "_TPFXY_l=" + str(
        l) + ".npz"
else:
    name = "optimization_matrices/optimization_matrices_P_" + op_name_2.upper() + "_" + op_name_0.upper() + "_TPFXY_l=" + str(
        l) + ".npz"
P_0_2 = sp.load_npz(name) * factor_0 * factor_2
P_0_2 = P_0_2 + P_0_2.T

if path.exists(
        "optimization_matrices/optimization_matrices_P_" + op_name_1.upper() + "_" + op_name_2.upper() + "_TPFXY_l=" + str(
                l) + ".npz"):
    name = "optimization_matrices/optimization_matrices_P_" + op_name_1.upper() + "_" + op_name_2.upper() + "_TPFXY_l=" + str(
        l) + ".npz"
else:
    name = "optimization_matrices/optimization_matrices_P_" + op_name_2.upper() + "_" + op_name_1.upper() + "_TPFXY_l=" + str(
        l) + ".npz"
P_1_2 = sp.load_npz(name) * factor_1 * factor_2
P_1_2 = P_1_2 + P_1_2.T

# # check operators
# print("R_1_0", R_1_0)
# print("R_1_1", R_1_1)
# print("R_1_2", R_1_2)
# print("R_2_0", R_2_0)
# print("R_2_1", R_2_1)
# print("R_2_2", R_2_2)
#
# print("P_0_0", P_0_0)
# print("P_1_1", P_1_1)
# print("P_2_2", P_2_2)
#
# print("P_0_1", P_0_1 + P_0_1.T)
# print("P_0_2", P_0_2 + P_0_2.T)
# print("P_1_2", P_1_2 + P_1_2.T)

# start_time = time.time()

var_order = len(R_1_0)
# find optimal coefficients
coefficients_1 = np.zeros((res_1, res_2, var_order))
coefficients_2 = np.zeros((res_1, res_1, var_order))

start_time = time.time()
for i, p_1 in enumerate(params1):
    print("i", i)
    for j, p_2 in enumerate(params2):
        print("j", j)

        P = P_0_0 + (p_1 ** 2) * P_1_1 + (p_2 ** 2) * P_2_2 \
            + sign_0 * sign_1 * p_1 * P_0_1 \
            + sign_0 * sign_2 * p_2 * P_0_2 \
            + sign_1 * sign_2 * p_1 * p_2 * P_1_2

        R_1 = sign_1 * (sign_0 * R_1_0 + sign_1 * p_1 * R_1_1 + sign_2 * p_2 * R_1_2)
        R_2 = sign_2 * (sign_0 * R_2_0 + sign_1 * p_1 * R_2_1 + sign_2 * p_2 * R_2_2)

        coefficients_1[i, j, :] = linalg.spsolve(P, 0.5 * R_1)
        coefficients_2[i, j, :] = linalg.spsolve(P, 0.5 * R_2)

end_time = time.time()
print("Time:", end_time - start_time)

name = "optimal_coefficients_TPFXY_l" + str(l) + "_" + op_name_0.upper() + "_" + op_name_1.upper() \
       + "_" + op_name_2.upper() + "_s0=" + str(s_0) + "_s1=" + str(s_1) + "_s2=" + str(s_2) \
       + "_res_1=" + str(res_1) + "_res_2=" + str(res_2) + "_start1=" + str(start1).replace(".", "-") \
       + "_end1=" + str(end_1).replace(".", "-") + "_start2=" + str(start2).replace(".", "-") + "_end2=" \
       + str(end_2).replace(".", "-") + ".npz"

# np.savez_compressed(name, p1=params1, p2=params2, c1=coefficients_1, c2=coefficients_2)
print(coefficients_1)
print(coefficients_2)