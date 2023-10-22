import numpy as np
import scipy.sparse as sp
import os.path
from os import path
import zipfile
import io

def parity(op_name):

    p = 0
    if op_name == op_name[::-1]:
        p = 1

    return p

# create optimal coefficients for AGP for a two parameter Hamiltonian
# H = s_0 * H_0 + p_1 * s_1 * H_1 + p_2 * s_2 * H_2
# with parameters p_1, p_2 and signs s_i which are passed as 0/1 => (-1)^0 , (-1)^1

import scipy.sparse.linalg as linalg
import time
import sys

# # parameters
# l = int(sys.argv[1])                                             # range cutoff for variational strings
# L = int(sys.argv[2])                                             # number of spins
# res_h = int(sys.argv[3])                                         # number of grid points on x axis
# res_g = int(sys.argv[4])                                         # number of grid points on y axis

# parameters
l = 8                                             # range cutoff for variational strings

s_0 = 0                                            # signs of operators
s_1 = 1
s_2 = 1


op_name_0 = "zz"                                # operators in the hamiltonian
op_name_1 = "z"
op_name_2 = "x"


res_1 = 50                                        # number of grid points on x axis
res_2 = 25                                       # number of grid points on y axis
end_1 = 3.
end_2 = 1.5

params1 = np.linspace(1.e-6, end_1, res_1)
params2 = np.linspace(1.e-6, end_2, res_2)

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

# read in R and P matrices for the given operators
name_zip = "optimization_matrices/optimization_matrices_l=" + str(l) + ".zip"
with zipfile.ZipFile(name_zip) as zipper:

    # R matrices
    name = "optimization_matrices_R_" + op_name_1.upper() + "_" + op_name_0.upper() + "_TPY_l=" + str(
        l) + ".npz"

    with io.BufferedReader(zipper.open(name, mode='r')) as f:
        data = np.load(f)
        R_1_0 = data["R"] * factor_1 * factor_0

    name = "optimization_matrices_R_" + op_name_1.upper() + "_" + op_name_1.upper() + "_TPY_l=" + str(
        l) + ".npz"
    with io.BufferedReader(zipper.open(name, mode='r')) as f:
        data = np.load(f)
        R_1_1 = data["R"] * factor_1 * factor_1

    name = "optimization_matrices_R_" + op_name_1.upper() + "_" + op_name_2.upper() + "_TPY_l=" + str(
        l) + ".npz"
    with io.BufferedReader(zipper.open(name, mode='r')) as f:
        data = np.load(f)
        R_1_2 = data["R"] * factor_1 * factor_2

    name = "optimization_matrices_R_" + op_name_2.upper() + "_" + op_name_0.upper() + "_TPY_l=" + str(
        l) + ".npz"
    with io.BufferedReader(zipper.open(name, mode='r')) as f:
        data = np.load(f)
        R_2_0 = data["R"] * factor_2 * factor_0

    name = "optimization_matrices_R_" + op_name_2.upper() + "_" + op_name_1.upper() + "_TPY_l=" + str(
        l) + ".npz"
    with io.BufferedReader(zipper.open(name, mode='r')) as f:
        data = np.load(f)
        R_2_1 = data["R"] * factor_2 * factor_1

    name = "optimization_matrices_R_" + op_name_2.upper() + "_" + op_name_2.upper() + "_TPY_l=" + str(
        l) + ".npz"
    with io.BufferedReader(zipper.open(name, mode='r')) as f:
        data = np.load(f)
        R_2_2 = data["R"] * factor_2 * factor_2

    # P matrices
    # symmetric
    name = "optimization_matrices_P_" + op_name_0.upper() + "_" + op_name_0.upper() + "_TPY_l=" + str(
        l) + ".npz"
    with io.BufferedReader(zipper.open(name, mode='r')) as f:
        P_0_0 = sp.load_npz(f) * factor_0 * factor_0

    name = "optimization_matrices_P_" + op_name_1.upper() + "_" + op_name_1.upper() + "_TPY_l=" + str(
        l) + ".npz"
    with io.BufferedReader(zipper.open(name, mode='r')) as f:
        P_1_1 = sp.load_npz(f) * factor_1 * factor_1

    name = "optimization_matrices_P_" + op_name_2.upper() + "_" + op_name_2.upper() + "_TPY_l=" + str(
        l) + ".npz"
    with io.BufferedReader(zipper.open(name, mode='r')) as f:
        P_2_2 = sp.load_npz(f) * factor_2 * factor_2

    # not symmetric, need to check if operators are ordered correctly
    name = "optimization_matrices_P_" + op_name_0.upper() + "_" + op_name_1.upper() + "_TPY_l=" + str(
        l) + ".npz"
    if name not in zipper.namelist():
        name = "optimization_matrices_P_" + op_name_1.upper() + "_" + op_name_0.upper() + "_TPY_l=" + str(l) + ".npz"

    with io.BufferedReader(zipper.open(name, mode='r')) as f:
        P_0_1 = sp.load_npz(f) * factor_0 * factor_1
    P_0_1 = P_0_1 + P_0_1.T

    name = "optimization_matrices_P_" + op_name_0.upper() + "_" + op_name_2.upper() + "_TPY_l=" + str(
        l) + ".npz"
    if name not in zipper.namelist():
        name = "optimization_matrices_P_" + op_name_2.upper() + "_" + op_name_0.upper() + "_TPY_l=" + str(l) + ".npz"

    with io.BufferedReader(zipper.open(name, mode='r')) as f:
        P_0_2 = sp.load_npz(f) * factor_0 * factor_2
    P_0_2 = P_0_2 + P_0_2.T

    name = "optimization_matrices_P_" + op_name_1.upper() + "_" + op_name_2.upper() + "_TPY_l=" + str(
        l) + ".npz"
    if name not in zipper.namelist():
        name = "optimization_matrices_P_" + op_name_2.upper() + "_" + op_name_1.upper() + "_TPY_l=" + str(l) + ".npz"

    with io.BufferedReader(zipper.open(name, mode='r')) as f:
        P_1_2 = sp.load_npz(f) * factor_1 * factor_2
    P_1_2 = P_1_2 + P_1_2.T



start_time = time.time()

var_order = len(R_1_0)

# find optimal coefficients
coefficients_1 = np.zeros((res_1, res_2, var_order))
coefficients_2 = np.zeros((res_1, res_1, var_order))
for i, p_1 in enumerate(params1):
    for j, p_2 in enumerate(params2):

        print("i, j:", i, j)
        
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

name = "optimal_coefficients_TPY_l" + str(l) + "_" + op_name_0.upper() + "_" + op_name_1.upper() \
       + "_" + op_name_2.upper() + "_s0=" + str(s_0) + "_s1=" + str(s_1) + "_s2=" + str(s_2) \
       + "_res_1=" + str(res_1) + "_res_2=" + str(res_2) + "_end1=" + str(end_1).replace(".", "-") \
       + "_end2=" + str(end_2).replace(".", "-") + ".npz"

# prefix = ""
# prefix = "C:/Users/ARakc/Dropbox/data/agp/"
prefix = "D:/Dropbox/data/agp/"

np.savez_compressed(prefix + name, p1=params1, p2=params2, c1=coefficients_1, c2=coefficients_2)