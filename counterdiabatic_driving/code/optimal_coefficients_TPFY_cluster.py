# create optimal coefficients for AGP for a two parameter Hamiltonian
# H = s_0 * H_0 + p_1 * s_1 * H_1 + p_2 * s_2 * H_2
# with parameters p_1, p_2 and signs s_i which are passed as 0/1 => (-1)^0 , (-1)^1

# uses mpi for shared or distributed memory

import sys
import zipfile
import io

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as linalg
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
#
# op_name_0 = str(sys.argv[12])
# op_name_1 = str(sys.argv[13])
# op_name_2 = str(sys.argv[14])


# parameters
l = 8  # range cutoff for variational strings
res_1 = 50  # number of grid points on x axis
res_2 = 25  # number of grid points on y axis

s_0 = 1  # signs of operators
s_1 = 0
s_2 = 1

op_name_0 = "zz"  # operators in the hamiltonian
op_name_1 = "z1z"
op_name_2 = "x"

start1 = 1.e-6
start2 = 1.e-6
end_1 = 1.0
end_2 = 1.0


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

factor_0 = 0.5 * (2. - parity(op_name_0))
factor_1 = 0.5 * (2. - parity(op_name_1))
factor_2 = 0.5 * (2. - parity(op_name_2))


name_zip = "optimization_matrices/optimization_matrices_l=" + str(l) + ".zip"
with zipfile.ZipFile(name_zip) as zipper:

    # R matrices
    name = "optimization_matrices_R_" + op_name_1.upper() + "_" + op_name_0.upper() + "_TPFY_l=" + str(
        l) + ".npz"

    with io.BufferedReader(zipper.open(name, mode='r')) as f:
        data = np.load(f)
        R_1_0 = data["R"] * factor_1 * factor_0

    name = "optimization_matrices_R_" + op_name_1.upper() + "_" + op_name_1.upper() + "_TPFY_l=" + str(
        l) + ".npz"
    with io.BufferedReader(zipper.open(name, mode='r')) as f:
        data = np.load(f)
        R_1_1 = data["R"] * factor_1 * factor_1

    name = "optimization_matrices_R_" + op_name_1.upper() + "_" + op_name_2.upper() + "_TPFY_l=" + str(
        l) + ".npz"
    with io.BufferedReader(zipper.open(name, mode='r')) as f:
        data = np.load(f)
        R_1_2 = data["R"] * factor_1 * factor_2

    name = "optimization_matrices_R_" + op_name_2.upper() + "_" + op_name_0.upper() + "_TPFY_l=" + str(
        l) + ".npz"
    with io.BufferedReader(zipper.open(name, mode='r')) as f:
        data = np.load(f)
        R_2_0 = data["R"] * factor_2 * factor_0

    name = "optimization_matrices_R_" + op_name_2.upper() + "_" + op_name_1.upper() + "_TPFY_l=" + str(
        l) + ".npz"
    with io.BufferedReader(zipper.open(name, mode='r')) as f:
        data = np.load(f)
        R_2_1 = data["R"] * factor_2 * factor_1

    name = "optimization_matrices_R_" + op_name_2.upper() + "_" + op_name_2.upper() + "_TPFY_l=" + str(
        l) + ".npz"
    with io.BufferedReader(zipper.open(name, mode='r')) as f:
        data = np.load(f)
        R_2_2 = data["R"] * factor_2 * factor_2


    # P matrices
    # symmetric
    name = "optimization_matrices_P_" + op_name_0.upper() + "_" + op_name_0.upper() + "_TPFY_l=" + str(
        l) + ".npz"
    with io.BufferedReader(zipper.open(name, mode='r')) as f:
        P_0_0 = sp.load_npz(f) * factor_0 * factor_0

    name = "optimization_matrices_P_" + op_name_1.upper() + "_" + op_name_1.upper() + "_TPFY_l=" + str(
        l) + ".npz"
    with io.BufferedReader(zipper.open(name, mode='r')) as f:
        P_1_1 = sp.load_npz(f) * factor_1 * factor_1

    name = "optimization_matrices_P_" + op_name_2.upper() + "_" + op_name_2.upper() + "_TPFY_l=" + str(
        l) + ".npz"
    with io.BufferedReader(zipper.open(name, mode='r')) as f:
        P_2_2 = sp.load_npz(f) * factor_2 * factor_2

    # not symmetric, need to check if operators are ordered correctly
    name = "optimization_matrices_P_" + op_name_0.upper() + "_" + op_name_1.upper() + "_TPFY_l=" + str(
        l) + ".npz"
    if name not in zipper.namelist():
        name = "optimization_matrices_P_" + op_name_1.upper() + "_" + op_name_0.upper() + "_TPFY_l=" + str(l) + ".npz"

    with io.BufferedReader(zipper.open(name, mode='r')) as f:
        P_0_1 = sp.load_npz(f) * factor_0 * factor_1
    P_0_1 = P_0_1 + P_0_1.T

    name = "optimization_matrices_P_" + op_name_0.upper() + "_" + op_name_2.upper() + "_TPFY_l=" + str(
        l) + ".npz"
    if name not in zipper.namelist():
        name = "optimization_matrices_P_" + op_name_2.upper() + "_" + op_name_0.upper() + "_TPFY_l=" + str(l) + ".npz"

    with io.BufferedReader(zipper.open(name, mode='r')) as f:
        P_0_2 = sp.load_npz(f) * factor_0 * factor_2
    P_0_2 = P_0_2 + P_0_2.T

    name = "optimization_matrices_P_" + op_name_1.upper() + "_" + op_name_2.upper() + "_TPFY_l=" + str(
        l) + ".npz"
    if name not in zipper.namelist():
        name = "optimization_matrices_P_" + op_name_2.upper() + "_" + op_name_1.upper() + "_TPFY_l=" + str(l) + ".npz"

    with io.BufferedReader(zipper.open(name, mode='r')) as f:
        P_1_2 = sp.load_npz(f) * factor_1 * factor_2
    P_1_2 = P_1_2 + P_1_2.T


var_order = len(R_1_0)
step = res_1 // number_of_processes
rest = res_1 - number_of_processes * step

coeff_1 = []
coeff_2 = []

# time each process
start_time = MPI.Wtime()

for i in range(rank * step, (rank + 1) * step, 1):
    p_1 = params1[i]
    print(i, rank, flush=True)

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
    p_1 = params1[i]
    print(i, rank, flush=True)

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


    name = "optimal_coefficients_TPFY_l" + str(l) + "_" + op_name_0.upper() + "_" + op_name_1.upper() \
           + "_" + op_name_2.upper() + "_s0=" + str(s_0) + "_s1=" + str(s_1) + "_s2=" + str(s_2) \
           + "_res_1=" + str(res_1) + "_res_2=" + str(res_2) + "_start1=" + str(start1).replace(".", "-") \
           + "_end1=" + str(end_1).replace(".", "-") + "_start2=" + str(start2).replace(".", "-") + "_end2=" \
           + str(end_2).replace(".", "-") + ".npz"

    np.savez_compressed(name, p1=params1, p2=params2, c1=coefficients_1, c2=coefficients_2)

