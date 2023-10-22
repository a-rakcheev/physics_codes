import numpy as np
import scipy.sparse as sp

def parity(op_name):

    p = 0
    if op_name == op_name[::-1]:
        p = 1

    return p


l = 7
L = 13
op_name_1 = "z"
op_name_2 = "z"

c_1 = (0.5 ** len(op_name_1)) * 0.5 * (2. - parity(op_name_1))
c_2 = (0.5 ** len(op_name_2)) * 0.5 * (2. - parity(op_name_2))
print(c_1, c_2, c_1 * c_2)

name = "optimization_matrices/ltfi_optimization_matrices_P_" + op_name_1.upper() + "_" + op_name_2.upper() \
       + "_mpi_L=" + str(L) + "_l=" + str(l) + ".npz"
# name = "optimization_matrices/optimization_matrices_P_" + op_name_1.upper() + "_" + op_name_2.upper() \
#        + "_TPFY_l=" + str(l) + ".npz"



P = sp.load_npz(name)
P = P / (c_1 * c_2)

name = "optimization_matrices_P_" + op_name_1.upper() + "_" + op_name_2.upper() \
       + "_TPY_l=" + str(l) + ".npz"
# name = "optimization_matrices_P_" + op_name_1.upper() + "_" + op_name_2.upper() \
#        + "_TPFY_l=" + str(l) + ".npz"
sp.save_npz(name, P)

