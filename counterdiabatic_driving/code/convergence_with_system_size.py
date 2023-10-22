import numpy as np
import scipy.sparse as sp

l = 5
op_name_1 = "xxz"
op_name_2 = "xxz"

Ll = np.arange(l, 2 * l + 2)

for L in Ll:

    print("Sizes:", L, L + 1)
    name = "ltfi_optimization_matrices_P_" + op_name_1.upper() + "_" + op_name_2.upper()\
           + "_mpi_L=" + str(L) + "_l=" + str(l) + ".npz"

    mat_1 = sp.load_npz(name)
    name = "ltfi_optimization_matrices_P_" + op_name_1.upper() + "_" + op_name_2.upper() \
           + "_mpi_L=" + str(L + 1) + "_l=" + str(l) + ".npz"

    mat_2 = sp.load_npz(name)
    print("Difference:")
    print(mat_2 - mat_1)