import numpy as np
import scipy.sparse as sp

l = 7
L = 2 * l - 1
op_name_1 = "z"
op_name_2 = "zz"

num_op = np.loadtxt("operators/operators_TPY_size_full.txt").astype(int)
size = num_op[l - 1]

name = "optimization_matrices/ltfi_optimization_matrices_P_" + op_name_1.upper() + "_" + op_name_2.upper() \
       + "_mp_L=" + str(L) + "_l=" + str(l) + ".npz"

data = np.load(name)
mat = data["P"]


if op_name_1 != op_name_2:

    P = sp.csr_matrix(mat)

else:

    indices = np.triu_indices(size)

    # # test
    # P_triu = np.zeros((size, size))
    # P_triu[indices] = mat

    idx_1 = []
    idx_2 = []
    val = []

    for k in range(len(mat)):

        if k % 1000 == 0:
            print(k, len(mat))
        i = indices[0][k]
        j = indices[1][k]
        v = mat[k]

        idx_1.append(i)
        idx_2.append(j)
        val.append(v)

        if i != j:

            idx_2.append(i)
            idx_1.append(j)
            val.append(v)

    P = sp.csr_matrix((val, (idx_1, idx_2)), shape=(size, size))
    P.eliminate_zeros()


# # test
# print("Diff:", np.linalg.norm(mat - mat.T))
#
# P_csr = sp.triu(mat).todense()
# print("Diff:", np.linalg.norm(P_triu - P_csr))
save_name = "optimization_matrices/ltfi_optimization_matrices_P_" + op_name_1.upper() + "_" + op_name_2.upper() \
       + "_mpi_L=" + str(L) + "_l=" + str(l) + ".npz"
sp.save_npz(save_name, P)
