# find the indices of the available operators from a higher symmetry list
# the base list is the TPY list
import numpy as np
import scipy.sparse as sp


l = 8
op_name_1 = "z1z"
op_name_2 = "z1z"

# names = ["x", "z", "xx", "xz", "yy", "zz", "x1x", "x1z", "xxx", "xxz",
#          "xyy", "xzx", "xzz", "y1y", "yxy", "yyz", "yzy", "z1z", "zxz", "zzz"]
#
# for i, op_name_1 in enumerate(names):
#     print(op_name_1)
#     for op_name_2 in names[i:]:

# get operator indices
data = np.load("conversion_matrix_TPY_TPFY_l=" + str(l) + ".npz")
indices = data["arr_0"]

# get full matrix
name = "optimization_matrices/optimization_matrices_P_" + op_name_1.upper() \
     + "_" + op_name_2.upper() + "_TPY_l=" + str(l) + ".npz"

mat = sp.load_npz(name).tocoo()

# find all entries
entries = list(zip(mat.row, mat.col, mat.data))

idx1 = []
idx2 = []
val = []

# iterate over all entries and check if indices are available
for k, triplet in enumerate(entries):

     ind1 = np.searchsorted(indices, triplet[0])
     if ind1 < len(indices):
            ind2 = np.searchsorted(indices, triplet[1])
            if ind2 < len(indices):

                   if indices[ind1] == triplet[0] and indices[ind2] == triplet[1]:

                          idx1.append(ind1)
                          idx2.append(ind2)
                          val.append(triplet[2])

mat = sp.csr_matrix((val, (idx1, idx2)), (len(indices), len(indices)))

name = "optimization_matrices_P_" + op_name_1.upper() + "_" + op_name_2.upper() \
     + "_TPFY_l=" + str(l) + ".npz"

sp.save_npz(name, mat)