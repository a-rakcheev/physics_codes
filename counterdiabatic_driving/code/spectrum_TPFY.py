import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as linalg
import sys

import zipfile
import io

def parity(op_name):

    p = 0
    if op_name == op_name[::-1]:
        p = 1

    return p


# L = int(sys.argv[1])
L = 14
k_idx = 0  # momentum index (full momentum is (2 pi / L) * index), should be 0, 1, ..., L - 1
k_name = "0"  # currently 0 or pi (k_idx = L // 2) available
par = 1

s_0 = 1                                            # signs of operators
s_1 = 0
s_2 = 1

sign_0 = (-1) ** s_0
sign_1 = (-1) ** s_1
sign_2 = (-1) ** s_2

op_name_0 = "zz"                                # operators in the hamiltonian
op_name_1 = "z1z"
op_name_2 = "x"

res_1 = 50                                        # number of grid points on x axis
res_2 = 25                                       # number of grid points on y axis

start1 = 1.e-6
start2 = 1.e-6
end_1 = 1.
end_2 = 1.

params1 = np.linspace(start1, end_1, res_1)
params2 = np.linspace(start2, end_2, res_2)

# adjust factors due to parity
# the parity of the operator for example zz leads to double counting
# # when creating 1zz1 + zz11 + 11zz + z11z + 1zz1 + 11zz + zz11 + z11z, in contrast to
# # 1xy1 + xy11 + 11xy + y11x + 1yx1 + 11yx + yx11 + x11y
# # the parity is 0, 1 in these cases

factor_0 = 0.5 * (2. - parity(op_name_0))
factor_1 = 0.5 * (2. - parity(op_name_1))
factor_2 = 0.5 * (2. - parity(op_name_2))

# get size
size_dict = dict()
size_dict["8"] = 18
size_dict["10"] = 44
size_dict["12"] = 122
size_dict["14"] = 362
size_dict["16"] = 1162

size = size_dict[str(L)]

name_zip_op = "operators/operators_TPF_L=" + str(L) + ".zip"
with zipfile.ZipFile(name_zip_op) as zipper_op:

    # hamiltonians (if real)
    mat_name = op_name_0 + "_TPF_L=" + str(L) + "_k=" + k_name + "_p=" + str(par) + ".npz"
    with io.BufferedReader(zipper_op.open(mat_name, mode='r')) as f_op:
        data = np.load(f_op)
        indptr = data["indptr"]
        indices = data["idx"]
        val = data["val"]
    h_0 = factor_0 * sp.csr_matrix((val, indices, indptr), shape=(size, size)).real

    mat_name = op_name_1 + "_TPF_L=" + str(L) + "_k=" + k_name + "_p=" + str(par) + ".npz"
    with io.BufferedReader(zipper_op.open(mat_name, mode='r')) as f_op:
        data = np.load(f_op)
        indptr = data["indptr"]
        indices = data["idx"]
        val = data["val"]
    h_1 = factor_1 * sp.csr_matrix((val, indices, indptr), shape=(size, size)).real

    mat_name = op_name_2 + "_TPF_L=" + str(L) + "_k=" + k_name + "_p=" + str(par) + ".npz"
    with io.BufferedReader(zipper_op.open(mat_name, mode='r')) as f_op:
        data = np.load(f_op)
        indptr = data["indptr"]
        indices = data["idx"]
        val = data["val"]
    h_2 = factor_2 * sp.csr_matrix((val, indices, indptr), shape=(size, size)).real


evals = np.zeros((res_1, res_2, size))
evecs = np.zeros((res_1, res_2, size, size), dtype=np.float64)

print("Number of States:", size)
for i, p_1 in enumerate(params1):
    print("i:", i)

    for j, p_2 in enumerate(params2):

        h_tot = sign_0 * h_0 + sign_1 * p_1 * h_1 + sign_2 * p_2 * h_2
        ev, evec = np.linalg.eigh(h_tot.todense())

        evals[i, j, :] = ev
        evecs[i, j, :, :] = evec

# prefix_save = "C:/Users/ARakc/Dropbox/data/agp/"
prefix_save = "D:/Dropbox/data/agp/"
# prefix_save = ""

name = "spectrum_TPFY_L" + str(L) + "_op0=" + op_name_0 + "_op1=" + op_name_1 + "_op2=" + op_name_2\
       + "_s0=" + str(s_0) + "_s1=" + str(s_1) + "_s2=" + str(s_2) \
       + "_res_1=" + str(res_1) + "_res_2=" + str(res_2) + "_start1=" + str(start1).replace(".", "-") \
       + "_end1=" + str(end_1).replace(".", "-") + "_start2=" + str(start2).replace(".", "-") + "_end2=" \
       + str(end_2).replace(".", "-") + ".npz"

np.savez_compressed(prefix_save + name, ev=evals, evec=evecs, p1=params1, p2=params2)