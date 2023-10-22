# evaluate the expectation values of the agp in order to evaluate metrics
# the evaluation is done in the K=0, P=1 sector

import zipfile
import io

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import scipy.sparse as sp
import time
from mpl_toolkits.axes_grid1 import make_axes_locatable

# parameters
def parity(op_name):

    p = 0
    if op_name == op_name[::-1]:
        p = 1

    return p

# parameters
L = 10                              # number of spins
l = 8                               # range cutoff for variational strings
L_coeff = 2 * l - 1                 # number of spins of the coefficient computation
r = 7                               # evaluate up to range (needs to be <= l)

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

param_label_1 = r"$\kappa$"
param_label_2 = r"$g$"
step1 = 1                            # sampling along x axis (every n-th point)
step2 = 1                            # sampling along y axis (every n-th point)


# adjust factors due to parity
# the parity of the operator for example zz leads to double counting
# # when creating 1zz1 + zz11 + 11zz + z11z + 1zz1 + 11zz + zz11 + z11z, in contrast to
# # 1xy1 + xy11 + 11xy + y11x + 1yx1 + 11yx + yx11 + x11y
# # the parity is 0, 1 in these cases

factor_0 = 0.5 * (2. - parity(op_name_0))
factor_1 = 0.5 * (2. - parity(op_name_1))
factor_2 = 0.5 * (2. - parity(op_name_2))


# coefficients
prefix = "C:/Users/ARakc/Dropbox/data/agp/"
# prefix = "D:/Dropbox/data/agp/"
# prefix = ""

name = "optimal_coefficients_TPFY_l" + str(l) + "_" + op_name_0.upper() + "_" + op_name_1.upper() \
       + "_" + op_name_2.upper() + "_s0=" + str(s_0) + "_s1=" + str(s_1) + "_s2=" + str(s_2) \
       + "_res_1=" + str(res_1) + "_res_2=" + str(res_2) + "_start1=" + str(start1).replace(".", "-") \
       + "_end1=" + str(end_1).replace(".", "-") + "_start2=" + str(start2).replace(".", "-") + "_end2=" \
       + str(end_2).replace(".", "-") + ".npz"

data = np.load(prefix + name)
coefficients_1 = data["c1"]
coefficients_2 = data["c2"]


# operators
k_idx = 0                           # momentum index (full momentum is (2 pi / L) * index), should be 0, 1, ..., L - 1
k_name = "0"                        # currently 0 or pi (k_idx = L // 2) available
par = 1                             # parity sector

# create operators up to range r and save them in memoryoperators = []
# get size
size_dict = dict()
size_dict["8"] = 18
size_dict["10"] = 44
size_dict["12"] = 122
size_dict["14"] = 362
size_dict["16"] = 1162

size = size_dict[str(L)]

operators = []
op_names = []

for k in np.arange(2, r + 1, 1):
    # fill the strings up to the correct system size
    op_file = "operators/operators_TPFY_l" + str(k) + ".txt"
    with open(op_file, "r") as readfile:
        for line in readfile:

            op_str = line[0:k]
            op_names.append(op_str)

print("op_names read", len(op_names))

name_zip_op = "operators/operators_TPF_L=" + str(L) + ".zip"
with zipfile.ZipFile(name_zip_op) as zipper_op:

    for op_str in op_names:
        mat_name = op_str + "_TPF_L=" + str(L) + "_k=" + k_name + "_p=" + str(par) + ".npz"

        with io.BufferedReader(zipper_op.open(mat_name, mode='r')) as f_op:
            data = np.load(f_op)
            indptr = data["indptr"]
            indices = data["idx"]
            val = data["val"]

        mat = sp.csr_matrix((val, indices, indptr), shape=(size, size))
        operators.append(mat)

print("operators created")

prefix = "C:/Users/ARakc/Dropbox/data/agp/"
# prefix = "D:/Dropbox/data/agp/"
# prefix = ""

name = prefix + "groundstate_TPFY_L" + str(L) + "_op0=" + op_name_0 + "_op1=" + op_name_1 + "_op2=" + op_name_2\
       + "_s0=" + str(s_0) + "_s1=" + str(s_1) + "_s2=" + str(s_2) \
       + "_res_1=" + str(res_1) + "_res_2=" + str(res_2) + "_start1=" + str(start1).replace(".", "-") \
       + "_end1=" + str(end_1).replace(".", "-") + "_start2=" + str(start2).replace(".", "-") + "_end2=" \
       + str(end_2).replace(".", "-") + ".npz"
data = np.load(name)
evec = data["evec"]


# for each parameter point compute the
# compute metric through expectation values

metric = np.zeros((res_1 // step1, res_2 // step2, 2, 2))

start_time = time.time()
for i in range(res_1 // step1):
    print(i)
    for j in range(res_2 // step2):

        c_1 = coefficients_1[i * step1, j * step2, :]
        c_2 = coefficients_2[i * step1, j * step2, :]

        A_1 = sp.csr_matrix((size, size), dtype=np.complex128)
        A_2 = sp.csr_matrix((size, size), dtype=np.complex128)

        for k in range(len(operators)):

            mat = operators[k]

            A_1 = A_1 + c_1[k] * mat
            A_2 = A_2 + c_2[k] * mat

        ground_state = evec[i * step1, j * step2, :]

        state_1 = A_1 @ ground_state
        state_2 = A_2 @ ground_state

        metric[i, j, 0, 0] = (state_1.T @ state_1 - (ground_state.T @ state_1) ** 2).real
        metric[i, j, 1, 1] = (state_2.T @ state_2 - (ground_state.T @ state_2) ** 2).real
        metric[i, j, 0, 1] = (state_2.T @ state_1 - (ground_state.T @ state_1) * (ground_state.T @ state_2)).real
        metric[i, j, 1, 0] = metric[i, j, 0, 1]

end_time = time.time()
print("Time:", end_time - start_time)

# plotting
plt.figure(1, figsize=(6, 3.25), constrained_layout=True)
cmap = "jet"

xl = params1[::step1]
yl = params2[::step2]
X, Y = np.meshgrid(xl, yl)

major_u = np.zeros_like(X)
major_v = np.zeros_like(X)
minor_u = np.zeros_like(X)
minor_v = np.zeros_like(X)

major_norm = np.zeros_like(X)
minor_norm = np.zeros_like(X)
norm = np.zeros_like(X)

for i, x in enumerate(xl):
    for j, y in enumerate(yl):

        g = metric[i, j, :, :]
        ev, evec = np.linalg.eigh(g)
        idx_sort = np.argsort(np.absolute(ev))

        major_u[j, i] = evec[0, idx_sort[1]]
        major_v[j, i] = evec[1, idx_sort[1]]
        major_norm[j, i] = ev[idx_sort[1]]

        minor_u[j, i] = evec[0, idx_sort[0]]
        minor_v[j, i] = evec[1, idx_sort[0]]
        minor_norm[j, i] = ev[idx_sort[0]]

        norm[j, i] = np.sqrt(np.absolute(ev[0] * ev[1]))

weight_major = major_norm / major_norm.max()
weight_minor = minor_norm / major_norm.max()
v_min = weight_minor.min()

plt.grid()
plt.quiver(X, Y, minor_u, minor_v, norm / L, norm=colors.LogNorm(vmin=10 ** -3, vmax=5 * 10 ** 0), cmap=cmap, pivot="tail")
plt.quiver(X, Y, -minor_u, -minor_v, norm / L, norm=colors.LogNorm(vmin=10 ** -3, vmax=5 * 10 ** 0), cmap=cmap, pivot="tail")

plt.colorbar()

plt.xlim(xl[0], xl[-1])
plt.ylim(yl[0], yl[-1])
plt.xlabel(param_label_1)
plt.ylabel(param_label_2)

name = "fidelity_metric_TPFY_L" + str(L) + "_l=" + str(l) + "_r=" + str(r) + "_op0=" + op_name_0 + "_op1=" + op_name_1 \
       + "_op2=" + op_name_2 + "_s0=" + str(s_0) + "_s1=" + str(s_1) + "_s2=" + str(s_2) + "_res_1=" + str(res_1) \
       + "_res_2=" + str(res_2) + "_start1=" + str(start1).replace(".", "-") + "_end1=" + str(end_1).replace(".", "-") \
       + "_start2=" + str(start2).replace(".", "-") + "_end2=" + str(end_2).replace(".", "-") + ".pdf"

plt.savefig(name, format="pdf")
plt.show()

