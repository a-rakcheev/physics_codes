import zipfile
import io
import pauli_string_functions as pauli_func
import hamiltonians_32 as ham32
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt


def eigensystem(ham_0, ham_1, param):
    ham_tot = (1. - param) * ham_0 + param * ham_1
    ev, evec = np.linalg.eigh(ham_tot)
    return ev, evec

L = 2
# size = 2 ** L
#
# positions_z, labels_z = ham32.input_h_z(L)
# positions_x, labels_x = ham32.input_h_x(L)
# positions_y, labels_y = ham32.input_h_y(L)
# positions_zz, labels_zz = ham32.input_h_zz_1d(L, 1)
# positions_xx, labels_xx = ham32.input_h_xx_1d(L, 1)
# positions_yy, labels_yy = ham32.input_h_yy_1d(L, 1)
#
# print("inputs obtained")
#
# mat_x = ham32.operator_sum_real(L, 1, np.full(L, 1.0), positions_x, labels_x).todense()
# mat_y = ham32.operator_sum_complex(L, 1, np.full(L, 1.0), positions_y, labels_y).todense()
# mat_z = ham32.operator_sum_real(L, 1, np.full(L, 1.0), positions_z, labels_z).todense()
#
# mat_xx = ham32.operator_sum_real(L, 2, np.full(L, 1.0), positions_xx, labels_xx).todense()
# mat_yy = ham32.operator_sum_complex(L, 2, np.full(L, 1.0), positions_yy, labels_yy).todense()
# mat_zz = ham32.operator_sum_real(L, 2, np.full(L, 1.0), positions_zz, labels_zz).todense()



k_name = "0"
par = 1                             # parity sector

# create all operators as matrices
# create operators up to range r and save them in memoryoperators = []
name_zip = "operators/1d_chain_indices_and_periods.zip"
with zipfile.ZipFile(name_zip) as zipper:

    name = "1d_chain_TP_indices_and_periods_L=" + str(L) + "_k=" + k_name + ".npz"
    with io.BufferedReader(zipper.open(name, mode='r')) as f:

        data = np.load(f)
        periods = data["period"]
        parities = data["parity"]
        size = len(periods)

name_zip_op = "operators/operators_TP_L=" + str(L) + ".zip"
with zipfile.ZipFile(name_zip_op) as zipper_op:


    mat_name = "x_TP_L=" + str(L) + "_k=" + k_name + "_p=" + str(par) + ".npz"
    with io.BufferedReader(zipper_op.open(mat_name, mode='r')) as f_op:
        data = np.load(f_op)
        indptr = data["indptr"]
        indices = data["idx"]
        val = data["val"]
    mat_x = sp.csr_matrix((val, indices, indptr), shape=(size, size)).todense()

    mat_name = "y_TP_L=" + str(L) + "_k=" + k_name + "_p=" + str(par) + ".npz"
    with io.BufferedReader(zipper_op.open(mat_name, mode='r')) as f_op:
        data = np.load(f_op)
        indptr = data["indptr"]
        indices = data["idx"]
        val = data["val"]
    mat_y = sp.csr_matrix((val, indices, indptr), shape=(size, size)).todense()

    mat_name = "z_TP_L=" + str(L) + "_k=" + k_name + "_p=" + str(par) + ".npz"
    with io.BufferedReader(zipper_op.open(mat_name, mode='r')) as f_op:
        data = np.load(f_op)
        indptr = data["indptr"]
        indices = data["idx"]
        val = data["val"]
    mat_z = sp.csr_matrix((val, indices, indptr), shape=(size, size)).todense()


    mat_name = "xy_TP_L=" + str(L) + "_k=" + k_name + "_p=" + str(par) + ".npz"
    with io.BufferedReader(zipper_op.open(mat_name, mode='r')) as f_op:
        data = np.load(f_op)
        indptr = data["indptr"]
        indices = data["idx"]
        val = data["val"]
    mat_xy = sp.csr_matrix((val, indices, indptr), shape=(size, size)).todense()

    mat_name = "yz_TP_L=" + str(L) + "_k=" + k_name + "_p=" + str(par) + ".npz"
    with io.BufferedReader(zipper_op.open(mat_name, mode='r')) as f_op:
        data = np.load(f_op)
        indptr = data["indptr"]
        indices = data["idx"]
        val = data["val"]
    mat_yz = sp.csr_matrix((val, indices, indptr), shape=(size, size)).todense()

    mat_name = "xx_TP_L=" + str(L) + "_k=" + k_name + "_p=" + str(par) + ".npz"
    with io.BufferedReader(zipper_op.open(mat_name, mode='r')) as f_op:
        data = np.load(f_op)
        indptr = data["indptr"]
        indices = data["idx"]
        val = data["val"]
    mat_xx = sp.csr_matrix((val, indices, indptr), shape=(size, size)).todense()

    mat_name = "yy_TP_L=" + str(L) + "_k=" + k_name + "_p=" + str(par) + ".npz"
    with io.BufferedReader(zipper_op.open(mat_name, mode='r')) as f_op:
        data = np.load(f_op)
        indptr = data["indptr"]
        indices = data["idx"]
        val = data["val"]
    mat_yy = sp.csr_matrix((val, indices, indptr), shape=(size, size)).todense()

    mat_name = "zz_TP_L=" + str(L) + "_k=" + k_name + "_p=" + str(par) + ".npz"
    with io.BufferedReader(zipper_op.open(mat_name, mode='r')) as f_op:
        data = np.load(f_op)
        indptr = data["indptr"]
        indices = data["idx"]
        val = data["val"]
    mat_zz = sp.csr_matrix((val, indices, indptr), shape=(size, size)).todense()
    
mat_0 = np.zeros((4, 4))
mat_0[0, 0] = 4
mat_0[1, 1] = 2
mat_0[2, 2] = 2
mat_0[3, 3] = 4
mat_0[1, 2] = 2
mat_0[2, 1] = 2

# h_0 = mat_zz
# h_1 = (mat_xx + mat_yy + 2. * mat_z + 0.5 * mat_x)

h_0 = -mat_zz + 0.1 * mat_zz
h_1 = mat_x

start = 0.
end = 1.0
steps = 1001

params = np.linspace(start, end, steps)
spectrum = np.zeros((steps, size))
evecs = np.zeros((steps, size, size), dtype=np.complex128)

for i, p in enumerate(params):

    ev, evec = eigensystem(h_0, h_1, p)
    spectrum[i, :] = ev
    evecs[i, :, :] = evec

P_0 = np.absolute(evecs[:, :, 0]) ** 2
print(P_0[0, :])

plt.subplot(1, 2, 1)
plt.plot(params, spectrum, marker="o", markersize=1)

plt.grid()
plt.xlabel(r"$\lambda$")
plt.ylabel(r"$E$")


plt.subplot(1, 2, 2)
plt.pcolormesh(P_0.T, cmap="tab20c_r", vmin=0., vmax=1.0)

plt.colorbar()
plt.xlabel(r"$\lambda$")
plt.ylabel(r"$P$")


plt.show()