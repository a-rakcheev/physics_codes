import sys
sys.path.append("/home/artem/Dropbox/codebase/")
sys.path.append("C:/Users/rakche_a-adm/Dropbox/codebase/")
sys.path.append("/mnt/d64a440d-7b6c-45ed-aa6c-c9d991212d84/Dropbox/codebase/")

import numpy as np
import hamiltonians_32 as ham32
import matplotlib.pyplot as plt

# L = int(sys.argv[1])
L = 6
res = 10001                                        # number of grid points on x axis
hl = np.linspace(0.0001, 5.0, res)

plt.figure(1, figsize=(18, 5))

# identity
h_id = L * np.ones(2 ** L)

# field
positions_z, labels_z = ham32.input_h_z(L)
fields_z = np.ones(L)
h_z = ham32.operator_sum_real(L, 1, fields_z, positions_z, labels_z).diagonal()

# input for ZZ hamiltonians
positions_zz = []
positions_z1z = []
positions_z11z = []
labels_zz = []


for i in range(L):
    positions_zz.append([i + 1, (i + 1) % L + 1])
    positions_z1z.append([i + 1, (i + 2) % L + 1])
    positions_z11z.append([i + 1, (i + 3) % L + 1])

    labels_zz.append([3, 3])

positions_zz = np.array(positions_zz, dtype=np.int64)
positions_z1z = np.array(positions_z1z, dtype=np.int64)
positions_z11z = np.array(positions_z11z, dtype=np.int64)
labels_zz = np.array(labels_zz, dtype=np.int8)

h_zz = ham32.operator_sum_real_diag(L, 2, np.ones(L), positions_zz, labels_zz).diagonal()
h_z1z = ham32.operator_sum_real_diag(L, 2, np.ones(L), positions_z1z, labels_zz).diagonal()
h_z11z = ham32.operator_sum_real_diag(L, 2, np.ones(L), positions_z11z, labels_zz).diagonal()

# ham_zdz
h_zdz = 0.25 * (h_id + 2 * h_z + h_zz) + (1 / 2 ** 6) * 0.25 * (h_id + 2 * h_z + h_z1z) + (1 / 3 ** 6) * 0.25 * (h_id + 2 * h_z + h_z11z) 


# arrays for measurements
evals = np.zeros((res, 4))
evals_num = np.zeros((res, 10))
idx_num = np.zeros(res)

evals_ana = np.zeros((res, 4))

for i, h in enumerate(hl):

    h_tot = h_zdz - h * 0.5 * (h_z + h_id)
    idx = np.argmin(h_tot)

    # print(h, idx, np.binary_repr(idx, L))
    evals_num[i, :] = np.sort(h_tot)[0:10] / L
    idx_num[i] = idx

for d in range(4):
    
    # analytic
    if d == 0:
        evals_ana[:, d] = -hl + 1. + (1 / 2 ** 6) + (1 / 3 ** 6)
    elif d == 1:
        evals_ana[:, d] = -hl / 2 + (1 / 2 ** 7)
    elif d == 2:
        evals_ana[:, d] = -hl / 3 + (1 / 3 ** 7)
    elif d == 3:
        evals_ana[:, d] = -hl / 4

    # # state = np.zeros(size, dtype=np.complex128)
    # idx = 0

    # for k in range(L // (d + 1)):
    #     idx += 2 ** (k * (d + 1))

    # print(d, idx, np.binary_repr(idx, L))
    # print("Z:", h_z[idx] / L, "N:", 0.5 * (h_id + h_z)[idx] / L)
    # print("ZZ:", h_zz[idx] / L, "NN:", 0.25 * (h_id + 2 * h_z + h_zz)[idx] / L)
    # print("Z1Z:", h_z1z[idx] / L, "N1N:", 0.25 * (h_id + 2 * h_z + h_z1z)[idx] / L)
    # print("Z11Z:", h_z11z[idx] / L, "N11N:", 0.25 * (h_id + 2 * h_z + h_z11z)[idx] / L)
    # print("NdN:", h_zdz[idx] / L)



plt.figure(1, figsize=(12, 5))
plt.subplot(1, 3, 1)

for d in range(4):
    plt.plot(hl, evals_ana[:, d], label=r"$ |\Psi_{" + str(d) + r"}\rangle $")

plt.grid()
plt.legend()

plt.ylim(-2.5, 1.)
plt.xscale("log")

plt.xlabel(r"$h$", fontsize=12)
plt.ylabel(r"$\epsilon_{0}$", fontsize=12)
plt.title("analytic")


plt.subplot(1, 3, 2)

# for d in range(4):
#     plt.plot(hl, evals[:, d], label=r"$ |\Psi_{" + str(d) + r"}\rangle $")

for i in range(4):
    plt.plot(hl, evals_num[:, i + 1] - evals_num[:, 0], color="black", lw=1)

plt.grid()
# plt.legend()

# plt.ylim(-2.5, 1.)
plt.xscale("log")
plt.yscale("log")

plt.xlabel(r"$h$", fontsize=12)
plt.ylabel(r"$\epsilon_{0}$", fontsize=12)
plt.title("numeric")


plt.subplot(1, 3, 3)

plt.plot(hl, idx_num)
plt.grid()

plt.xscale("log")
plt.yscale("log")

plt.xlabel(r"$h$", fontsize=12)
plt.ylabel(r"$\mathcal{I}_{0}$", fontsize=12)
plt.title("numeric")



plt.subplots_adjust(wspace=0.25, hspace=0.25, top=0.925, bottom=0.1, left=0.1, right=0.95)
plt.show()