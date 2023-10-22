import numpy as np
import scipy.sparse as sp

for L in range(2, 9, 1):

    print("L", L)
    # L = 2                       # number of sites
    k_idx = 0  # momentum index (full momentum is (2 pi / L) * index), should be 0, 1, ..., L - 1
    k_name = "0"  # currently 0 or pi (k_idx = L // 2) available
    par = 1

    if k_idx >= L or k_idx < 0:
        raise (ValueError, "k index needs to be between 0 and L - 1")

    name = "1d_chain_TP_indices_and_periods_L=" + str(L) + "_k=" + k_name + ".npz"
    data = np.load(name)
    states = data["idx"]
    periods = data["period"]
    parities = data["parity"]

    x_plus = np.zeros(len(periods))
    for i in range(len(periods)):

        period = periods[i]
        parity = parities[i]

        norm = (L ** 2) / (period * (1. + parity % 2))
        x_plus[i] = L / (np.sqrt(norm) * np.sqrt(2) ** L)

    op_name = "x"
    mat_name = op_name + "_TP_L=" + str(L) + "_k=" + k_name + "_p=" + str(par) + ".npz"
    data = np.load(mat_name)
    indptr = data["indptr"]
    indices = data["idx"]
    val = data["val"]
    h_mat = sp.csr_matrix((val, indices, indptr), shape=(len(periods), len(periods))).todense()
    ev, evec = np.linalg.eigh(h_mat)
    vx = evec[:, -1].flatten()
    # print(vx)
    # print(x_plus)
    print("Norm:", np.linalg.norm(vx - x_plus, 2), np.linalg.norm(vx + x_plus, 2))