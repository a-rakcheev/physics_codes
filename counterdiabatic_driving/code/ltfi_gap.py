import numpy as np
import qa_functions as qa
import scipy.sparse.linalg as spla
import sys

# parameters

L = int(sys.argv[1])
h_res = 100
g_res = 50
hl = np.linspace(1.e-10, 1.5, h_res)
gl = np.linspace(1.e-10, 0.75, g_res)
bc = "pbc"
k_max = 4

# save correlations and spin expectation vals in arrays
energies = np.zeros((h_res, g_res, k_max))

# hamiltonians (Spin 1/2)
sigma = qa.spin_matrix(L)
h_z = qa.hamiltonian_z_sparse(L, sigma, 1.0)
h_zz = qa.hamiltonian_j_sparse(L, sigma, bc)
h_x = qa.hamiltonian_x_sparse_from_diag(L, 1.0)

for i, h in enumerate(hl):
    for j, g in enumerate(gl):

        print(h, g)

        h_tot = h_zz + h * h_z + g * h_x
        ev = spla.eigsh(h_tot, k=k_max, which="SA", return_eigenvectors=False)
        energies[i, j, :] = ev

np.savez_compressed("ltfi_spectrum_L" + str(L) + ".npz", hl=hl, gl=gl, ev=energies)
