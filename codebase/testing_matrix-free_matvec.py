import sys
import os
import numpy as np
import hamiltonians_32 as ham32
import numba as nb
import time

np.random.seed()

# test
# N = int(sys.argv[1])
N = 16
bc = 0
J = np.full(N, 1.0)
s = np.random.rand(1)[0]
runs = 10

# parallelism - seems not to work (or is the overhead to large?)
# num_threads = 2
# nb.set_num_threads(num_threads)
# os.environ["OMP_NUM_THREADS"] = str(1)
# os.environ["MKL_NUM_THREADS"] = str(1)
# os.environ["NUMBA_PARALLEL_DIAGNOSTICS"] = "4"

positions_z, labels_z = ham32.input_h_z(N)
positions_x, labels_x = ham32.input_h_x(N)
positions_zz, labels_zz = ham32.input_h_zz_1d(N, bc)

mat_z = ham32.operator_sum_real_diag(N, 1, J, positions_z, labels_z)
mat_x = ham32.operator_sum_real(N, 1, J, positions_x, labels_x)
mat_zz = ham32.operator_sum_real_diag(N, 2, J, positions_zz, labels_zz)
mat_ising = mat_z + mat_zz
ham_ising = mat_ising.diagonal()


# # chech correctness - done
# start_time = time.time()
# for i in range(runs):
#
#     test_vec = np.random.rand(2 ** N) + 1.j * np.random.rand(2 ** N)
#     test_vec /= np.linalg.norm(test_vec)
#
#     # true multiply
#     mult_vec = s * mat_ising @ test_vec - (1 - s) * mat_x @ test_vec
#
#     # matrix-free multiply
#     mult_mf_vec = ham32.matvec_qa(ham_ising, test_vec, s, N)
#
#     print(i, np.linalg.norm(mult_vec - mult_mf_vec))
#
# end_time = time.time()
# print("Time:", end_time - start_time)


# check timing - done
# matrix-free matvec very competitive - outperforms the standard sparse matvec for N >= 20 !!!


# normal sparse matvec
start_time = time.time()
for i in range(runs):

    test_vec = np.random.rand(2 ** N) + 1.j * np.random.rand(2 ** N)
    test_vec /= np.linalg.norm(test_vec)

    ham_tot = s * mat_ising - (1 - s) * mat_x

    # true multiply
    mult_vec = ham_tot @ test_vec

end_time = time.time()
print("Time sparse matvec:", end_time - start_time)

# matrix-free matvec
start_time = time.time()
for i in range(runs):

    test_vec = np.random.rand(2 ** N) + 1.j * np.random.rand(2 ** N)
    test_vec /= np.linalg.norm(test_vec)

    # matrix-free multiply
    mult_mf_vec = ham32.matvec_qa(ham_ising, test_vec, s, N)

end_time = time.time()
print("Time matrix-free matvec:", end_time - start_time)

# chech parallelism - doesnt work (or too large overhead)
# print("threading layer:", nb.threading_layer(), ", threads:", nb.get_num_threads())
