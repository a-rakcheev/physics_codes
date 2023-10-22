import sys
import os
import numpy as np
import hamiltonians_32 as ham32
import numba as nb
import time

np.random.seed()
N = 4
runs = 1

for i in range(runs):
    randvec = np.random.rand(2 ** N) + 1.j * np.random.rand(2 ** N)
    randvec /= np.linalg.norm(randvec)

    randvec2 = np.copy(randvec)

    # magnetization

    for n in range(N):

        # full
        positions = np.array([[i + 1]], dtype=np.int64)
        labels = np.array([[1]], dtype=np.int8)

        op = ham32.operator_sum_real(N, 1, np.ones(1), positions, labels)
        m = randvec.T.conj() @ op @ randvec

        # matrix-free
        m2 = randvec2.T.conj() @ ham32.single_operator_complex_mf(randvec2, N, 1, 1., np.array([i + 1], dtype=np.int64), np.array([1], dtype=np.int8))
        # print(m.real - m2.real)

    # correlation
    lab = [[1, 1], [2, 2], [3, 3], [2, 3], [3, 2]]

    # labels
    for l in range(5):

        print("l:", l)
        # bonds
        count = 0
        for n in range(N):
            for k in range(n + 1, N):

                # full
                positions = np.array([[n + 1, k + 1]], dtype=np.int64)
                labels = np.array([lab[l]], dtype=np.int8)

                op = ham32.operator_sum_complex(N, 2, np.ones(1), positions, labels)
                print(op)
                c = randvec.T.conj() @ op @ randvec

                # matrix-free
                c2 = randvec2.T.conj() @ ham32.single_operator_complex_mf(randvec2, N, 2, 1., np.array([n + 1, k + 1], dtype=np.int64), np.array(lab[l], dtype=np.int8))

                # print(np.linalg.norm(op @ randvec - ham32.single_operator_complex_mf(randvec2, N, 2, 1., np.array([n + 1, k + 1], dtype=np.int64), np.array(lab[l], dtype=np.int8))))
                print(c, c2, c.real - c2.real)

                count += 1
