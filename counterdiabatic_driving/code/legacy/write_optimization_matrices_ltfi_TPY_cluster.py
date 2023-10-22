import numpy as np
import sys
import multiprocessing as mp
from commute_stringgroups_no_quspin import *
m = maths()


def fill_P_matrices(index_tuple):

    i = index_tuple[0]
    j = index_tuple[1]

    # print("i, j:", i, j)
    # same terms ( i, j <-> j, i symmetry)
    if i <= j:
        # comm ZZ
        prod_ZZ_ZZ = C_ZZ[i] * C_ZZ[j]
        tr_ZZ_ZZ = prod_ZZ_ZZ.trace().real
        P_ZZ_ZZ[i + j * var_order] = -tr_ZZ_ZZ
        P_ZZ_ZZ[j + i * var_order] = -tr_ZZ_ZZ

        # comm X
        prod_X_X = C_X[i] * C_X[j]
        tr_X_X = prod_X_X.trace().real
        P_X_X[i + j * var_order] = -tr_X_X
        P_X_X[j + i * var_order] = -tr_X_X

        # comm Z
        prod_Z_Z = C_Z[i] * C_Z[j]
        tr_Z_Z = prod_Z_Z.trace().real
        P_Z_Z[i + j * var_order] = -tr_Z_Z
        P_Z_Z[j + i * var_order] = -tr_Z_Z

    # cross terms
    # comm ZZ
    prod_X_ZZ = C_X[i] * C_ZZ[j]
    tr_X_ZZ = prod_X_ZZ.trace().real
    P_X_ZZ[j + i * var_order] = -tr_X_ZZ

    prod_Z_ZZ = C_Z[i] * C_ZZ[j]
    tr_Z_ZZ = prod_Z_ZZ.trace().real
    P_Z_ZZ[j + i * var_order] = -tr_Z_ZZ

    # comm X
    prod_Z_X = C_Z[i] * C_X[j]
    tr_Z_X = prod_Z_X.trace().real
    P_Z_X[j + i * var_order] = -tr_Z_X


if __name__ == "__main__":

    L = int(sys.argv[1])
    l = int(sys.argv[2])
    number_of_processes = int(sys.argv[3])
    tr_I = 2 ** L


    # hamiltonians
    h_x = equation()
    for i in range(L):
        op = ''.join(roll(list('x' + '1' * (L - 1)), i))
        h_x[op] = 0.5

    h_z = equation()
    for i in range(L):
        op = ''.join(roll(list('z' + '1' * (L - 1)), i))
        h_z[op] = 0.5

    h_zz = equation()
    for i in range(L):
        op = ''.join(roll(list('zz' + '1' * (L - 2)), i))
        h_zz += equation({op: 0.25})

    # read in TPY operators up to range l
    variational_operators = []
    for k in np.arange(1, l + 1, 1):

        # fill the strings up to the correct system size
        op_file = "operators_TPY_l" + str(k) + ".txt"
        with open(op_file, "r") as readfile:
            for line in readfile:

                op_str = line[0:k] + '1' * (L - k)
                op_eq = equation()
                for i in range(L):
                    op = "".join(roll(list(op_str), i))
                    op_eq += equation({op: 1.0})
                    op_rev = "".join(roll(list(op_str[::-1]), i))
                    op_eq += equation({op_rev: 1.0})

                variational_operators.append(op_eq)

    var_order = len(variational_operators)

    # create matrices for optimization
    # commutators and R matrix
    C_Z = []
    C_X = []
    C_ZZ = []

    R_X_ZZ = np.zeros(var_order)
    R_X_X = np.zeros(var_order)
    R_X_Z = np.zeros(var_order)

    R_Z_ZZ = np.zeros(var_order)
    R_Z_X = np.zeros(var_order)
    R_Z_Z = np.zeros(var_order)

    for i in range(var_order):
        comm_Z = m.c(h_z, variational_operators[i])
        comm_ZZ = m.c(h_zz, variational_operators[i])
        comm_X = m.c(h_x, variational_operators[i])

        C_Z.append(comm_Z)
        C_X.append(comm_X)
        C_ZZ.append(comm_ZZ)

        prod_X_ZZ = h_x * comm_ZZ
        R_X_ZZ[i] = (2.j * prod_X_ZZ.trace()).real
        prod_X_Z = h_x * comm_Z
        R_X_Z[i] = (2.j * prod_X_Z.trace()).real
        prod_X_X = h_x * comm_X
        R_X_X[i] = (2.j * prod_X_X.trace()).real

        prod_Z_ZZ = h_z * comm_ZZ
        R_Z_ZZ[i] = (2.j * prod_Z_ZZ.trace()).real
        prod_Z_Z = h_z * comm_Z
        R_Z_Z[i] = (2.j * prod_Z_Z.trace()).real
        prod_Z_X = h_z * comm_X
        R_Z_X[i] = (2.j * prod_Z_X.trace()).real

    # P matrices
    P_ZZ_ZZ = mp.Array('d', var_order ** 2)
    P_X_ZZ = mp.Array('d', var_order ** 2)
    P_Z_ZZ = mp.Array('d', var_order ** 2)
    P_X_X = mp.Array('d', var_order ** 2)
    P_Z_X = mp.Array('d', var_order ** 2)
    P_Z_Z = mp.Array('d', var_order ** 2)

    pool = mp.Pool(processes=number_of_processes)
    comp = pool.map_async(fill_P_matrices, [(i, j) for i in range(var_order) for j in range(var_order)])
    comp.wait()

    P_ZZ_ZZ = np.array(P_ZZ_ZZ).reshape(var_order, var_order)
    P_X_ZZ = np.array(P_X_ZZ).reshape(var_order, var_order)
    P_Z_ZZ = np.array(P_Z_ZZ).reshape(var_order, var_order)
    P_X_X = np.array(P_X_X).reshape(var_order, var_order)
    P_Z_X = np.array(P_Z_X).reshape(var_order, var_order)
    P_Z_Z = np.array(P_Z_Z).reshape(var_order, var_order)

    name = "optimization_matrices_parallel_ltfi_L=" + str(L) + "_l=" + str(l) + ".npz"
    np.savez_compressed(name, R_X_ZZ=R_X_ZZ / tr_I, R_X_Z=R_X_Z / tr_I, R_X_X=R_X_X / tr_I, R_Z_X=R_Z_X / tr_I,
                        R_Z_Z=R_Z_Z / tr_I, R_Z_ZZ=R_Z_ZZ / tr_I, P_X_X=P_X_X / tr_I, P_X_ZZ=P_X_ZZ / tr_I,
                        P_Z_ZZ=P_Z_ZZ / tr_I, P_Z_X=P_Z_X / tr_I, P_Z_Z=P_Z_Z / tr_I, P_ZZ_ZZ=P_ZZ_ZZ / tr_I)



