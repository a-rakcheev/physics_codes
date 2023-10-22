import numpy as np
import matplotlib.pyplot as plt
import time
from commute_stringgroups_no_quspin import *
m = maths()

Ll = np.arange(4, 11, 1)
ll = [4]

times = np.zeros((len(Ll), len(ll)))
for idx_L, L in enumerate(Ll):
    for idx_l, l in enumerate(ll):

        print(idx_L, idx_l)
        start_time = time.time()

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


        # P matrix
        P_ZZ_ZZ = np.zeros((var_order, var_order))
        P_X_ZZ = np.zeros((var_order, var_order))
        P_Z_ZZ = np.zeros((var_order, var_order))
        P_ZZ_X = np.zeros((var_order, var_order))
        P_X_X = np.zeros((var_order, var_order))
        P_Z_X = np.zeros((var_order, var_order))
        P_ZZ_Z = np.zeros((var_order, var_order))
        P_X_Z = np.zeros((var_order, var_order))
        P_Z_Z = np.zeros((var_order, var_order))

        for i in range(var_order):
            for j in np.arange(i, var_order, 1):

                # comm ZZ
                prod_ZZ_ZZ = C_ZZ[i] * C_ZZ[j]
                tr_ZZ_ZZ = prod_ZZ_ZZ.trace().real
                P_ZZ_ZZ[i, j] = -tr_ZZ_ZZ
                P_ZZ_ZZ[j, i] = -tr_ZZ_ZZ

                prod_X_ZZ = C_X[i] * C_ZZ[j]
                tr_X_ZZ = prod_X_ZZ.trace().real
                P_X_ZZ[i, j] = -tr_X_ZZ
                P_X_ZZ[j, i] = -tr_X_ZZ

                prod_Z_ZZ = C_Z[i] * C_ZZ[j]
                tr_Z_ZZ = prod_Z_ZZ.trace().real
                P_Z_ZZ[i, j] = -tr_Z_ZZ
                P_Z_ZZ[j, i] = -tr_Z_ZZ

                # comm X
                prod_ZZ_X = C_ZZ[i] * C_X[j]
                tr_ZZ_X = prod_ZZ_X.trace().real
                P_ZZ_X[i, j] = -tr_ZZ_X
                P_ZZ_X[j, i] = -tr_ZZ_X

                prod_X_X = C_X[i] * C_X[j]
                tr_X_X = prod_X_X.trace().real
                P_X_X[i, j] = -tr_X_X
                P_X_X[j, i] = -tr_X_X

                prod_Z_X = C_Z[i] * C_X[j]
                tr_Z_X = prod_Z_X.trace().real
                P_Z_X[i, j] = -tr_Z_X
                P_Z_X[j, i] = -tr_Z_X

                # comm Z
                prod_ZZ_Z = C_ZZ[i] * C_Z[j]
                tr_ZZ_Z = prod_ZZ_Z.trace().real
                P_ZZ_Z[i, j] = -tr_ZZ_Z
                P_ZZ_Z[j, i] = -tr_ZZ_Z

                prod_X_Z = C_X[i] * C_Z[j]
                tr_X_Z = prod_X_Z.trace().real
                P_X_Z[i, j] = -tr_X_Z
                P_X_Z[j, i] = -tr_X_Z

                prod_Z_Z = C_Z[i] * C_Z[j]
                tr_Z_Z = prod_Z_Z.trace().real
                P_Z_Z[i, j] = -tr_Z_Z
                P_Z_Z[j, i] = -tr_Z_Z


        C_Z = None
        C_X = None
        C_ZZ = None

        end_time = time.time()
        times[idx_L, idx_l] = end_time - start_time

plt.plot(Ll, times[:, 0], marker="o", color="navy", ls="--")
plt.grid()

plt.xlabel("L")
plt.ylabel("time")
plt.title(r"Time - $l=2$")

plt.show()