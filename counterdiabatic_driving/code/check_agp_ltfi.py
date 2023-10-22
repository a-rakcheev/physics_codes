import pickle
import numpy as np
import sys
import multiprocessing as mp
from commute_stringgroups_v2 import *

m = maths()


L = 4  # number of spins
res_x = 20  # number of grid points on x axis
res_y = 20  # number of grid points on y axis
order = 10  # order of commutator ansatz
l = 4  # range cutoff for variational strings
S = 2  # subspace size (for subspace computations)
beta = 0.01  # inverse temperature (for finite temperature computations)


xl = np.linspace(1.e-6, 1.5, res_x)
yl = np.linspace(1.e-6, 1.5, res_y)

metric_grid = mp.Array('d', res_x * res_y * 4)
prefix = "/home/artem/Dropbox/bu_research/data/"
name = prefix + "optimize_agp_ltfi_L" + str(L) + "_l" + str(l) + "_order" + str(order) + "_res_x" + str(
    res_x) + "_res_y" + str(
    res_y) + ".pkl"

with open(name, "rb") as readfile:
    agp_x = pickle.load(readfile)
    agp_y = pickle.load(readfile)

readfile.close()

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


for i in range(res_x):
    for j in range(res_y):

        print("i, j:", i, j)

        x = xl[i]
        y = yl[j]

        A_x = agp_x[i * res_y + j]
        A_y = agp_y[i * res_y + j]

        # define hamiltonian
        ham = h_zz - y * h_x - x * h_z
        ham_deriv_x = (-1) * h_z
        ham_deriv_y = (-1) * h_x

        comm_x = m.c(A_x, ham)
        comm_y = m.c(A_y, ham)

        chi_x = ham_deriv_x + 1.j * comm_x
        chi_y = ham_deriv_y + 1.j * comm_y

        # print((chi_x * chi_x).trace())
        print((m.c(chi_x, ham) * m.c(chi_x, ham)).trace())

        # print((chi_y * chi_y).trace())
        print((m.c(chi_y, ham) * m.c(chi_y, ham)).trace())


        # ham_trace = (ham * ham).trace().real
        #
        # metric = np.zeros((2, 2))
        # metric[0, 0] = (chi_x * chi_x).trace().real / ham_trace
        # metric[1, 1] = (chi_y * chi_y).trace().real / ham_trace
        # metric[0, 1] = (chi_x * chi_y).trace().real / ham_trace