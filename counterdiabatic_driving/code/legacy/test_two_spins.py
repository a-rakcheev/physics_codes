# test first orders gauge potentials for tfi
# in annealing protocol

# the protocol is (1-s)H_x + sH_z
# therefore (d/ds)H = H_z-H_x = (H - H_x) / s
# s = t/T, therefore (d/dt)s = 1/T
# this sets the prefactor of the gauge potential

import sys
print(sys.version)
import numpy as np
from commute_stringgroups_v2 import *

# parameters
N = 2               # number of spins
order = 1          # order of variational gauge potential

# hamiltonians
m = maths()
J = 1.
h = 2.


h_zz = equation()
for i in range(N):
    op = ''.join(roll(list('zz' + '1' * (N - 2)), i))
    h_zz[op] = 1.

h_x = equation()
for i in range(N):
    op = ''.join(roll(list('x' + '1' * (N - 1)), i))
    h_x[op] = 1.

h_z = equation()
for i in range(N):
    op = ''.join(roll(list('z' + '1' * (N - 1)), i))
    h_z[op] = 1.

h_yz = equation()
for i in range(N):
    op = ''.join(roll(list('yz' + '1' * (N - 2)), i))
    h_yz[op] = 1.

h_y = equation()
for i in range(N):
    op = ''.join(roll(list('y' + '1' * (N - 1)), i))
    h_y[op] = 1.


# create variational gauge potential for the TFI
ham = -h * h_z - J * h_zz
ham_deriv = -h * h_x

print("Hamiltonian:", ham)
print("Hamiltonian Derivative:", ham_deriv)

# we need a list of equations for the different orders
# and then another list of coefficients


# gauge_pot = [(h ** 2) * h_y, h * J * h_yz]


# compute the gauge potential hamiltonians
# note the factor of 1.j in the 1st order
gauge_pot = []
for i in range(order):

    if i == 0:

        pot = 1.j * m.c(ham, ham_deriv)
        gauge_pot.append(pot)

    else:

        pot = gauge_pot[i - 1]
        pot = m.c(ham, pot)
        pot = m.c(ham, pot)
        gauge_pot.append(pot)

# put in variational gauge potential

print("Gauge Potential (unoptimized):", gauge_pot)

# # optimize the coefficients
# C = []
# P = np.zeros((order, order))
# R = np.zeros(order)
#
# # create commutators and fill R matrix
# for i in range(order):
#
#     print("Order", i + 1)
#     comm = m.c(ham, gauge_pot[i])
#     print("[H, A]", comm)
#     C.append(comm)
#     prod = ham_deriv * comm
#     print("H_deriv * [H, A]", prod.trace())
#     R[i] = (2.j * prod.trace()).real
#
# # fill P matrix
# for i in range(order):
#     for j in np.arange(i, order, 1):
#
#         prod = C[i] * C[j]
#         print("prod:", prod)
#         tr = prod.trace().real
#         print("Trace:", tr)
#         P[i, j] = -tr
#         P[j, i] = -tr
#
# print("R:", R)
# print("P:", P)
# print("Eigenvalues:", np.linalg.eigvalsh(P))
#
# # the optimal coefficient is given by 1/2 * (P)^(-1) R
# # if (-P) is positive definite
# print("Inverse:", np.linalg.inv(P))
# coeff = 0.5 * np.dot(np.linalg.inv(P), R)
# print("Coefficient Vector:", coeff)
#
#
# # add up operators from gauge potential, since the analytical formula is defined for pure pauli strings
#
# gauge_operator = gauge_pot[0] * coeff[0]
# for i in range(order - 1):
#
#     gauge_operator += gauge_pot[i + 1] * coeff[i + 1]
#
# print("Optimized Gauge Potential:", gauge_operator)
#
# # exact gauge potential
# exact_gauge_pot = 0.5 * (h ** 2 / (h ** 2 - J ** 2)) * h_y - 0.5 * (h * J / (h ** 2 - J ** 2)) * h_yz
# print("Exact Gauge Potential:", exact_gauge_pot)


# test shifts
print(shifts("1zx1"))
print(shifts_and_conjugates("1zx1"))

# test op split
gauge_full = gauge_pot[0]
for i in range(len(gauge_pot) - 1):

    gauge_full += gauge_pot[i + 1]

print("Gauge Full:", gauge_full)
print("Gauge Operators:", split_equation_shifts_and_conjugates(gauge_full))