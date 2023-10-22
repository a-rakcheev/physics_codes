import numpy as np
from commute_stringgroups_v2 import*
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 11:51:29 2019

@author: Jonathan Wurtz
"""

# hamiltonians
m = maths()
## Make sure equation() and m.* are already in enviornment


# Make an equation...

# This is the same as $\sigma_x^0 + 2.3\sigma_y^0\sigma_y^1$ on 3 spins
E1 = equation({'x11': 1, 'yy1': 2.3})
E2 = equation({'z11': 3, '11y': -2})  # Another equation.

N = 3


h_x = equation()
for i in range(N):
    # This is my way of making strings of operators. There are probably easier ways.
    op = ''.join(roll(list('x' + '1' * (N - 1)), i))
    h_x[op] = 0.5

h_zz = equation()
for i in range(N):
    # This is my way of making strings of operators. There are probably easier ways.
    op = ''.join(roll(list('zz' + '1' * (N - 2)), i))
    h_zz[op] = 0.25

h_zz2 = equation()
for i in range(N):
    # This is my way of making strings of operators. There are probably easier ways.
    op = ''.join(roll(list('z1z' + '1' * (N - 3)), i))
    h_zz2[op] = 0.25


print(h_x)
print(h_zz)
print(h_zz2)


print(h_x.trace())
print(h_zz.trace())
print(h_zz2.trace())

# print("[x, zz]")
# print(m.c(h_x, h_zz))
#
# print("[x, zz2]")
# print(m.c(h_x, h_zz2))
#
# print("[zz2, zz]")
# print(m.c(h_zz2, h_zz))

mat_x = h_x.make_operator().todense().real
mat_zz = h_zz.make_operator().todense().real
mat_zz2 = h_zz2.make_operator().todense().real

print(mat_x)
print(mat_zz)
print(mat_zz2)

print(np.trace(mat_x))
print(np.trace(mat_zz))
print(np.trace(mat_zz2))

# Basis is inhereted from the locop function.


# The object "m" (which comes from running the script) has all sorts
#  of math-y things.
#
# For these things, they have reasonable doc-strings.
# m.from_matrix extracts pauli-string-matrices from a sparse or dense array





# ! ! ! ! ! ! ! ! ! ! ! ! !
#
# I think at some point I may have messed up the Baker-Campbel-Hausdorf expansions
#  ... so be warned. I haven't figured out where it is wrong.
#
# ! ! ! ! ! ! ! ! ! ! ! ! !

x_squared = h_x * h_x
print(x_squared)
print(x_squared.trace())