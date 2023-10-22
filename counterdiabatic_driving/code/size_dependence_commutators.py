import numpy as np
from commute_stringgroups_v2 import *
m = maths()

# parameters
# N = 4

for N in [4, 6, 8]:

    # hamiltonians
    h_x = equation()
    for i in range(N):
        op = ''.join(roll(list('x' + '1' * (N - 1)), i))
        h_x[op] = 0.5

    h_z = equation()
    for i in range(N):
        op = ''.join(roll(list('z' + '1' * (N - 1)), i))
        h_z[op] = 0.5

    h_zz = equation()
    for i in range(N):
        op = ''.join(roll(list('zz' + '1' * (N - 2)), i))
        h_zz += equation({op: 0.25})


    o_xyz = equation()
    for i in range(N):
        op = ''.join(roll(list('xyz' + '1' * (N - 3)), i))
        o_xyz += equation({op: 0.25})

        op = ''.join(roll(list('zyx' + '1' * (N - 3)), i))
        o_xyz += equation({op: 0.25})


    o_xyz = equation()
    for i in range(N):
        op = ''.join(roll(list('xyz' + '1' * (N - 3)), i))
        o_xyz += equation({op: 0.25})

        op = ''.join(roll(list('zyx' + '1' * (N - 3)), i))
        o_xyz += equation({op: 0.25})

    ham = 0.3 * h_z + 0.7 * h_x + h_zz
    ham_deriv_x = -1 * h_x
    ham_deriv_y = -1 * h_z
    # print(ham)
    # print(o_xyz)
    print(m.c(ham, o_xyz))

