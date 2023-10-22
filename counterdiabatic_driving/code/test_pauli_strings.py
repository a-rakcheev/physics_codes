import numpy as np
import time
from commute_stringgroups_no_quspin import *
m = maths()

L = 2
# hamiltonians
h_x = equation()
for i in range(L):
    op = ''.join(roll(list('x' + '1' * (L - 1)), i))
    h_x[op] = 0.5

h_y = equation()
for i in range(L):
    op = ''.join(roll(list('y' + '1' * (L - 1)), i))
    h_y[op] = 0.5

h_z = equation()
for i in range(L):
    op = ''.join(roll(list('z' + '1' * (L - 1)), i))
    h_z[op] = 0.5


h_xx = equation()
for i in range(L):
    op = ''.join(roll(list('xx' + '1' * (L - 2)), i))
    h_xx += equation({op: 0.25})

h_yy = equation()
for i in range(L):
    op = ''.join(roll(list('yy' + '1' * (L - 2)), i))
    h_yy += equation({op: 0.25})

h_zz = equation()
for i in range(L):
    op = ''.join(roll(list('zz' + '1' * (L - 2)), i))
    h_zz += equation({op: 0.25})

h_xy = equation()
for i in range(L):
    op = ''.join(roll(list('xy' + '1' * (L - 2)), i))
    h_xy[op] = 0.25

h_yx = equation()
for i in range(L):
    op = ''.join(roll(list('yx' + '1' * (L - 2)), i))
    h_yx[op] = 0.25


# print(h_zz)
# print(h_xy)
# print(h_yx)

s_1x = equation()
s_1x["1x"] = 0.5

s_xy = equation()
s_xy["xy"] = 0.25

print(m.c(s_1x, s_xy))
start_time = time.time()
for i in range(100000):
    m.c(s_1x, s_xy)
end_time = time.time()
print(end_time - start_time)


print(m.c(h_x, h_xy))
start_time = time.time()
for i in range(100000):
    m.c(h_x, h_xy)
end_time = time.time()
print(end_time - start_time)


# print(m.c(h_xy, h_z))