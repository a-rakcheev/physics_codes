import numpy as np
from commute_stringgroups_no_quspin import *
import time
m = maths()

L = 10

# hamiltonians
h_xz = equation()
for i in range(L):
    op = ''.join(roll(list('xz' + '1' * (L - 2)), i))
    h_xz[op] = 1.0
    op = ''.join(roll(list('zx' + '1' * (L - 2)), i))
    h_xz[op] = 1.0

h_xy = equation()
for i in range(L):
    op = ''.join(roll(list('xy' + '1' * (L - 2)), i))
    h_xy += equation({op: 1.0})
    op = ''.join(roll(list('yx' + '1' * (L - 2)), i))
    h_xy += equation({op: 1.0})


h_comm = m.c(h_xz, h_xy)
print(h_comm)
print("Commute:")
start = time.time()
for i in range(10000):
    h_comm = m.c(h_xz, h_xy)
end = time.time()
print("Time:", end - start)

prod = h_xz * h_xz
print(prod.trace())
print("Trace:")
start = time.time()
for i in range(10000):
    tr = prod.trace()
end = time.time()
print("Time:", end - start)
