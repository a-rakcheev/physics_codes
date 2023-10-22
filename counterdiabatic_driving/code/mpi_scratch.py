import numpy as np
from mpi4py import MPI

# initialize
comm = MPI.COMM_WORLD

# get rank
rank = comm.Get_rank()

# size
number_of_processes = comm.Get_size()

# problem size
N = 11
step = N // number_of_processes
rest = np.arange(N - number_of_processes * step)

print(step)

l2 = []
for i in range(rank * step, (rank + 1) * step):
    # for j in range(100):
    #     i += j
    l2.append(i)

if rank < len(rest):
    i = number_of_processes * step + rest[rank]
    l2.append(i)

print("process:", rank)

# gather list
l = comm.gather(l2, root=0)

if rank == 0:
    print(l)

