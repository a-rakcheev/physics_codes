# compute state representatives and their index for translational invariance in 1d for a given momentum k

import numpy as np
from sortedcontainers import sortedlist
import sys
slist = sortedlist.SortedList()


def period(bitstring):

    l = len(bitstring)
    p = 1
    for i in range(1, l):
        bitstring_shifted = bitstring[l - i:] + bitstring[:l - i]
        if bitstring_shifted == bitstring:
            break
        p += 1

    return p


def index(bitstring):

    idx = 0
    for i, b in enumerate(bitstring[::-1]):

        idx += int(b) * (2 ** i)

    return idx


# parameters
L = int(sys.argv[1])                       # number of sites
k_idx = int(sys.argv[2])
k_name = str(sys.argv[3])
# k_idx = L // 2                   # momentum index (full momentum is (2 pi / L) * index), should be 0, 1, ..., L - 1
# k_name = "pi"

if k_idx >= L or k_idx < 0:
    raise(ValueError, "k index needs to be between 0 and L - 1")

# lists to be saved
state_rep = []              # state representatives (their index will be their position in the list)
periods = []                 # periods of the representatives upon translation (T ^ period |state> = |state>)


# states are obtained as follows:

# loop through all i in 2 ^ L, for each i obtain the period and check if it is commensurable with the momentum
# if yes include in list of representative / periods
# either way remove all possible translated states from the index list (over which one loops)
# for this purpose one can use a sorted list with fast remove and access times

slist.update(range(2 ** L))
pow2 = 2 ** np.arange(L - 1, -1, -1)

while len(slist) > 0:

    i = slist[0]

    # binary representation
    bin_i = np.binary_repr(i, L)

    # find period
    p = period(bin_i)

    # check commensurability with momentum
    ratio = L / p
    commensurable = False

    if k_idx == 0:
        commensurable = True

    else:
        for m in range(1, p):
            if ratio == k_idx / m:
                commensurable = True

    if commensurable:

        state_rep.append(i)
        periods.append(p)

    # remove all states with this representative

    del slist[0]
    for j in range(1, p):
        bin_j = bin_i[L - j:] + bin_i[:L - j]
        idx_j = index(bin_j)
        slist.remove(idx_j)

size = len(state_rep)

# idx_name = "1d_chain_T_indices_and_periods_L=" + str(L) + "_k=" + k_name + ".npz"
# size_name = "1d_chain_T_size_k=" + k_name + ".txt"
#
# with open(size_name, "a") as writefile:
#
#
#     writefile.write(str(L) + " " + str(size) + "\n")
#
# np.savez_compressed(idx_name, idx=state_rep, period=periods)
