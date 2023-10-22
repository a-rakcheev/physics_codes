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


def selfparity(bitstring):

    p = 1
    if bitstring[::-1] == bitstring:
        p = 0

    return p


def find_index(bitstring, stringperiod):

    l = len(bitstring)
    idx = index(bitstring)
    for i in range(1, stringperiod):
        idx2 = index(bitstring[l - i:] + bitstring[:l - i])

        if idx2 < idx:
            idx = idx2

    return idx


# parameters

L = int(sys.argv[1])                       # number of sites
k_idx = int(sys.argv[2])
k_name = str(sys.argv[3])

if k_idx >= L or k_idx < 0:
    raise(ValueError, "k index needs to be between 0 and L - 1")

name = "1d_chain_T_indices_and_periods_L=" + str(L) + "_k=" + k_name + ".npz"
data = np.load(name)
states = data["idx"]
periods = data["period"]


# lists to be saved
state_rep = []              # state representatives (their index will be their position in the list)
periods_new = []
self_parities = []                 # periods of the representatives upon translation (T ^ period |state> = |state>)


# states are obtained as follows:

# loop through all i in the list for the momentum
# check if the state is self-reflected and add to representatives list and parity list
# if not self-reflected find the representative of the reflected state and remove it
# for this purpose one can use a sorted list with fast remove and access times

slist.update(states)
states = None

while len(slist) > 0:

    i = slist.pop(0)
    state_rep.append(i)
    # print(i, len(slist))

    # binary representation
    bin_i = np.binary_repr(i, L)

    # find period
    p = period(bin_i)
    periods_new.append(p)

    # find parity
    P = selfparity(bin_i)

    # find and remove reflected state if it exists
    if P == 1:

        bin_j = bin_i[::-1]
        idx_j = find_index(bin_j, p)
        if idx_j == i:

            P = 2

        else:
            slist.remove(idx_j)

    self_parities.append(P)

size = len(state_rep)

# print(state_rep)
# print(periods_new)
# print(self_parities)

idx_name = "1d_chain_TP_indices_and_periods_L=" + str(L) + "_k=" + k_name + ".npz"
size_name = "1d_chain_TP_size_k=" + k_name + ".txt"

with open(size_name, "a") as writefile:

    writefile.write(str(L) + " " + str(size) + "\n")

np.savez_compressed(idx_name, idx=state_rep, period=periods_new, parity=self_parities)
