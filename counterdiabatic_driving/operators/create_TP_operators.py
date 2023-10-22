import numpy as np
import scipy.sparse as sp
import sys
import time
import zipfile
import io

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


# find representative of bitstring and how often it is translated
# and whether it is permuted to obtain it
def find_index_TP(bitstring, stringperiod):

    l = len(bitstring)
    idx = index(bitstring)
    idx1 = idx
    count = 0

    for i in range(1, stringperiod):

        idx2 = index(bitstring[l - i:] + bitstring[:l - i])
        count += 1

        if idx2 < idx:
            idx = idx2

    if idx1 < idx:
        count = 0

    # reflected state
    bitstring_rev = bitstring[::-1]
    idx_rev = index(bitstring_rev)
    idx1_rev = idx_rev
    count_rev = 0

    for i in range(1, stringperiod):

        idx_rev2 = index(bitstring_rev[l - i:] + bitstring_rev[:l - i])
        count_rev += 1

        if idx_rev2 < idx_rev:
            idx_rev = idx_rev2

    if idx1_rev < idx_rev:
        count_rev = 0

    if idx_rev < idx:
        idx = idx_rev
        count = count_rev

    return idx, count


# apply a pauli string on an integer, the string is specified by the positions and labels of the non-identity matrices
# return new integer and multiplicative factor
# positions are between 1 and L (from left to right in bits)
def apply_pauli_string(index, positions, labels, size):

    a = 1.
    for i, P in enumerate(labels):
        l = size - positions[i]       # strings are applied from left to right but bits are arranged from right to left

        if P == "x":

            index = index ^ 2 ** l

        elif P == "y":

            val = 1.j * (2. * ((index >> l) % 2) - 1.)
            index = index ^ 2 ** l
            a *= val

        elif P == "z":

            val = 2. * ((index >> l) % 2) - 1.
            index = index
            a *= val

        else:

            raise ValueError("Unknown Pauli Matrix Label " + P + "! Only x, y, z supported.")

    return index, a


def mod_offset(a, m, o):
    floor = np.floor((a - o) / m)
    return (a - m * floor).astype(int)


# apply translation operator T^-1 H T to pauli string
# note that if the states get translated to the right the strings get translated to the left
# hence the -distance
def translate_string(positions, distance, size):

    translated_positions = mod_offset(positions - distance, size, 1)

    return translated_positions


# apply translation operator P^-1 H P to pauli string
def reflect_string(positions, size):

    reflected_positions = size - positions + 1
    return reflected_positions


def parse_opname(opname):

    p = 0
    pos = []
    lab = []

    for label in opname:

        p += 1

        if label == "1":
            continue

        elif label == "x":
            pos.append(p)
            lab.append("x")

        elif label == "y":
            pos.append(p)
            lab.append("y")

        else:
            pos.append(p)
            lab.append("z")

    return np.array(pos), np.array(lab)


# ADD FURTHER PARITY AND MOMENTA

# parameters

# L = 16                       # number of sites
l = 6
k_idx = 0
k_name = "0"
par = 1


for L in np.arange(2, 4, 2):

    start_time = time.time()
    if k_idx >= L or k_idx < 0:
        raise (ValueError, "k index needs to be between 0 and L - 1")

    name_zip = "1d_chain_indices_and_periods.zip"
    with zipfile.ZipFile(name_zip) as zipper:

        name = "1d_chain_TP_indices_and_periods_L=" + str(L) + "_k=" + k_name + ".npz"
        with io.BufferedReader(zipper.open(name, mode='r')) as f:
            data = np.load(f)
            periods = data["period"]
            parity = data["parity"]
            states = data["idx"]
            size = len(periods)

    # op_names = np.loadtxt("operators_TPY_l" + str(l) + ".txt", dtype=str)
    op_names = ["yx", "zx", "zy"]
    for op_name in op_names:
        print(op_name)
        ham_positions, ham_labels = parse_opname(op_name)

        # index and value arrays
        idx_1 = []
        idx_2 = []
        vals = []


        # the hamiltonian is assumed to be translationally invariant and thus a sum of operators in the form
        # H = H_1 + H_2 + ... + H_L = H_1 + T^-1 H_1 T + ... + T^-L H_1 T^L
        # note that this form is assumed, even if the hamiltonian has a period less than L
        # however in this case the computation itself can be simplified accordingly

        # thus it is represented by the first term H_1 = P_1 x P_2 ... x P_L, with Pauli operators P_i = 1, x, y, z
        # the matrix element H_ij can be obtained by applying H_1 to |j> :
        # H_1 |j> = a_1|s_1> (|s_1> is a pure integer and not necessarily a representative, a is a multiplicative factor)

        # if |s> is commensurable with k, one needs to find the representative |j_1> = T^l_1 |s_1>
        # the the matrix element (H_1)_ij = sqrt(N_j / N_i) e^-ikl_1 a, the norms are defined by the period and self_parity (= 0, 1, 2)
        # for the translational invariance N_i = (L^2 / R_i)
        # for the parity this is only changed if p = 1, p = 0 means that the state is self reflected
        # and p = 2 means that it is not self reflected, but its reflection is included in the translated state already
        # the norm multiplication factor is therefore 1 + p mod 2
        #  therefore sqrt(N_j / N_i) = sqrt(R_i / R_j) * sqrt((1 + p_j mod 2) / (1 + p_i mod 2))

        for i, idx in enumerate(states):

            # translations and reflections of the basis hamiltonian
            for n in range(L):

                pos = translate_string(ham_positions, n, L)

                # apply to the state
                idx_ham, factor = apply_pauli_string(idx, pos, ham_labels, L)

                # binary representation
                bin_ham = np.binary_repr(idx_ham, L)

                # period
                p = period(bin_ham)

                # check commensurability with momentum
                ratio = L / p
                commensurable = False

                if k_idx == 0:
                    commensurable = True

                else:
                    for m in range(1, p):
                        if ratio == k_idx / m:
                            commensurable = True

                # check commensurability with parity (ADD LATER - for p = 1 all are commensurate)
                # compute matrix element
                if commensurable:

                    # find representative and number of translations
                    idx_j, l = find_index_TP(bin_ham, p)

                    # find index of representative
                    j = np.searchsorted(states, idx_j)

                    # find factor due to normalizations
                    normprod_i = 1. / (periods[i] * (1. + parity[i] % 2))
                    normprod_j = 1. / (periods[j] * (1. + parity[j] % 2))
                    norm_factor = np.sqrt(normprod_j / normprod_i)

                    # add to lists
                    idx_1.append(i)
                    idx_2.append(j)

                    vals.append(factor * norm_factor * np.exp(-1.j * k_idx * 2. * np.pi * l / L))

                    # reflected
                    pos = reflect_string(pos, L)

                    # apply to the state
                    idx_ham, factor = apply_pauli_string(idx, pos, ham_labels, L)

                    # binary representation
                    bin_ham = np.binary_repr(idx_ham, L)

                    # period
                    p = period(bin_ham)

                    # check commensurability with momentum
                    ratio = L / p
                    commensurable = False

                    if k_idx == 0:
                        commensurable = True

                    else:
                        for m in range(1, p):
                            if ratio == k_idx / m:
                                commensurable = True

                    # check commensurability with parity (ADD LATER - for p = 1 all are commensurate)
                    # compute matrix element
                    if commensurable:
                        # find representative and number of translations
                        idx_j, l = find_index_TP(bin_ham, p)

                        # find index of representative
                        j = np.searchsorted(states, idx_j)

                        # find factor due to normalizations
                        normprod_i = 1. / (periods[i] * (1. + parity[i] % 2))
                        normprod_j = 1. / (periods[j] * (1. + parity[j] % 2))
                        norm_factor = np.sqrt(normprod_j / normprod_i)

                        # add to lists
                        idx_1.append(i)
                        idx_2.append(j)

                        #### check factor of 2 due to parity
                        vals.append(factor * norm_factor * np.exp(-1.j * k_idx * 2. * np.pi * l / L))


        # construct csr matrix, which will automatically sum duplicate matrix elements and construct an index pointer
        # note that we dont use the hermiticity at the moment, but it could further reduce space

        h_mat = sp.csr_matrix((vals, (idx_1, idx_2)))
        vals = None
        idx_1 = None
        idx_2 = None

        mat_name = op_name + "_TP_L=" + str(L) + "_k=" + k_name + "_p=" + str(par) + ".npz"
        size_name = op_name + "_TP_L_k=" + k_name + "_p=" + str(par) + ".txt"

        with open(size_name, "a") as writefile:
            writefile.write(str(L) + " " + str(len(h_mat.data)) + "\n")

        np.savez_compressed(mat_name, indptr=h_mat.indptr, idx=h_mat.indices, val=h_mat.data)

    end_time = time.time()
    print(L, "- Time:", end_time - start_time, flush=True)



