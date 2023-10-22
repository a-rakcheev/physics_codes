# compute the reflection operator and save as  numpy arrays
# one only needs the column indices
import numpy as np

for L in np.arange(1, 11, 1):
    size = 2 ** L

    idx = np.zeros(size, dtype=np.int32)
    for i in range(size):

        j = int(np.binary_repr(i, L)[::-1], 2)
        idx[i] = j


    name = "reflection_operator_indices_L=" + str(L) + ".npy"
    np.save(name, idx)






