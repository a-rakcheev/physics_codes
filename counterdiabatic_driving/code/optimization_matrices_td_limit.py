import numpy as np

l = 6
Ll = np.arange(9, 13, 1)
num_op = np.loadtxt("operators/operators_TPY_size_full.txt").astype(int)
size = num_op[l - 1]
print("size:", size)

R_X_ZZ_mat = np.zeros((len(Ll), size))
R_X_Z_mat = np.zeros((len(Ll), size))
R_X_X_mat = np.zeros((len(Ll), size))
R_Z_X_mat = np.zeros((len(Ll), size))
R_Z_Z_mat = np.zeros((len(Ll), size))
R_Z_ZZ_mat = np.zeros((len(Ll), size))

P_X_X_mat = np.zeros((len(Ll), size * (size + 1) // 2))
P_X_ZZ_mat = np.zeros((len(Ll), size, size))
P_Z_ZZ_mat = np.zeros((len(Ll), size, size))
P_Z_X_mat = np.zeros((len(Ll), size, size))
P_Z_Z_mat = np.zeros((len(Ll), size * (size + 1) // 2))
P_ZZ_ZZ_mat = np.zeros((len(Ll), size * (size + 1) // 2))


for i, L in enumerate(Ll):

    name = "ltfi_optimization_matrices_L=" + str(L) + "_l=" + str(l) + ".npz"
    data = np.load(name)

    R_X_ZZ = data["R_X_ZZ"]
    R_X_Z = data["R_X_Z"]
    R_X_X = data["R_X_X"]
    R_Z_X = data["R_Z_X"]
    R_Z_Z = data["R_Z_Z"]
    R_Z_ZZ = data["R_Z_ZZ"]

    P_X_X = data["P_X_X"]
    P_X_ZZ = data["P_X_ZZ"]
    P_Z_ZZ = data["P_Z_ZZ"]
    P_Z_X = data["P_Z_X"]
    P_Z_Z = data["P_Z_Z"]
    P_ZZ_ZZ = data["P_ZZ_ZZ"]

    R_X_ZZ_mat[i, :] = R_X_ZZ
    R_X_Z_mat[i, :] = R_X_Z
    R_X_X_mat[i, :] = R_X_X
    R_Z_X_mat[i, :] = R_Z_X
    R_Z_Z_mat[i, :] = R_Z_Z
    R_Z_ZZ_mat[i, :] = R_Z_ZZ

    P_X_X_mat[i, :] = P_X_X
    P_X_ZZ_mat[i, :, :] = P_X_ZZ
    P_Z_ZZ_mat[i, :, :] = P_Z_ZZ
    P_Z_X_mat[i, :, :] = P_Z_X
    P_Z_Z_mat[i, :] = P_Z_Z
    P_ZZ_ZZ_mat[i, :] = P_ZZ_ZZ

for i in range(len(Ll) - 1):

    print("System Sizes:", Ll[i], Ll[i+1])
    R_X_ZZ = R_X_ZZ_mat[i, :]
    R_X_Z = R_X_Z_mat[i, :]
    R_X_X = R_X_X_mat[i, :]
    R_Z_X = R_Z_X_mat[i, :]
    R_Z_Z = R_Z_Z_mat[i, :]
    R_Z_ZZ = R_Z_ZZ_mat[i, :]

    P_X_X = P_X_X_mat[i, :]
    P_X_ZZ = P_X_ZZ_mat[i, :, :]
    P_Z_ZZ = P_Z_ZZ_mat[i, :, :]
    P_Z_X = P_Z_X_mat[i, :, :]
    P_Z_Z = P_Z_Z_mat[i, :]
    P_ZZ_ZZ = P_ZZ_ZZ_mat[i, :]

    R_X_ZZ_2 = R_X_ZZ_mat[i + 1, :]
    R_X_Z_2 = R_X_Z_mat[i + 1, :]
    R_X_X_2 = R_X_X_mat[i + 1, :]
    R_Z_X_2 = R_Z_X_mat[i + 1, :]
    R_Z_Z_2 = R_Z_Z_mat[i + 1, :]
    R_Z_ZZ_2 = R_Z_ZZ_mat[i + 1, :]

    P_X_X_2 = P_X_X_mat[i + 1, :]
    P_X_ZZ_2 = P_X_ZZ_mat[i + 1, :, :]
    P_Z_ZZ_2 = P_Z_ZZ_mat[i + 1, :, :]
    P_Z_X_2 = P_Z_X_mat[i + 1, :, :]
    P_Z_Z_2 = P_Z_Z_mat[i + 1, :]
    P_ZZ_ZZ_2 = P_ZZ_ZZ_mat[i + 1, :]

    print("Difference R_X_ZZ:", np.linalg.norm(R_X_ZZ - R_X_ZZ_2))
    print("Difference R_X_X:", np.linalg.norm(R_X_X - R_X_X_2))
    print("Difference R_X_Z:", np.linalg.norm(R_X_Z - R_X_Z_2))
    print("Difference R_Z_X:", np.linalg.norm(R_Z_X - R_Z_X_2))
    print("Difference R_Z_Z:", np.linalg.norm(R_Z_Z - R_Z_Z_2))
    print("Difference R_Z_ZZ:", np.linalg.norm(R_Z_ZZ - R_Z_ZZ_2))

    print("Difference P_X_X:", np.linalg.norm(P_X_X - P_X_X_2))
    print("Difference P_Z_Z:", np.linalg.norm(P_Z_Z - P_Z_Z_2))
    print("Difference P_ZZ_ZZ:", np.linalg.norm(P_ZZ_ZZ - P_ZZ_ZZ_2))
    print("Difference P_X_ZZ:", np.linalg.norm(P_X_ZZ - P_X_ZZ_2))
    print("Difference P_Z_X:", np.linalg.norm(P_Z_X - P_Z_X_2))
    print("Difference P_Z_ZZ:", np.linalg.norm(P_Z_ZZ - P_Z_ZZ_2))

