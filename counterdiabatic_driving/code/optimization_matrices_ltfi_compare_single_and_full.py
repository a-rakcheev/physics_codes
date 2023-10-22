import numpy as np

l = 6
L = 11

# full
name = "ltfi_optimization_matrices_L=" + str(L) + "_l=" + str(l) + ".npz"
data = np.load(name)
#
# R_X_ZZ = data["R_X_ZZ"]
# R_X_Z = data["R_X_Z"]
# R_X_X = data["R_X_X"]
# R_Z_X = data["R_Z_X"]
# R_Z_Z = data["R_Z_Z"]
# R_Z_ZZ = data["R_Z_ZZ"]
#
P_X_X = data["P_X_X"]
P_X_ZZ = data["P_X_ZZ"]
P_Z_ZZ = data["P_Z_ZZ"]
P_Z_X = data["P_Z_X"]
P_Z_Z = data["P_Z_Z"]
P_ZZ_ZZ = data["P_ZZ_ZZ"]

# single
# name = "ltfi_optimization_matrices_R_L=" + str(L) + "_l=" + str(l) + ".npz"
# data = np.load(name)
# R_X_ZZ_single = data["R_X_ZZ"]
# R_X_Z_single = data["R_X_Z"]
# R_X_X_single = data["R_X_X"]
# R_Z_X_single = data["R_Z_X"]
# R_Z_Z_single = data["R_Z_Z"]
# R_Z_ZZ_single = data["R_Z_ZZ"]

# name = "ltfi_optimization_matrices_P_Z_ZZ_L=" + str(L) + "_l=" + str(l) + ".npz"
# data = np.load(name)
# P_Z_ZZ_single = data["P_Z_ZZ"]
#
# name = "ltfi_optimization_matrices_P_ZZ_ZZ_L=" + str(L) + "_l=" + str(l) + ".npz"
# data = np.load(name)
# P_ZZ_ZZ_single = data["P_ZZ_ZZ"]
#
# name = "ltfi_optimization_matrices_P_Z_Z_L=" + str(L) + "_l=" + str(l) + ".npz"
# data = np.load(name)
# P_Z_Z_single = data["P_Z_Z"]
#
# name = "ltfi_optimization_matrices_P_Z_X_L=" + str(L) + "_l=" + str(l) + ".npz"
# data = np.load(name)
# P_Z_X_single = data["P_Z_X"]
#
# name = "ltfi_optimization_matrices_P_X_ZZ_L=" + str(L) + "_l=" + str(l) + ".npz"
# data = np.load(name)
# P_X_ZZ_single = data["P_X_ZZ"]
#
# name = "ltfi_optimization_matrices_P_X_X_L=" + str(L) + "_l=" + str(l) + ".npz"
# data = np.load(name)
# P_X_X_single = data["P_X_X"]

# print(np.linalg.norm(R_X_ZZ - R_X_ZZ_single))
# print(np.linalg.norm(R_X_Z - R_X_Z_single))
# print(np.linalg.norm(R_X_X - R_X_X_single))
# print(np.linalg.norm(R_Z_ZZ - R_Z_ZZ_single))
# print(np.linalg.norm(R_Z_Z - R_Z_Z_single))
# print(np.linalg.norm(R_Z_X - R_Z_X_single))

# print(np.linalg.norm(P_X_ZZ - P_X_ZZ_single))
# print(np.linalg.norm(P_X_X - P_X_X_single))
# print(np.linalg.norm(P_Z_ZZ - P_Z_ZZ_single))
# print(np.linalg.norm(P_ZZ_ZZ - P_ZZ_ZZ_single))
# print(np.linalg.norm(P_Z_Z - P_Z_Z_single))
# print(np.linalg.norm(P_Z_X - P_Z_X_single))

# multiprocessing
name = "ltfi_optimization_matrices_P_Z_ZZ_mp_L=" + str(L) + "_l=" + str(l) + ".npz"
data = np.load(name)
P_Z_ZZ_mp = data["P"]

name = "ltfi_optimization_matrices_P_ZZ_ZZ_mp_L=" + str(L) + "_l=" + str(l) + ".npz"
data = np.load(name)
P_ZZ_ZZ_mp = data["P"]

name = "ltfi_optimization_matrices_P_Z_Z_mp_L=" + str(L) + "_l=" + str(l) + ".npz"
data = np.load(name)
P_Z_Z_mp = data["P"]

name = "ltfi_optimization_matrices_P_Z_X_mp_L=" + str(L) + "_l=" + str(l) + ".npz"
data = np.load(name)
P_Z_X_mp = data["P"]

name = "ltfi_optimization_matrices_P_X_ZZ_mp_L=" + str(L) + "_l=" + str(l) + ".npz"
data = np.load(name)
P_X_ZZ_mp = data["P"]

name = "ltfi_optimization_matrices_P_X_X_mp_L=" + str(L) + "_l=" + str(l) + ".npz"
data = np.load(name)
P_X_X_mp = data["P"]


print(np.linalg.norm(P_X_X_mp - P_X_X))
print(np.linalg.norm(P_Z_Z_mp - P_Z_Z))
print(np.linalg.norm(P_ZZ_ZZ_mp - P_ZZ_ZZ))
print(np.linalg.norm(P_X_ZZ_mp - P_X_ZZ))
print(np.linalg.norm(P_Z_ZZ_mp - P_Z_ZZ))
print(np.linalg.norm(P_Z_X_mp - P_Z_X))

# print(np.linalg.norm(P_X_ZZ_mp - P_X_ZZ_single))
# print(np.linalg.norm(P_X_X_mp - P_X_X_single))
# print(np.linalg.norm(P_Z_ZZ_mp - P_Z_ZZ_single))
# print(np.linalg.norm(P_ZZ_ZZ_mp - P_ZZ_ZZ_single))
# print(np.linalg.norm(P_Z_Z_mp - P_Z_Z_single))
# print(np.linalg.norm(P_Z_X_mp - P_Z_X_single))