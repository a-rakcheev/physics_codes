import numpy as np
from commute_stringgroups_no_quspin import *
m = maths()

L = 12
l = 3

name = "optimization_matrices_ltfi_L=" + str(L) + "_l=" + str(l) + ".npz"

data = np.load(name)
R_X_ZZ = data["R_X_ZZ"]
R_X_Z = data["R_X_Z"]
R_X_X = data["R_X_X"]
R_Z_Z = data["R_Z_Z"]
R_Z_ZZ = data["R_Z_ZZ"]
R_Z_X = data["R_Z_X"]

P_X_X = data["P_X_X"]
P_X_ZZ = data["P_X_ZZ"]
P_Z_ZZ = data["P_Z_ZZ"]
P_Z_X = data["P_Z_X"]
P_Z_Z = data["P_Z_Z"]
P_ZZ_ZZ = data["P_ZZ_ZZ"]

var_order = len(R_Z_Z)


