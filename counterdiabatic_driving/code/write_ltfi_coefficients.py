# write script to submit jobs and scripts for the jobs, each job passes
# variables to a python program

import numpy as np

# parameters
l = 8
L = 2 * l - 1
res_h = 2
res_g = 2
threads = 16

# file names
qsub = "qsub_ltfi_coeff.sh"
name_of_job = "ltfi_coeff_"


namestring = str(l)
with open("script_opt_" + str(l) + ".sh", "w") as scriptfile:

    scriptfile.write("#!/bin/bash \n")
    scriptfile.write("\n")
    scriptfile.write("#$ -q std.q \n")
    scriptfile.write("#$ -N " + name_of_job + namestring + "\n")
    scriptfile.write("#$ -o " + name_of_job + namestring + ".out \n")
    scriptfile.write("#$ -M Artem.Rakcheev@uibk.ac.at \n")
    scriptfile.write("#$ -m ea \n")
    scriptfile.write("#$ -j yes \n")
    scriptfile.write("#$ -wd /scratch/c7051098/agp \n")
    scriptfile.write("#$ -l h_vmem=500M \n")
    scriptfile.write("#$ -pe openmp " + str(threads) + "\n")
    scriptfile.write("\n")
    scriptfile.write("export OMP_NUM_THREADS=$NSLOTS \n")
    scriptfile.write("$CONDA_PREFIX/bin/python ")
    scriptfile.write("ltfi_optimal_coefficients_from_matrices.py " + str(l) + " " + str(L) + " " + str(res_h) + " " + str(res_g))

with open(qsub, "a") as jobfile:

    jobfile.write("qsub script_opt_" + str(l) + ".sh \n")