# write script to submit jobs and scripts for the jobs, each job passes
# variables to a python program

import numpy as np

# parameters
l = 8
L = 15
processes = [32, 32, 32, 32, 32, 32]

# file names
qsub = "qsub_ltfi_opt.sh"
name_of_job = "ltfi_opt"
jobs = ["_P_X_ZZ", "_P_X_Z", "_P_Z_ZZ", "_P_X_X", "_P_Z_Z", "_P_ZZ_ZZ"]

for idx, job in enumerate(jobs):

    namestring = "_L" + str(L) + "_l" + str(l) + job

    with open("script_opt_" + str(idx) + ".sh", "w") as scriptfile:

        scriptfile.write("#!/bin/bash \n")
        scriptfile.write("\n")
        scriptfile.write("#$ -q std.q \n")
        scriptfile.write("#$ -N " + name_of_job + namestring + "\n")
        scriptfile.write("#$ -o " + name_of_job + namestring + ".out \n")
        scriptfile.write("#$ -M Artem.Rakcheev@uibk.ac.at \n")
        scriptfile.write("#$ -m ea \n")
        scriptfile.write("#$ -j yes \n")
        scriptfile.write("#$ -wd /scratch/c7051098/agp \n")
        scriptfile.write("#$ -l h_vmem=1G \n")
        scriptfile.write("#$ -pe openmp " + str(processes[idx]) + "\n")
        scriptfile.write("\n")
        scriptfile.write("export OMP_NUM_THREADS=1 \n")
        scriptfile.write("$CONDA_PREFIX/bin/python ")
        scriptfile.write("optimization_matrices_ltfi_multiprocessing" + job + ".py "
                         + str(l) + " " + str(L) + " " + str(processes[idx]))

    with open(qsub, "a") as jobfile:

        jobfile.write("qsub script_opt_" + str(idx) + ".sh \n")