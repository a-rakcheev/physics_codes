# write script to submit jobs and scripts for the jobs, each job passes
# variables to a python program

import numpy as np

# parameters
l = 5
Ll = [8]
threads = 4

# file names

qsub = "qsub_ltfi_opt.sh"
name_of_job = "ltfi_opt_"

for idx, L in enumerate(Ll):

    namestring = str(idx)

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
        scriptfile.write("#$ -l h_vmem=500M \n")
        scriptfile.write("#$ -pe openmp " + str(threads) + "\n")
        scriptfile.write("\n")
        scriptfile.write("export OMP_NUM_THREADS=$NSLOTS \n")
        scriptfile.write("$CONDA_PREFIX/bin/python ")
        scriptfile.write("optimization_matrices_ltfi_cluster.py " + str(l) + " " + str(L) + " " + str(threads))

    with open(qsub, "a") as jobfile:

        jobfile.write("qsub script_opt_" + str(idx) + ".sh \n")
