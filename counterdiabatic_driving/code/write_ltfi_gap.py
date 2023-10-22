# write script to submit jobs and scripts for the jobs, each job passes
# variables to a python program

import numpy as np

# parameters
Ll = [16, 18, 20]
processes = 16

# file names
qsub = "qsub_gap.sh"
name_of_job = "gap_"

for idx, L in enumerate(Ll):
    with open("script_gap_" + str(idx) + ".sh", "w") as scriptfile:

        scriptfile.write("#!/bin/bash \n")
        scriptfile.write("\n")
        scriptfile.write("#$ -q std.q \n")
        scriptfile.write("#$ -N " + name_of_job + str(idx) + "\n")
        scriptfile.write("#$ -o " + name_of_job + str(idx) + ".out \n")
        scriptfile.write("#$ -M Artem.Rakcheev@uibk.ac.at \n")
        scriptfile.write("#$ -m ea \n")
        scriptfile.write("#$ -j yes \n")
        scriptfile.write("#$ -wd /scratch/c7051098/agp \n")
        scriptfile.write("#$ -l h_vmem=1G \n")
        scriptfile.write("#$ -pe openmp " + str(processes) + "\n")
        scriptfile.write("\n")
        scriptfile.write("export OMP_NUM_THREADS=$NSLOTS \n")
        scriptfile.write("$CONDA_PREFIX/bin/python ")
        scriptfile.write("ltfi_gap.py " + str(L))

    with open(qsub, "a") as jobfile:

        jobfile.write("qsub script_gap_" + str(idx) + ".sh \n")