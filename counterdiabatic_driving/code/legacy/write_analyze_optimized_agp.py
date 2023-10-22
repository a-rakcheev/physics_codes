# write script to submit jobs and scripts for the jobs, each job passes
# variables to a python program

import numpy as np

# parameters

L = 4
res_x = 50
res_y = 50
order = 15
processes = 8
l = 4
rl = [1, 2, 3, 4]

# file names

qsub = "qsub_analyzes_agp.sh"
name_of_job = "analyze_agp_"

for idx, r in enumerate(rl):
    with open("script_analyze_agp_" + str(idx) + ".sh", "w") as scriptfile:

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
        scriptfile.write("export OMP_NUM_THREADS=1 \n")
        scriptfile.write("$CONDA_PREFIX/envs/Py27/bin/python ")
        scriptfile.write("process_analyze_agp.py " + str(L) + " " + str(res_x) + " " + str(res_y)
                         + " " + str(order) + " " + str(processes) + " " + str(l) + " " + str(r))

    with open(qsub, "a") as jobfile:

        jobfile.write("qsub script_analyze_agp_" + str(idx) + ".sh \n")