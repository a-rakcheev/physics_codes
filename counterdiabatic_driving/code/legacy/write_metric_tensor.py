# write script to submit jobs and scripts for the jobs, each job passes
# variables to a python program

import numpy as np

# parameters

L = 8
res_x = 50
res_y = 50
order = 15 
ll = np.arange(2, 6, 1)
processes = 32

# file names

qsub = "qsub_metric.sh"
name_of_job = "metric_"

for idx, l in enumerate(ll):

    namestring = str(idx)

    with open("script_metric_" + str(idx) + ".sh", "w") as scriptfile:

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
        scriptfile.write("#$ -pe openmp " + str(processes) + "\n")
        scriptfile.write("\n")
        scriptfile.write("export OMP_NUM_THREADS=1 \n")
        scriptfile.write("$CONDA_PREFIX/envs/Py27/bin/python ")
        scriptfile.write("optimize_agp_ltfi.py " + str(L) + " " + str(res_x) + " " + str(res_y)
                         + " " + str(order) + " " + str(processes) + " " + str(l))

    with open(qsub, "a") as jobfile:

        jobfile.write("qsub script_metric_" + str(idx) + ".sh \n")
