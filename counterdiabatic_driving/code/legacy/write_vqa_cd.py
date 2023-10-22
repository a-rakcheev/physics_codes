# write script to submit jobs and scripts for the jobs, each job passes
# variables to a python program

import numpy as np

# parameters

N = 4
order = 20
ls = np.arange(2, N + 1, 1)
res = 100
T = 1.0
s = 0.99

# file names

qsub = "qsub_vqa_cd.sh"
name_of_job = "vqa_cd_"

for idx, l in enumerate(ls):

    namestring = str(idx)

    with open("script_vqa_cd_" + str(idx) + ".sh", "w") as scriptfile:

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
        scriptfile.write("#$ -pe openmp 8 \n")
        scriptfile.write("\n")
        scriptfile.write("export OMP_NUM_THREADS=$NSLOTS \n")
        scriptfile.write("$CONDA_PREFIX/bin/python \n")
        scriptfile.write("python vqa_cd_driving.py " + str(N) + " " + str(order) + " " + str(l)
                         + " " + str(res) + " " + str(T) + " " + str(s))

    with open(qsub, "a") as jobfile:

        jobfile.write("qsub script_vqa_cd_" + str(idx) + ".sh \n")