#!/bin/bash
#PBS -N ribbon
#PBS -A ngt-630-aa
#PBS -l walltime=8:00:00
#PBS -l nodes=1:gpus=2
#PBS -l feature=k80
#PBS -o $HOME/histo/data/rm311_128requad_test_hull/out/$(USER)_$(JOBID)_$(JOBNAME).out
#PBS -e $HOME/histo/data/rm311_128requad_test_hull/out/rpizarro_${MOAB_JOBID}_ribbon.err

source /home/rpizarro/histo/gpu_venv/bin/activate

cd "${PBS_O_WORKDIR}"
python ribbon.test_convex_hull.py
