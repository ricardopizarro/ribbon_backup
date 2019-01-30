#!/bin/bash
#PBS -N ribbon
#PBS -A ngt-630-aa
#PBS -l walltime=12:00:00
#PBS -l nodes=1:gpus=2
#PBS -l feature=k80
#PBS -o $HOME/histo/data/out/$(USER)_$(JOBID)_$(JOBNAME).out
#PBS -e $HOME/histo/data/out/rpizarro_${MOAB_JOBID}_ribbon.err

source /home/rpizarro/histo/gpu_venv/bin/activate

cd "${PBS_O_WORKDIR}"

python ribbon.retile_quad.102.py

# msub ribbon.HELIOS_GPU.n2soft.sh


