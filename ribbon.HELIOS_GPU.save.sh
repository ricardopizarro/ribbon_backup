#!/bin/bash
#PBS -N ribbon
#PBS -A ngt-630-aa
#PBS -l walltime=1:00:00
#PBS -l nodes=1:gpus=2
#PBS -o $HOME/histo/out/$(USER)_$(JOBID)_$(JOBNAME).out
#PBS -e $HOME/histo/out/rpizarro_${MOAB_JOBID}_ribbon.err

source /home/rpizarro/histo/gpu_venv/bin/activate

cd "${PBS_O_WORKDIR}"
# python ribbon.save_CNN_unet.py
python ribbon.save_CNN_unet.py 


