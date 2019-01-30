#!/bin/bash
#PBS -N ribbon
#PBS -A ngt-630-aa
#PBS -l walltime=12:00:00
#PBS -l nodes=1:gpus=2
#PBS -l feature=k80
#PBS -o $HOME/histo/weights/weights.spatial.weight/drop_000/decay_000_500ep/out/$(USER)_$(JOBID)_$(JOBNAME).out
#PBS -e $HOME/histo/weights/weights.spatial.weight/drop_000/decay_000_500ep/out/rpizarro_${MOAB_JOBID}_ribbon.err

source /home/rpizarro/histo/gpu_venv/bin/activate

cd "${PBS_O_WORKDIR}"
# python ribbon.save_CNN_unet.py
python ribbon.train_CNN_unet.spatial_weight.drop00.decay000_500ep.py

# msub ribbon.HELIOS_GPU.spatial.drop00.decay000_500ep.sh

