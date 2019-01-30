#!/bin/bash
#PBS -N ribbon
#PBS -A ngt-630-aa
#PBS -l walltime=12:00:00
#PBS -l nodes=1:gpus=4
#PBS -l feature=k80
#PBS -o $HOME/histo/weights/nb-gpus/nd1_gpu04/out/$(USER)_$(JOBID)_$(JOBNAME).out
#PBS -e $HOME/histo/weights/nb-gpus/nd1_gpu04/out/rpizarro_${MOAB_JOBID}_ribbon.err

source /home/rpizarro/histo/gpu_venv/bin/activate

cd "${PBS_O_WORKDIR}"
# python ribbon.save_CNN_unet.py
python ribbon.train_CNN_unet.spatial.drop00.decay000.model500ep.py

# msub ribbon.HELIOS_GPU.spatial.drop00.decay000.model500ep.sh

