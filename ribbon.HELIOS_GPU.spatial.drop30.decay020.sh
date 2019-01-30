#!/bin/bash
#PBS -N ribbon
#PBS -A ngt-630-aa
#PBS -l walltime=5:00:00
#PBS -l nodes=1:gpus=2
#PBS -l feature=k80
#PBS -o $HOME/histo/weights/weights.spatial.weight/drop_030/decay_020/out/$(USER)_$(JOBID)_$(JOBNAME).out
#PBS -e $HOME/histo/weights/weights.spatial.weight/drop_030/decay_020/out/rpizarro_${MOAB_JOBID}_ribbon.err

source /home/rpizarro/histo/gpu_venv/bin/activate

cd "${PBS_O_WORKDIR}"
# python ribbon.save_CNN_unet.py
python ribbon.train_CNN_unet.spatial_weight.drop30.decay020.py

# msub ribbon.HELIOS_GPU.spatial.drop30.decay020.sh
# msub ribbon.HELIOS_GPU.n2soft.d30.sh


