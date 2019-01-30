#!/bin/bash
#PBS -N ribbon
#PBS -A ngt-630-aa
#PBS -l walltime=12:00:00
#PBS -l nodes=1:gpus=2
#PBS -l feature=k80
#PBS -o $HOME/histo/prediction/drop/20180522_hull/drop_030/out/$(USER)_$(JOBID)_$(JOBNAME).out
#PBS -e $HOME/histo/prediction/drop/20180522_hull/drop_030/out/rpizarro_${MOAB_JOBID}_ribbon.err

source /home/rpizarro/histo/gpu_venv/bin/activate

cd "${PBS_O_WORKDIR}"
# python ribbon.save_CNN_unet.py
python ribbon.test_CNN_unet.n2soft.d30.py
# python ribbon.train_CNN_unet.n2soft.py 
# python ribbon.resize_slices.py
# msub ribbon.HELIOS_GPU.n2soft.sh


