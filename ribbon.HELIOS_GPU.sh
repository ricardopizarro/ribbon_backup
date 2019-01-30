#!/bin/bash
#PBS -N ribbon
#PBS -A ngt-630-aa
#PBS -l walltime=12:00:00
#PBS -l nodes=1:gpus=2
#PBS -l feature=k80
#PBS -o $HOME/histo/prediction/spatial_weight/drop_030/decay_000/set198/resize/out/$(USER)_$(JOBID)_$(JOBNAME).out
#PBS -e $HOME/histo/prediction/spatial_weight/drop_030/decay_000/set198/resize/out/rpizarro_${MOAB_JOBID}_ribbon.err

source /home/rpizarro/histo/gpu_venv/bin/activate

cd "${PBS_O_WORKDIR}"
# python ribbon.save_CNN_unet.py
# python ribbon.train_CNN_unet.py 
# python ribbon.test_convex_hull.py

python ribbon.resize_slices.py
python ribbon.quad_to_png.py
python ribbon.png_to_pdf.py

# msub ribbon.HELIOS_GPU.n2soft.sh


