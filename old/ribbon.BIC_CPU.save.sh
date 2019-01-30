#!/bin/sh

export PATH=/usr/local/cuda-8.0/bin:$PATH
export CUDA_HOME=/usr/local/cuda-8.0
export LD_LIBRARY_PATH=/usr/local/cuda-8.0/targets/x86_64-linux/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-8.0/extras/CUPTI/lib64
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/x86_64-linux-gnu/libcudnn
source /data/shmuel/shmuel1/rap/histo/venv_cpu/bin/activate
# pip freeze
cd /data/shmuel/shmuel1/rap/histo/src/
# echo $HOSTNAME > outfile.log

export TF_CPP_MIN_LOG_LEVEL=2
# export PYTHONPATH=${PWD}:$PYTHONPATH
python ribbon.save_CNN_unet.py
# python ribbon.test_CNN_unet.fig.py
