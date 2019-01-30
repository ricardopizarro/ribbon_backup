#!/bin/sh

mv ~/ribbon.BIC_* ~/log_ribn
# qsub -q all.q -l h_vmem=50G ribbon.BIC_GPU.test.sh
# qsub -q gpu.q -l h_vmem=200G ribbon.BIC_GPU.test.sh
qsub -q gpu.q -l h_vmem=200G ribbon.BIC_GPU.sh
