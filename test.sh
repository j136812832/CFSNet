#!/bin/bash
#MV2_USE-CUDA=1 MV2_ENABLE_AFFINITY=0 MV2_SMP_USE_CMA=0
#SBATCH -p pixel
#SBATCH --gres=gpu:1
#SBATCH --nodelist=SH-IDC1-10-198-6-84
#SBATCH --cpus-per-task=8
#SBATCH --job-name=face1
#SBATCH -o /mnt/lustre/zhangyijie/wenjian/CFSNet-master/output.txt
srun --mpi=pmi2 python train.py -opt ./settings/train/train_denoise.json
echo "Summit the face1 by run \'sbatch\'"
