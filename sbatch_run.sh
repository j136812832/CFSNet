#!/bin/bash
#MV2_USE_CUDA=1 MV2_ENABLE_AFFINITY=0 MV2_SMP_USE_CMA=0
#SBATCH -p pixel
#SBATCH --gres=gpu:1
#SBATCH --nodelist=SH-IDC1-10-198-6-84
#SBATCH --job-name=taskz
#SBATCH -o /mnt/lustre/zhangyijie/wenjian/GFRNet-pytorch-beta-working_refactor/logs/taskz/%j.txt
srun --mpi=pmi2 python train.py --train_config ./script/yaml/taskz.yaml  --num_gpus 1 --log_path ./log_path/taskz/ --tensorboard_path log_pt/taskz/ --snapshot_path ./snapshot/taskz  --to_YUV
echo "Submit the taskz job by run \'sbatch\'" 
