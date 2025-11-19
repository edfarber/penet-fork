#!/bin/sh
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --mem=64G
#SBATCH -t 4:00:00
#SBATCH -p gpu --gres=gpu:1
#SBATCH -o /users/edfarber/test.log

python test.py --data_dir /users/edfarber/scratch/dataset/multimodalpulmonaryembolismdataset/0/ \
                 --ckpt_path /users/edfarber/train_logs/PositiveWeighting_20251113_132622/best.pth.tar \
                 --results_dir results \
                 --phase test \
                 --name test2 \
                 --dataset pe \
                 --gpu_ids 0