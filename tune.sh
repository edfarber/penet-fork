#!/bin/sh
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --mem=64G
#SBATCH -t 6:00:00
#SBATCH -p gpu --gres=gpu:1
#SBATCH -o /users/edfarber/tune.log

/users/edfarber/miniconda3/envs/penet_new/bin/python scripts/hparam_search.py \
  --data_dir /users/edfarber/scratch/dataset/multimodalpulmonaryembolismdataset/0/ \
  --dataset pe \
  --name penet_search \
  --storage sqlite:///penet_optuna.db \
  --study_name penet_optuna \
  --trials 50 \
  --search_epochs 40 \
  --num_workers 4 \
  --direction maximize