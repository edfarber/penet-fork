#!/bin/sh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 4:00:00
#SBATCH -p gpu --gres=gpu:1
#SBATCH -o /users/edfarber/train.log

/users/edfarber/scratch/penet-fork/.venv/bin/python scripts/run_args.py train_configs.json