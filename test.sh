#!/bin/sh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 4:00:00
#SBATCH -p gpu --gres=gpu:1
#SBATCH -o /users/edfarber/test.log

# Path to your python environment
PYTHON_EXEC=/users/edfarber/scratch/penet-fork/.venv/bin/python

# Path to your data
DATA_DIR=/users/edfarber/scratch/dataset/multimodalpulmonaryembolismdataset/0/

# Path to the checkpoint you want to test
# IMPORTANT: Update this to point to the specific checkpoint you want to evaluate!
CKPT_PATH=/users/edfarber/train_logs/Test3_BestParams_20251120_021646/best.pth.tar

# Run the test using the config wrapper to ensure correct preprocessing
$PYTHON_EXEC scripts/run_test_with_config.py train_configs.json \
    --ckpt_path $CKPT_PATH \
    --data_dir $DATA_DIR \
    --phase test