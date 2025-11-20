#!/bin/sh
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --mem=64G
#SBATCH -t 4:00:00
#SBATCH -p gpu --gres=gpu:1
#SBATCH -o /users/edfarber/train.log

/users/edfarber/scratch/penet-fork/.venv/bin/python train.py --data_dir=/users/edfarber/scratch/dataset/multimodalpulmonaryembolismdataset/0/ \
                --ckpt_path=/users/edfarber/scratch/penet/xnet_kin_90.pth.tar \
                --save_dir=/users/edfarber/train_logs \
		--name=Test3 \
		--abnormal_prob=0.5 \
                --agg_method=max \
                --batch_size=16 \
                --best_ckpt_metric=val_AUROC \
                --crop_shape=192,192 \
                --cudnn_benchmark=True \
                --dataset=pe \
                --do_classify=True \
                --epochs_per_eval=1 \
                --epochs_per_save=1 \
                --fine_tune=True \
                --fine_tuning_boundary=classifier \
                --fine_tuning_lr=1e-2 \
                --gpu_ids=0 \
                --include_normals=True \
                --iters_per_print=32 \
                --iters_per_visual=8000 \
                --learning_rate=5e-3 \
                --lr_decay_step=600000 \
                --lr_scheduler=cosine_warmup \
                --lr_warmup_steps=10000 \
                --model=PENetClassifier \
                --model_depth=50 \
                --num_classes=1 \
                --num_epochs=1 \
                --num_slices=24 \
                --num_visuals=8 \
                --num_workers=8 \
                --optimizer=adam \
                --pe_types='["central", "segmental"]' \
                --resize_shape=208,208 \
                --sgd_dampening=0.9 \
                --sgd_momentum=0.9 \
                --use_hem=False \
                --use_pretrained=True \
                --weight_decay=1e-3 \
                --use_amp=True