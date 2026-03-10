#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

mkdir -p /home/adminster/DYH/eatdata/results_20percent

python /home/adminster/DYH/fairseq/fairseq_cli/hydra_train.py -m \
    --config-dir /home/adminster/DYH/EAT/config \
    --config-name finetuning  \
    common.user_dir=/home/adminster/DYH/EAT \
    checkpoint.save_dir=/home/adminster/DYH/eatdata/results_20percent \
    checkpoint.restore_file=/home/adminster/DYH/eatdata/results_20percent/checkpoint_last.pt \
    checkpoint.best_checkpoint_metric=mAP \
    dataset.batch_size=32 \
    task.data=/home/adminster/DYH/EAT_manifest/AS20K_20percent \
    task.target_length=1024 \
    task.roll_aug=true \
    optimization.max_update=40000 \
    optimizer.groups.default.lr_scheduler.warmup_updates=4000 \
    model.model_path=/home/adminster/DYH/eatdata/EAT-base_epoch10_ft_AS20K.pt \
    model.num_classes=527 \
    model.mixup=0.8 \
    model.mask_ratio=0.2 \
    model.prediction_mode=PredictionMode.CLS_TOKEN \
    common.tensorboard_logdir=/home/adminster/DYH/eatdata/results_20percent/tb \
    2>&1 | tee /home/adminster/DYH/eatdata/results_20percent/train.log
