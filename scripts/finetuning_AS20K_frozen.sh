#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

# Create output directory
mkdir -p /home/adminster/DYH/eatdata/results_20percent_keephead_10ep

# Keep the original classifier head weights, freeze encoder, train head for 10 epochs
python /home/adminster/DYH/fairseq/fairseq_cli/hydra_train.py -m \
    --config-dir /home/adminster/DYH/EAT/config \
    --config-name finetuning  \
    common.user_dir=/home/adminster/DYH/EAT \
    checkpoint.save_dir=/home/adminster/DYH/eatdata/results_20percent_keephead_10ep \
    checkpoint.no_epoch_checkpoints=false \
    checkpoint.save_interval=1 \
    checkpoint.keep_best_checkpoints=1 \
    checkpoint.best_checkpoint_metric=mAP \
    checkpoint.maximize_best_checkpoint_metric=true \
    dataset.batch_size=64 \
    task.data=/home/adminster/DYH/EAT_manifest/AS20K_20percent \
    task.target_length=1024 \
    task.roll_aug=false \
    optimization.max_epoch=10 \
    optimizer.groups.default.lr_scheduler.warmup_updates=100 \
    dataset.num_workers=16 \
    dataset.valid_subset=eval \
    model.model_path=/home/adminster/DYH/eatdata/EAT-base_epoch10_ft_AS20K.pt \
    model.num_classes=527 \
    model.mixup=0.0 \
    model.mixup_prob=0.0 \
    model.specaug=false \
    model.mask_ratio=0.2 \
    model.prediction_mode=PredictionMode.CLS_TOKEN \
    +model.linear_classifier=true \
    +model.reset_classifier=false \
    common.tensorboard_logdir=/home/adminster/DYH/eatdata/results_20percent_keephead_10ep/tb \
    2>&1 | tee /home/adminster/DYH/eatdata/results_20percent_keephead_10ep/train.log
