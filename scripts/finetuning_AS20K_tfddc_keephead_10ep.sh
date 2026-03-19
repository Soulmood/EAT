#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p /home/adminster/DYH/eatdata/results_20percent_tfddc_keephead_10ep

python /home/adminster/DYH/fairseq/fairseq_cli/hydra_train.py -m \
    --config-dir /home/adminster/DYH/EAT/config \
    --config-name finetuning  \
    common.user_dir=/home/adminster/DYH/EAT \
    checkpoint.save_dir=/home/adminster/DYH/eatdata/results_20percent_tfddc_keephead_10ep \
    checkpoint.no_epoch_checkpoints=false \
    checkpoint.save_interval=1 \
    checkpoint.keep_best_checkpoints=1 \
    checkpoint.best_checkpoint_metric=mAP \
    checkpoint.maximize_best_checkpoint_metric=true \
    dataset.batch_size=2 \
    optimization.update_freq=[8] \
    dataset.num_workers=8 \
    task.data=/home/adminster/DYH/EAT_manifest/AS20K_20percent \
    task.target_length=1024 \
    task.roll_aug=false \
    dataset.valid_subset=eval \
    optimization.max_epoch=10 \
    optimizer.groups.default.lr_scheduler.warmup_updates=100 \
    model.model_path=/home/adminster/DYH/eatdata/EAT-base_epoch10_ft_AS20K.pt \
    model.num_classes=527 \
    model.mixup=0.0 \
    model.mixup_prob=0.0 \
    model.specaug=false \
    model.mask_ratio=0.2 \
    model.prediction_mode=PredictionMode.CLS_TOKEN \
    model.use_tfddc=true \
    model.tfddc_num_layers=2 \
    model.linear_classifier=true \
    model.train_local_encoder=true \
    model.reset_classifier=false \
    common.tensorboard_logdir=/home/adminster/DYH/eatdata/results_20percent_tfddc_keephead_10ep/tb \
    2>&1 | tee /home/adminster/DYH/eatdata/results_20percent_tfddc_keephead_10ep/train.log
