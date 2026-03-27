#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SAVE_DIR=/home/adminster/DYH/eatdata/pretraining_AS20K_base
# RESTORE_FILE=/home/adminster/DYH/eatdata/pretraining_AS20K_base/checkpoint_13_35000.pt

mkdir -p "${SAVE_DIR}"

# 注意：不要在反斜杠续行的命令中间插入注释行，会导致后续参数被当成 shell 命令
python /home/adminster/DYH/fairseq/fairseq_cli/hydra_train.py -m \
    --config-dir /home/adminster/DYH/EAT/config \
    --config-name pretraining_AS20K \
    common.user_dir=/home/adminster/DYH/EAT \
    common.tensorboard_logdir="${SAVE_DIR}/tb" \
    checkpoint.save_dir="${SAVE_DIR}" \
    optimization.max_epoch=20 \
    optimization.max_update=1000000 \
    distributed_training.distributed_world_size=1 \
    dataset.batch_size=24 \
    task.data=/home/adminster/DYH/EAT_manifest/AS20K_local \
    task.h5_format=false \
    2>&1 | tee "${SAVE_DIR}/train.log"