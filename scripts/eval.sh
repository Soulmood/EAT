#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

python /home/adminster/DYH/EAT/evaluation/eval.py  \
    --label_file='/home/adminster/DYH/EAT/inference/labels.csv' \
    --eval_dir='/home/adminster/DYH/EAT_manifest/AS20K_local' \
    --model_dir='/home/adminster/DYH/EAT' \
    --checkpoint_dir='/home/adminster/DYH/eatdata/finetuning_AS20K_base/base_pretrain_finetune107.pt' \
    --target_length=1024 \
    --device='cuda' \
    --batch_size=256 \
    --ap_log_path='/home/adminster/DYH/eatdata/finetuning_AS20K_base/base_pretrain_finetune107_AS20K_local.txt'

# For optimal performance, 1024 is recommended for 10-second audio clips. (128 for 1-second)
# However, you should adjust the target_length parameter based on the duration and characteristics of your specific audio inputs.
# EAT-finetuned could make evaluation well even given truncated audio clips.
