#!/usr/bin/env bash
CONF_FILE="pytorch/params/hyper_params.json"
MODEL="gn_pytorch"
MODEL_FN="/mnt/blobfuse/train-output/conditioning/models/backup_conditioning_groups/training/checkpoint_best.pth.tar"

python -u backend_server.py \
    --model ${MODEL} \
    --model_fn ${MODEL_FN}