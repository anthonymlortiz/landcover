#!/usr/bin/env bash
CONF_FILE="pytorch/params/hyper_params.json"
MODEL="cfusionnet"
PORT=4445
MODEL_FN="/mnt/blobfuse/train-output/conditioning/models/backup_conditional_superres512/training/checkpoint_best.pth.tar"

python -u backend_server.py \
    --model ${MODEL} \
    --port ${PORT}\
    --model_fn ${MODEL_FN}