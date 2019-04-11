#!/usr/bin/env bash
CONF_FILE="pytorch/params/hyper_params.json"
MODEL="gn_fusionnet"
MODEL_FN="/mnt/blobfuse/train-output/conditioning/models/backup_fusionnet64_gn_runningstats8/training/checkpoint_best.pth.tar"

python -u backend_server.py \
    --model ${MODEL} \
    --model_fn ${MODEL_FN}