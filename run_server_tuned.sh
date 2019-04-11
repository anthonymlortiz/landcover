#!/usr/bin/env bash
MODEL="tuned_unet"
MODEL_FN="/mnt/blobfuse/train-output/conditioning/models/finetuning/finetuned_unet_gn_1000samples.pth.tar"

python -u backend_server.py \
    --model ${MODEL} \
    --model_fn ${MODEL_FN}