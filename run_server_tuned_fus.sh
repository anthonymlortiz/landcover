#!/usr/bin/env bash
MODEL="tuned_fusionnet"
MODEL_FN="/mnt/blobfuse/train-output/conditioning/models/finetuning/finetuned_fusionnet32_gn_100samples.pth.tar"

python -u backend_server.py \
    --model ${MODEL} \
    --model_fn ${MODEL_FN}