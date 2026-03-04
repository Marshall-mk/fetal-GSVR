#!/bin/bash

tag=aicregistry:5000/${USER}:gsvr2
echo "Submitting jobs..."

runai submit            \
    --backoff-limit 0   \
    --name gsvr      \
    --image ${tag}      \
    --gpu 1 \
    --memory 64Gi \
    --node-type "A100" \
    --project ${USER}   \
    --volume /nfs:/nfs  \
    --working-dir /nfs/home/kmuhammad/code/fetal-GSVR \
    --large-shm         \
    --run-as-user       \
    --command -- python train.py --config configs/config_subjects_real.yaml --flags.use_masks False --training.batch_size 5000000