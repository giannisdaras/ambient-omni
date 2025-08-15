#!/bin/bash
PYTHONPATH=.
annotated_datasets_path=./outputs/annotated_cifar10/annotated_blur0-8_fixed_1-89/
training_noise_config=blur0-8
inference_noise_config=blur0-8
corruption_probability=0.9
min_sigma=1.89

# Randomize torchrun master_port
MASTER_PORT=$(( ( RANDOM % 1000 )  + 10000 ))

export TORCH_NCCL_ENABLE_MONITORING=0
# export NCCL_BLOCKING_WAIT=0
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=72000

torchrun --nproc_per_node=1 annotate_fixed_sigma.py \
    --annotated_dataset_path=${annotated_datasets_path} \
    --dataset_path=./data/cifar10/train \
    --inference_noise_config=${inference_noise_config} \
    --corruption_probability=${corruption_probability} \
    --min_fixed_sigma=$min_sigma \
    --max_fixed_sigma=0