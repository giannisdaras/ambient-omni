#!/bin/bash
PYTHONPATH=.
ckpt_name=noise_classifier/blur0-6_prob0-5/00000-train-uncond-ddpmpp-edmcls-gpus8-batch512-fp32-TWeeb/network-snapshot-015053 # Replace with your own
annotated_datasets_path=./outputs/annotated_cifar10/$ckpt_name # Replace with your own
checkpoint_path=adrianrm/ambient-o-noise-classifier-blur06-prob05-iter15k # Using huggingface checkpoint
training_noise_config=blur0-6
inference_noise_config=blur0-6
corruption_probability=0.9

# Randomize torchrun master_port
MASTER_PORT=$(( ( RANDOM % 1000 )  + 10000 ))

export TORCH_NCCL_ENABLE_MONITORING=0
# export NCCL_BLOCKING_WAIT=0
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=72000

mkdir -p $annotated_datasets_path
torchrun --nproc_per_node=7 annotate.py \
    --annotated_dataset_path=${annotated_datasets_path} \
    --training_noise_config=${training_noise_config} \
    --inference_noise_config=${inference_noise_config} \
    --corruption_probability=${corruption_probability} \
    --checkpoint_path=${checkpoint_path}