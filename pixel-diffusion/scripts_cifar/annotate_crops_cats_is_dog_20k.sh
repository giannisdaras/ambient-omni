#!/bin/bash
PYTHONPATH=.

inference_noise_config=identity
corruption_probability=0
dataset_path=/data/vision/torralba/selfmanaged/torralba/projects/adrianr/ambient/scratch_ambient/ambient/datasets/afhqv2-64x64-partitioned/0
annotated_dataset_path=/data/vision/torralba/selfmanaged/torralba/projects/adrianr/ambient/data_softlink/annotated_afhq_classifier/cats_is_dogs/20k

checkpoint_path_4=/data/vision/torralba/selfmanaged/torralba/projects/adrianr/ambient/pixel-diffusion/ambient-syn-runs/afhq-64x64/train_cls_crops_dogs-100_and_cats-100/crop-4/00024-dogs-100_and_cats-100-uncond-ddpmpp-edmcls-half-gpus8-batch512-fp32-D1Y4f/network-snapshot-020070.pkl
checkpoint_path_8=/data/vision/torralba/selfmanaged/torralba/projects/adrianr/ambient/pixel-diffusion/ambient-syn-runs/afhq-64x64/train_cls_crops_dogs-100_and_cats-100/crop-8/00004-dogs-100_and_cats-100-uncond-ddpmpp-edmcls-half-gpus8-batch512-fp32-OmmFx/network-snapshot-020070.pkl
checkpoint_path_16=/data/vision/torralba/selfmanaged/torralba/projects/adrianr/ambient/pixel-diffusion/ambient-syn-runs/afhq-64x64/train_cls_crops_dogs-100_and_cats-100/crop-16/00000-dogs-100_and_cats-100-uncond-ddpmpp-edmcls-half-gpus8-batch512-fp32-soL5l/network-snapshot-020070.pkl
checkpoint_path_24=/data/vision/torralba/selfmanaged/torralba/projects/adrianr/ambient/pixel-diffusion/ambient-syn-runs/afhq-64x64/train_cls_crops_dogs-100_and_cats-100/crop-24/00000-dogs-100_and_cats-100-uncond-ddpmpp-edmcls-half-gpus8-batch512-fp32-jkNax/network-snapshot-020070.pkl
checkpoint_paths=${checkpoint_path_4},${checkpoint_path_8},${checkpoint_path_16},${checkpoint_path_24}

torchrun --nproc_per_node=7 annotate_crops.py \
          --annotated_dataset_path=${annotated_dataset_path} \
          --inference_noise_config=${inference_noise_config} \
          --corruption_probability=${corruption_probability} \
          --data=${dataset_path} \
          --checkpoint_paths=${checkpoint_paths} \
          --flip_probs

./slurm_jobs/slurm/merge_datasets.sh /data/vision/torralba/selfmanaged/torralba/projects/adrianr/ambient/data_softlink/annotated_afhq_classifier/cats100_help_dogs10_annotated/20k /data/vision/torralba/selfmanaged/torralba/projects/adrianr/ambient/scratch_ambient/ambient/datasets/afhqv2-64x64-partitioned/1:10 ${annotated_dataset_path}:100