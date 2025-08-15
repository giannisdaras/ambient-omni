#!/bin/bash
PYTHONPATH=.

cats_dataset_path=./data/afhqv2-64x64-partitioned/0 # cats, replace with your own
dogs_dataset_path=./data/afhqv2-64x64-partitioned/1 # dogs, replace with your own
checkpoint_path=./outputs/ambient-syn-runs/afhq-64x64/crops_classifier/cats-100_and_dogs-100
checkpoint_path_4=${checkpoint_path}/crop-4/00000-cats-100_and_dogs-100-uncond-ddpmpp-edmcls-gpus8-batch512-fp32-56WOn/network-snapshot-020070.pkl
checkpoint_path_8=${checkpoint_path}/crop-8/00000-cats-100_and_dogs-100-uncond-ddpmpp-edmcls-gpus8-batch512-fp32-0A6eU/network-snapshot-020070.pkl
checkpoint_path_16=${checkpoint_path}/crop-16/00000-cats-100_and_dogs-100-uncond-ddpmpp-edmcls-gpus8-batch512-fp32-lkjKu/network-snapshot-020070.pkl
checkpoint_path_24=${checkpoint_path}/crop-24/00000-cats-100_and_dogs-100-uncond-ddpmpp-edmcls-gpus8-batch512-fp32-g6Cgu/network-snapshot-020070.pkl
checkpoint_paths=${checkpoint_path_4},${checkpoint_path_8},${checkpoint_path_16},${checkpoint_path_24}

# Wrangle paths auto
checkpoint_basename=$(basename "$checkpoint_path")
checkpoint_path_network_snapshot=$(basename "$checkpoint_path_4" .pkl)
annotated_dataset_path=./outputs/annotated_afhq/cat_is_dog/${checkpoint_basename}/${checkpoint_path_network_snapshot}

# Annotate cat crops
mkdir -p $annotated_dataset_path
torchrun --nproc_per_node=8 annotate_crops.py \
          --annotated_dataset_path=${annotated_dataset_path} \
          --inference_noise_config=identity \
          --corruption_probability=0 \
          --data=${cats_dataset_path} \
          --checkpoint_paths=${checkpoint_paths}

# Merge cat crops dataset with 10% dogs data
merged_dataset_path=./outputs/annotated_afhq/cats_help_dogs/${checkpoint_basename}/${checkpoint_path_network_snapshot}
./scripts/ood_dataset_utils/merge_datasets_for_train_ood_diffusion.sh ${merged_dataset_path} ${dogs_dataset_path}:10 ${annotated_dataset_path}:100