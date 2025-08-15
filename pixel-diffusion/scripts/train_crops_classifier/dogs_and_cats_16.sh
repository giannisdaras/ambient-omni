#!/bin/bash
PYTHONPATH=.
outdir=./outputs
dataset_path=./data/cats-100_and_dogs-100
crop_size=16
dp=1.0
weight_decay=0.0

# Add args to outdir
dataset_name=$(basename "$dataset_path")
outdir=${outdir}/ambient-syn-runs/afhq-64x64/crops_classifier/${dataset_name}/crop-${crop_size}

# Randomize torchrun master_port
MASTER_PORT=$(( ( RANDOM % 1000 )  + 10000 ))

mkdir -p $outdir
torchrun --master_port $MASTER_PORT --nproc_per_node=8 train.py \
            --outdir=${outdir} \
            --data=${dataset_path} \
            --cond=0 --arch=ddpmpp --dump=50 --duration=200 \
            --precond=edmcls \
            --noise_config=identity \
            --corruption_probability=0 \
            --dataset_keep_percentage=${dp} \
            --weight_decay=${weight_decay} \
            --overwrite_cls_labels_path=${dataset_path}/labels.jsonl \
            --crop_size=${crop_size} \
            --expr_id=train_crops_classifier_dogs_and_cats_dp${dp}_wd${weight_decay} \
            --workers=2