#!/bin/bash
PYTHONPATH=.
outdir=./outputs
dataset_path=./outputs/annotated_afhq/cats_help_dogs/cats-100_and_dogs-100/network-snapshot-020070 # Replace with your own; path to annotated dataset
s_max=4
dp=1.0
weight_decay=0.0
cls_epsilon=0.05

extracted_path=$(echo "$dataset_path" | sed 's/.*annotated_afhq\///')
outdir=${outdir}/ambient-syn-runs/afhq-64x64/ood_diffusion/${extracted_path}/s-max-${s_max}
config_part=$(basename "$(dirname "$dataset_path")")

# Randomize torchrun master_port
MASTER_PORT=$(( ( RANDOM % 1000 )  + 10000 ))

torchrun --master_port $MASTER_PORT --nproc_per_node=8 train.py \
            --outdir=${outdir} \
            --data=${dataset_path} \
            --cond=0 --arch=ddpmpp --dump=40 --duration=200 \
            --precond=edm \
            --corruption_probability=0.0 \
            --dataset_keep_percentage=${dp} \
            --weight_decay=${weight_decay} \
            --workers 2 \
            --cls_epsilon=${cls_epsilon} \
            --expr_id=train_ood_diffusion_afhq_cats_help_dogs_${config_part}_s-max-${s_max}_dp${dp}_wd${weight_decay} \
            --s_max=${s_max}