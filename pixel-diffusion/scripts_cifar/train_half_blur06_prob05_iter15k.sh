#!/bin/bash
PYTHONPATH=.
s_max=4
mode=square
outdir=/data/vision/torralba/selfmanaged/torralba/scratch/adrianr/ambient/ambient-syn-runs/cifar/aaa-test_half_blur06_prob05/network-snapshot-015053
dataset_path=/data/vision/torralba/selfmanaged/torralba/projects/adrianr/ambient/data_softlink/annotated_cifar10_classifier/classifier_half_blur06_prob05/00001-train-uncond-ddpmpp-edmcls-half-gpus8-batch512-fp32-ntURX/network-snapshot-015053
dp=1.0
weight_decay=0.0
cls_epsilon=0.05


# Randomize torchrun master_port
MASTER_PORT=$(( ( RANDOM % 1000 )  + 10000 ))

torchrun --master_port $MASTER_PORT --nproc_per_node=8 train.py \
            --outdir=${outdir} \
            --data=${dataset_path} \
            --cond=0 --arch=ddpmpp --dump=50 --duration=200 \
            --precond=edm \
            --corruption_probability=0.0 \
            --dataset_keep_percentage=${dp} \
            --weight_decay=${weight_decay} \
            --workers 2 \
            --cls_epsilon=${cls_epsilon} \
            --expr_id=train_net_cifar10-corrupted-half_blur06_prob05_dp${dp}_wd${weight_decay}_s-max-${s_max}_mode${mode} \
            --s_max=${s_max} \
            --ambient_weight_mode=${mode}